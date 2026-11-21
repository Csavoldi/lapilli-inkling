// src/models/inkling2/pretok_inkling2.cpp - the inkling2 pre-tokenizer: the text split into the pieces BPE
// merges. The rules are llama.cpp's GPT-4O tokenizer split (src/llama-vocab.cpp, pre type GPT4O, clean_spaces
// left false), which the GGUF's tokenizer.ggml.pre = inkling2 asks for:
//   prefix? upper* lower+ ('s|'t|'re|'ve|'m|'ll|'d)?  |  prefix? upper+ lower*  |  digits{1,3}
//   | ' '? non-space-letter-number+ [\r\n/]*  |  \s*[\r\n]+  |  \s+(?!\S)  |  \s+
// with prefix = [^\r\n\p{L}\p{N}], "upper" = letter-or-mark outside ASCII a-z and "lower" = letter-or-mark
// outside ASCII A-Z (they overlap off ASCII, as in the original). The code points are decoded once, with their
// flags, and the pieces come back as byte offsets into the UTF-8 text.
#include "basalt/models/inkling2/pretok.hpp"

#include "unicode_flags.hpp"

namespace basalt::models::inkling2 {
namespace {

namespace tok = basalt::program::tokenizer;
using tok::decode_cp;
using tok::cpt_flag_table;
constexpr uint16_t kNumber = tok::kNumber;
constexpr uint16_t kLetter = tok::kLetter;
constexpr uint16_t kMark = tok::kMark;
constexpr uint16_t kWhitespace = tok::kWhitespace;

using Flags = uint16_t;

bool is_letter_mark(uint32_t cp, Flags f) { (void) cp; return (f & (kLetter | kMark)) != 0; }
bool is_upper_ish(uint32_t cp, Flags f) { return is_letter_mark(cp, f) && !(cp >= 'a' && cp <= 'z'); }
bool is_lower_ish(uint32_t cp, Flags f) { return is_letter_mark(cp, f) && !(cp >= 'A' && cp <= 'Z'); }
bool is_number(Flags f) { return (f & kNumber) != 0; }
bool is_space(Flags f) { return (f & kWhitespace) != 0; }
bool is_prefixable(uint32_t cp, Flags f) {
    return !(cp == '\r' || cp == '\n') && !is_letter_mark(cp, f) && !is_number(f);
}

/// The optional contraction after a word: (?i:'s|'t|'re|'ve|'m|'ll|'d).
size_t contraction_len(const uint32_t* cps, size_t n, size_t i) {
    if (i >= n || cps[i] != '\'') return 0;
    auto one = [&](uint32_t c) { return i + 1 < n && (cps[i + 1] == c || cps[i + 1] == c - ('a' - 'A')); };
    if (one('s') || one('t') || one('m') || one('d')) return 2;
    if (i + 2 < n) {
        const uint32_t a = cps[i + 1], b = cps[i + 2];
        const auto eq = [](uint32_t x, uint32_t c) { return x == c || x == c - ('a' - 'A'); };
        if ((eq(a, 'r') && eq(b, 'e')) || (eq(a, 'v') && eq(b, 'e'))) return 3;
        if (eq(a, 'l') && eq(b, 'l')) return 3;
    }
    return 0;
}

}  // namespace

void pretok_inkling2(const char* text, size_t size, std::vector<size_t>& ends) {
    ends.clear();
    const uint8_t* s = (const uint8_t*) text;
    const uint16_t* table = cpt_flag_table();
    std::vector<uint32_t> cps;
    std::vector<Flags> fl;
    std::vector<size_t> off;
    cps.reserve(size);
    fl.reserve(size);
    off.reserve(size + 1);
    for (size_t i = 0; i < size;) {
        size_t len = 0;
        const uint32_t cp = decode_cp(s, size, i, len);
        cps.push_back(cp);
        fl.push_back(table[cp < 0x110000 ? cp : 0xFFFD]);
        off.push_back(i);
        i += len;
    }
    const size_t n = cps.size();
    off.push_back(size);

    size_t at = 0;
    while (at < n) {
        const size_t i = at;
        // alternations 1 and 2: a word with its optional prefix and optional contraction
        size_t p = i;
        if (p < n && is_prefixable(cps[p], fl[p])) ++p;
        size_t up = p;
        while (up < n && is_upper_ish(cps[up], fl[up])) ++up;
        size_t lo = up;
        while (lo < n && is_lower_ish(cps[lo], fl[lo])) ++lo;
        size_t word_end = 0;
        if (lo > up) word_end = lo;                 // prefix? upper* lower+
        else if (up > p) word_end = up;             // prefix? upper+ lower*
        if (word_end > p) {
            const size_t e = word_end + contraction_len(cps.data(), n, word_end);
            at = e;
            ends.push_back(off[e]);
            continue;
        }
        // alternation 3: one to three digits
        if (is_number(fl[i])) {
            size_t e = i + 1;
            while (e < n && e < i + 3 && is_number(fl[e])) ++e;
            at = e;
            ends.push_back(off[e]);
            continue;
        }
        // alternation 4: an optional space, then a run of symbols with trailing newlines and slashes
        {
            size_t q = i;
            if (q < n && cps[q] == ' ') ++q;
            if (q < n && !is_space(fl[q]) && !is_letter_mark(cps[q], fl[q]) && !is_number(fl[q])) {
                size_t e = q + 1;
                while (e < n && !is_space(fl[e]) && !is_letter_mark(cps[e], fl[e]) && !is_number(fl[e])) ++e;
                while (e < n && (cps[e] == '\r' || cps[e] == '\n' || cps[e] == '/')) ++e;
                at = e;
                ends.push_back(off[e]);
                continue;
            }
        }
        // alternations 5 to 7: whitespace; \s*[\r\n]+ first (up to and including its first newline run),
        // then the trailing-whitespace arm, then plain \s+
        if (is_space(fl[i])) {
            size_t e = i;
            while (e < n && is_space(fl[e])) ++e;
            size_t m = 0;
            for (size_t q = i; q < e; ++q)
                if (cps[q] == '\r' || cps[q] == '\n') {
                    m = q + 1;
                    while (m < e && (cps[m] == '\r' || cps[m] == '\n')) ++m;
                    break;
                }
            const size_t take = m > 0 ? m : e;
            at = take;
            ends.push_back(off[take]);
            continue;
        }
        // a lone prefixable character: its own piece
        at = i + 1;
        ends.push_back(off[i + 1]);
    }
    if (ends.empty() || ends.back() != size) ends.push_back(size);
}

}  // namespace basalt::models::inkling2
