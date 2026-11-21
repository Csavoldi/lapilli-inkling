// include/basalt/models/inkling2/pretok.hpp - the inkling2 pre-tokenizer's split (the GPT-4O rules the GGUF's
// tokenizer.ggml.pre asks for; see src/models/inkling2/pretok_inkling2.cpp).
#pragma once

#include <cstddef>
#include <vector>

namespace basalt::models::inkling2 {

void pretok_inkling2(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace basalt::models::inkling2
