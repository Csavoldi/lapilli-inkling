// src/models/inkling2/adapter.cpp - the inkling2 adapter: pack validation, geometry, loading and the tokenizer
// seams. The model rides the shared engine (prefill_chunk / decode_window); it needs no start of its own.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/model.hpp"
#include "basalt/models/inkling2/pretok.hpp"

#include <memory>
#include <string>

namespace basalt::models::inkling2 {
namespace {

/// The architecture name in the looped GGUFs (and so in a pack's metadata and the server's dialect lookup).
constexpr const char* kGgufArch = "inkling2_loop";

using model::ModelAdapter;
using model::ModelDesc;

bool kv_i64(const GgufFile& g, const char* key, int64_t& out) {
    const MetaValue* v = g.get(key);
    if (!v) return false;
    out = (int64_t) v->u;
    return true;
}

class Adapter final : public ModelAdapter {
public:
    explicit Adapter(const char* arch) : arch_(arch) {}
    const char* arch() const override { return arch_; }

    std::string validate(const GgufFile& meta) const override {
        const MetaValue* a = meta.get("general.architecture");
        if (!a || (a->s != "inkling2" && a->s != kGgufArch))
            return "not an Inkling 2 GGUF (general.architecture is neither inkling2 nor " + std::string(kGgufArch) + ")";
        const std::string p = a->s + ".";
        int64_t blocks = 0, start = 0, end = 0, count = 0, rank = 0;
        if (!kv_i64(meta, (p + "block_count").c_str(), blocks) || blocks != kLayersPhys)
            return "inkling2: the GGUF does not carry " + std::to_string(kLayersPhys) + " physical layers";
        if (!kv_i64(meta, (p + "loop.start").c_str(), start) || !kv_i64(meta, (p + "loop.end").c_str(), end) ||
            !kv_i64(meta, (p + "loop.count").c_str(), count) || !kv_i64(meta, (p + "loop.gate_rank").c_str(), rank))
            return "inkling2: the loop keys are missing";
        if (start != kCoreStart || end != kCoreEnd || count != kPasses || rank != kGateRank)
            return "inkling2: a loop of [" + std::to_string(start) + ", " + std::to_string(end) + ") x" +
                   std::to_string(count) + " rank " + std::to_string(rank) + "; this engine reads [" +
                   std::to_string(kCoreStart) + ", " + std::to_string(kCoreEnd) + ") x" +
                   std::to_string(kPasses) + " rank " + std::to_string(kGateRank);
        const MetaValue* pattern = meta.get(p + "attention.sliding_window_pattern");
        if (!pattern || pattern->count != kLayersPhys)
            return "inkling2: the sliding-window pattern is missing";
        return "";
    }

    ModelDesc describe(const GgufFile& meta) const override {
        ModelDesc d;
        d.arch = "inkling2";
        d.n_layers = kLayersPhys;
        d.n_embd = kEmbd;
        d.n_vocab = kVocab;
        d.n_expert = kExperts;
        d.n_expert_used = kTopK;
        d.n_ff_exp = kFf;
        d.context_length = 1048576;
        return d;
    }

    model::Result load(const model::LoadRequest& req, std::unique_ptr<model::LoadedModel>& out) const override {
        if (req.paths.empty()) return {model::Status::Error, "no model path"};
        std::string err;
        std::unique_ptr<Weights> w;
        if (!Weights::load(req.paths.front(), req.max_context, w, err)) return {model::Status::Error, err};
        std::unique_ptr<LoadedInkling> m(new LoadedInkling(std::move(w)));
        m->desc_ = describe(GgufFile(req.paths.front()));
        m->dump_dir = req.diag_dump;
        out = std::move(m);
        return {};
    }

    model::TokenizerSpec tokenizer_spec() const override { return {"tokenizer.ggml.tokens", "inkling2"}; }
    model::TokenNames token_names() const override {
        model::TokenNames n;
        n.eos = {"<|content_model_end_sampling|>", "<|end_message|>"};
        n.turn = "<|message_user|>";
        n.reply_role = "";
        return n;
    }
    PreTokenizer pretokenizer() const override { return {"inkling2", pretok_inkling2}; }
    program::grammar::ToolForm tool_form() const override { return {}; }
    int max_expert_cards() const override { return 0; }

private:
    const char* arch_;
};

}  // namespace

// The model is `inkling2`; the looped GGUFs name their architecture `inkling2_loop`, which is also the name the
// pack carries and the registry and the server look a model up by. So the factory registers the adapter under
// that name too: one library, both spellings.
std::unique_ptr<model::ModelAdapter> make_adapter() {
    if (model::find_adapter(kGgufArch) == nullptr) model::register_adapter(std::make_unique<Adapter>(kGgufArch));
    return std::make_unique<Adapter>("inkling2");
}

}  // namespace basalt::models::inkling2
