// include/basalt/models/inkling2/session.hpp - one conversation of Inkling 2 Mini: KV pools per loop slot,
// the short-convolution state per slot, and the window buffers the layer walk reuses.
#pragma once

#include "basalt/model/adapter.hpp"
#include "basalt/models/inkling2/geometry.hpp"
#include "basalt/models/inkling2/model.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

class Session : public model::ModelSession {
public:
    explicit Session(const Weights& w, std::string dump_dir = {});
    ~Session() override;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    bool init(std::string& err);

    model::Result decode_window(const model::DecodeRequest& r) override;
    model::Result prefill_chunk(const model::PrefillRequest& r) override;
    int max_window() const override { return 1; }
    model::Result set_history(const int32_t* tokens, int64_t n) override { (void) tokens; (void) n; return {}; }

private:
    /// The whole layer walk over `T` rows at absolute positions pos[0..T): the shared path of decode and prefill.
    /// `wants_logits` adds the output mix, norm and the lm_head into `logits_` (decode only).
    model::Result run_rows(const int32_t* tokens_host, const int64_t* pos_host, int T, bool wants_logits);
    /// One layer application `il` (its weights are layer_phys(il)'s); advances the block bookkeeping.
    void layer(int64_t il, int T);
    /// --diag-dump: write a device buffer (floats) as <dir>/<name>.bin; no-op without a directory.
    void dump_f32(const char* step, int64_t il, const void* dev, int64_t floats);
    /// The same, for one column of a [T, cols, kEmbd] buffer: rows of kEmbd floats `pitch` floats apart.
    void dump_f32_col(const char* step, int64_t il, const void* dev, int64_t pitch, int64_t floats_per_row, int rows);
    void dump_i32_host(const char* step, int64_t il, const void* host, int64_t values);

    const Weights& w_;
    void* stream_ = nullptr;
    std::vector<void*> device_allocs_;
    std::string dump_dir_;
    void* dump_manifest_ = nullptr;       // FILE*, opaque here to keep the header clean of <cstdio>

    // per-slot state
    uint16_t* k_pools_ = nullptr;          // one pool per slot, [cap][8][128]
    uint16_t* v_pools_ = nullptr;
    std::vector<uint64_t> pool_base_;      // the slot's first cell, in values
    std::vector<int64_t> caps_host_;       // the slot's pool capacity
    float* conv_state_ = nullptr;          // [slot][3][kConvChannels]
    int64_t pos_next_ = 0;

    // window buffers (sized for the largest prefill chunk; decode rows reuse them)
    static constexpr int kMaxBlocks = 6;   // b0, b6, b12, b18 (+ cur in the mix); the output mix sees 5 states
    int cap_rows_ = 0;
    int32_t* d_tokens_ = nullptr;
    int64_t* d_pos_ = nullptr;
    float* d_tau_ = nullptr;
    int32_t* d_out_ = nullptr;
    float* x_ = nullptr;                   // [T, 2304] the running activations
    float* inpSA_ = nullptr;               // [T, 2304] saved before each layer's attention
    float* mixed_ = nullptr;               // [T, 3, 2304] the three attention inputs
    float* msgs_ = nullptr;                // [4, T, 2304] the loop messages (Q, K, V, FFN)
    float* q_ = nullptr;                   // [T, 3072]
    float* k_ = nullptr;                   // [T, 1024]
    float* v_ = nullptr;                   // [T, 1024]
    float* r_ = nullptr;                   // [T, 384]
    float* sinks_ = nullptr;               // [T, 24]
    float* table_ = nullptr;               // [T, kRelExtent+1, 24]
    float* attn_out_ = nullptr;            // [T, 3072]
    float* gate_buf_ = nullptr;            // [T, 3072]
    float* wo_buf_ = nullptr;              // [T, 2304]
    float* conv_kv_ = nullptr;             // [2, T, 1024] K and V after their convolutions
    float* conv2304_ = nullptr;            // [2, T, 2304] the post-attention and post-FFN convolutions
    float* dyn_ = nullptr;                 // [2, T, 8, 4] the K and V dynamic taps
    float* ffn_x_ = nullptr;               // [T, 2304]
    float* router_l_ = nullptr;            // [T, 128]
    int32_t* expert_ids_ = nullptr;        // [T, 8]
    float* expert_w_ = nullptr;            // [T, 8]
    float* gu_ = nullptr;                  // [1, 2*kFf]
    float* h_ = nullptr;                   // [1, kFf]
    float* parts_ = nullptr;               // [T, 8, 2304]
    uint8_t* scratch_q8_1_ = nullptr;      // xq [T,2304], wo [T,3072] and the expert h rows
    float* logits_ = nullptr;              // [T, kVocabPadded]
    float* blocks_ = nullptr;              // [T, kMaxBlocks, 2304] the pushed block activations
    float* bank_ = nullptr;                // [T, 2, 2304] the previous pass's block and output
    float* prelude_ = nullptr;             // [T, 2304]
    int n_blocks_ = 0;                     // host-side block bookkeeping for the current pass

    // host staging
    int32_t* h_ids_ = nullptr;             // pinned [cap_rows_ * 8]
    int64_t* h_pos_ = nullptr;             // pinned [cap_rows_]
    float* h_tau_ = nullptr;               // pinned [cap_rows_]
    float* h_logits_ = nullptr;            // pinned [kVocab]
    int32_t* h_out_ = nullptr;             // pinned [1]
};

}  // namespace basalt::models::inkling2
