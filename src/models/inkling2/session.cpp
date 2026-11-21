// src/models/inkling2/session.cpp - the layer walk of Inkling 2 Mini: one function shared by prefill and
// decode, the exact op order of the llama.cpp reference (models/inkling2 at 29b52d3 + pull 32704).
#include "basalt/models/inkling2/session.hpp"

#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

namespace basalt::models::inkling2 {
namespace {

namespace sk = ::basalt::kernels;      // the shared kernels; `kernels` below is this model's own

const Tensor* need(const Weights& w, const std::string& name) {
    const Tensor* t = w.find(name);
    if (!t) log::error("basalt inkling2", "the pack has no %s", name.c_str());
    return t;
}

/// The bytes of one row of `n_in` values in a ggml block type (Q8_0 or Q4_K here).
int64_t row_bytes(int type, int64_t n_in) {
    if (type == 8) return n_in / 32 * 34;
    if (type == 12) return n_in / 256 * 144;
    return -1;
}

}  // namespace

Session::Session(const Weights& w, std::string dump_dir) : w_(w), dump_dir_(std::move(dump_dir)) {}

Session::~Session() {
    if (stream_) cudaStreamDestroy((cudaStream_t) stream_);
    for (void* p : device_allocs_) cudaFree(p);
    if (dump_manifest_) std::fclose((std::FILE*) dump_manifest_);
}

// ------------------------------------------------------------------ the --diag-dump trace
void Session::dump_f32(const char* step, int64_t il, const void* dev, int64_t floats) {
    if (dump_dir_.empty()) return;
    std::vector<float> host((size_t) floats);
    cudaStream_t s = (cudaStream_t) stream_;
    cudaStreamSynchronize(s);
    cudaMemcpy(host.data(), dev, (size_t) floats * 4, cudaMemcpyDeviceToHost);
    char name[96], path[512];
    if (il >= 0) std::snprintf(name, sizeof name, "s%02lld.%s", (long long) il, step);
    else std::snprintf(name, sizeof name, "%s", step);
    std::snprintf(path, sizeof path, "%s/%s.bin", dump_dir_.c_str(), name);
    std::FILE* f = std::fopen(path, "wb");
    if (f) {
        std::fwrite(host.data(), 4, (size_t) floats, f);
        std::fclose(f);
    }
    if (dump_manifest_) std::fprintf((std::FILE*) dump_manifest_, "%s %lld\n", name, (long long) floats);
}

void Session::dump_f32_col(const char* step, int64_t il, const void* dev, int64_t pitch, int64_t row_floats,
                           int rows) {
    if (dump_dir_.empty()) return;
    std::vector<float> host((size_t) pitch * rows);
    cudaStream_t s = (cudaStream_t) stream_;
    cudaStreamSynchronize(s);
    cudaMemcpy2D(host.data(), row_floats * 4, dev, pitch * 4, row_floats * 4, rows, cudaMemcpyDeviceToHost);
    char name[96], path[512];
    if (il >= 0) std::snprintf(name, sizeof name, "s%02lld.%s", (long long) il, step);
    else std::snprintf(name, sizeof name, "%s", step);
    std::snprintf(path, sizeof path, "%s/%s.bin", dump_dir_.c_str(), name);
    std::FILE* f = std::fopen(path, "wb");
    if (f) {
        for (int r = 0; r < rows; ++r) std::fwrite(host.data() + r * row_floats, 4, (size_t) row_floats, f);
        std::fclose(f);
    }
    if (dump_manifest_) std::fprintf((std::FILE*) dump_manifest_, "%s %lld\n", name, (long long) (row_floats * rows));
}

void Session::dump_i32_host(const char* step, int64_t il, const void* host, int64_t values) {
    if (dump_dir_.empty()) return;
    char name[96], path[512];
    if (il >= 0) std::snprintf(name, sizeof name, "s%02lld.%s", (long long) il, step);
    else std::snprintf(name, sizeof name, "%s", step);
    std::snprintf(path, sizeof path, "%s/%s.bin", dump_dir_.c_str(), name);
    std::FILE* f = std::fopen(path, "wb");
    if (f) {
        std::fwrite(host, 4, (size_t) values, f);
        std::fclose(f);
    }
    if (dump_manifest_) std::fprintf((std::FILE*) dump_manifest_, "%s %lld\n", name, (long long) values);
}

bool Session::init(std::string& err) {
    if (cudaStreamCreateWithFlags((cudaStream_t*) &stream_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "cudaStreamCreate failed";
        return false;
    }
    cudaStream_t s = (cudaStream_t) stream_;
    // the shared MMVQ kernels take 1..8 activation rows: a prompt runs in windows of this many tokens
    cap_rows_ = 8;
    if (!dump_dir_.empty()) {
        ::mkdir(dump_dir_.c_str(), 0755);
        char path[512];
        std::snprintf(path, sizeof path, "%s/manifest.txt", dump_dir_.c_str());
        dump_manifest_ = std::fopen(path, "w");
    }

    // per-slot KV pools and convolution state
    caps_host_.resize(kSlots);
    int64_t cells = 0;
    for (int64_t il = 0; il < kSlots; ++il) {
        caps_host_[(size_t) il] = slot_kv_cap(il, w_.context());
        cells += caps_host_[(size_t) il];
    }
    const uint64_t pool_bytes = (uint64_t) cells * kEmbdK * 2 * 2;   // K and V, fp16
    uint16_t* pools = nullptr;
    if (cudaMalloc(&pools, pool_bytes) != cudaSuccess) {
        err = "the KV pools (" + std::to_string(pool_bytes >> 20) + " MiB) do not fit on the device";
        return false;
    }
    device_allocs_.push_back(pools);
    k_pools_ = pools;
    v_pools_ = pools + (uint64_t) cells * kEmbdK;
    pool_base_.resize(kSlots);
    {
        int64_t at = 0;
        for (int64_t il = 0; il < kSlots; ++il) {
            pool_base_[(size_t) il] = (uint64_t) at * kEmbdK;
            at += caps_host_[(size_t) il];
        }
    }
    float* conv = nullptr;
    const uint64_t conv_bytes = (uint64_t) kSlots * (kConv - 1) * kConvChannels * 4;
    if (cudaMalloc(&conv, conv_bytes) != cudaSuccess) {
        err = "the convolution state does not fit on the device";
        return false;
    }
    device_allocs_.push_back(conv);
    conv_state_ = conv;
    cudaMemsetAsync(conv_state_, 0, conv_bytes, s);

    // the window buffers, each grabbed in turn
    const int64_t R = cap_rows_;
    const int64_t f4 = R * 4;
    const auto grab = [&](void** p, uint64_t bytes) -> bool {
        if (bytes == 0) return true;
        if (cudaMalloc(p, bytes) != cudaSuccess) {
            err = "a session buffer of " + std::to_string(bytes >> 20) + " MiB does not fit";
            return false;
        }
        device_allocs_.push_back(*p);
        return true;
    };
    const uint64_t q8_1_bytes = (uint64_t) ((int64_t) R * (kEmbd / 32 * 36) + (int64_t) R * (3 * kEmbd / 32 * 36) +
                                            (int64_t) R * (kFf / 32 * 36) + 4096);
    const bool buffers =
        grab((void**) &scratch_q8_1_, q8_1_bytes) && grab((void**) &d_tokens_, (uint64_t) f4) &&
        grab((void**) &d_pos_, (uint64_t) (R * 8)) && grab((void**) &d_tau_, (uint64_t) f4) &&
        grab((void**) &d_out_, 4) && grab((void**) &x_, (uint64_t) (f4 * kEmbd)) &&
        grab((void**) &inpSA_, (uint64_t) (f4 * kEmbd)) && grab((void**) &mixed_, (uint64_t) (f4 * 3 * kEmbd)) &&
        grab((void**) &msgs_, (uint64_t) (f4 * 4 * kEmbd)) &&
        grab((void**) &q_, (uint64_t) (f4 * kHeads * kHeadDim)) && grab((void**) &k_, (uint64_t) (f4 * kEmbdK)) &&
        grab((void**) &v_, (uint64_t) (f4 * kEmbdK)) && grab((void**) &r_, (uint64_t) (f4 * kRelLen * kHeads)) &&
        grab((void**) &sinks_, (uint64_t) (f4 * kHeads)) && grab((void**) &table_, (uint64_t) (f4 * (kRelExtent + 1) * kHeads)) &&
        grab((void**) &attn_out_, (uint64_t) (f4 * kHeads * kHeadDim)) &&
        grab((void**) &gate_buf_, (uint64_t) (f4 * kHeads * kHeadDim)) && grab((void**) &wo_buf_, (uint64_t) (f4 * kEmbd)) &&
        grab((void**) &conv_kv_, (uint64_t) (f4 * 2 * kEmbdK)) && grab((void**) &conv2304_, (uint64_t) (f4 * 2 * kEmbd)) &&
        grab((void**) &dyn_, (uint64_t) (f4 * 2 * kConv * kHeadsKv)) && grab((void**) &ffn_x_, (uint64_t) (f4 * kEmbd)) &&
        grab((void**) &router_l_, (uint64_t) (R * kExperts * 4)) && grab((void**) &expert_ids_, (uint64_t) (R * kTopK * 4)) &&
        grab((void**) &expert_w_, (uint64_t) (R * kTopK * 4)) && grab((void**) &gu_, (uint64_t) (2 * kFf * 4)) &&
        grab((void**) &h_, (uint64_t) (kFf * 4)) && grab((void**) &parts_, (uint64_t) (R * kTopK * kEmbd * 4)) &&
        grab((void**) &logits_, (uint64_t) (8 * kVocabPadded * 4)) &&
        grab((void**) &blocks_, (uint64_t) (f4 * kMaxBlocks * kEmbd)) && grab((void**) &bank_, (uint64_t) (f4 * 2 * kEmbd)) &&
        grab((void**) &prelude_, (uint64_t) (f4 * kEmbd));
    if (!buffers) return false;
    if (cudaHostAlloc(&h_ids_, (size_t) R * kTopK * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_pos_, (size_t) R * 8, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_tau_, (size_t) R * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_logits_, (size_t) kVocab * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&h_out_, 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "the pinned staging buffers failed";
        return false;
    }
    log::info("basalt inkling2", "session ready: %lld cells, chunk %d", (long long) cells, cap_rows_);
    return true;
}

// ------------------------------------------------------------------ the layer walk
void Session::layer(int64_t il, int T) {
    cudaStream_t s = (cudaStream_t) stream_;
    const int64_t phys = layer_phys(il);
    const int64_t ip = layer_ip(il);
    const bool swa = layer_is_swa(phys);
    const bool looped = layer_pass(il) > 0;
    const std::string b = "blk." + std::to_string(phys) + ".";
    const int64_t row = (int64_t) T * kEmbd;

    // the input, saved before anything changes it
    dump_f32("inp", il, x_, row);
    cudaMemcpyAsync(inpSA_, x_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);

    // the attention-residual mix-in and the three attention inputs
    const Tensor* res_q = w_.find(b + "attn_res_query.weight");
    const float* in_q = x_;
    const float* in_k = x_;
    const float* in_v = x_;
    if (res_q) {
        // the mix over the saved blocks and the current input (last)
        kernels::res_mix(blocks_, kMaxBlocks, n_blocks_, x_, (const float*) res_q->dev, mixed_, T, stream_);
        dump_f32_col("mix0", il, mixed_, 3 * kEmbd, kEmbd, T);
        dump_f32_col("mix1", il, mixed_ + kEmbd, 3 * kEmbd, kEmbd, T);
        dump_f32_col("mix2", il, mixed_ + 2 * kEmbd, 3 * kEmbd, kEmbd, T);
        if (swa && looped) {
            // the three inputs start as the same mix and then come apart under the controllers
            for (int j = 1; j < 3; ++j)
                cudaMemcpy2DAsync(mixed_ + (int64_t) T * j * kEmbd, kEmbd * 4, mixed_, kEmbd * 4 * 3, kEmbd * 4, T,
                                  cudaMemcpyDeviceToDevice, s);
        }
        in_q = mixed_;
        in_k = swa && !looped ? mixed_ : mixed_ + row;
        in_v = swa && !looped ? mixed_ : mixed_ + 2 * row;
    }

    // the loop controllers on Q, K and V
    if (looped) {
        for (int j = 0; j < 3; ++j) {
            const char* c = j == 0 ? "q" : j == 1 ? "k" : "v";
            const Tensor* down = need(w_, b + "loop_" + std::string(c) + "_ctrl_down.weight");
            const Tensor* gain = need(w_, b + "loop_" + std::string(c) + "_ctrl_gain.weight");
            const Tensor* forget = need(w_, b + "loop_" + std::string(c) + "_ctrl_forget.weight");
            if (!down || !gain || !forget) return;
            const float* mj = msgs_ + (int64_t) j * row;
            float* inj = (float*) (j == 0 ? in_q : j == 1 ? in_k : in_v);
            kernels::loop_control(inj, mj, (const float*) down->dev, (const float*) gain->dev,
                                  (const float*) forget->dev, inj, T, stream_);
        }
    }

    // the input norms
    if (swa) {
        const Tensor* nrm = need(w_, b + "attn_norm.weight");
        if (!nrm) return;
        if (looped) {
            for (int j = 0; j < 3; ++j)
                sk::rms_norm_weighted((float*) (j == 0 ? in_q : j == 1 ? in_k : in_v), (const float*) nrm->dev, T,
                                       kEmbd, kRmsEps, stream_);
        } else {
            sk::rms_norm_weighted((float*) in_q, (const float*) nrm->dev, T, kEmbd, kRmsEps, stream_);
        }
    } else {
        const char* names[3] = {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight"};
        for (int j = 0; j < 3; ++j) {
            const Tensor* nrm = need(w_, b + names[j]);
            if (!nrm) return;
            sk::rms_norm_weighted((float*) (j == 0 ? in_q : j == 1 ? in_k : in_v), (const float*) nrm->dev, T,
                                       kEmbd, kRmsEps, stream_);
        }
    }
    {
        const int64_t pitch = (in_q == x_ || (in_q == mixed_ && in_k == in_q)) ? kEmbd : 3 * kEmbd;
        dump_f32_col("inq", il, in_q, pitch, kEmbd, T);
        dump_f32_col("ink", il, in_k, pitch, kEmbd, T);
        dump_f32_col("inv", il, in_v, pitch, kEmbd, T);
    }

    // the Q, K, V and relative-query projections
    const Tensor* wq = need(w_, b + "attn_q.weight");
    const Tensor* wk = need(w_, b + "attn_k.weight");
    const Tensor* wv = need(w_, b + "attn_v.weight");
    const Tensor* wr = need(w_, b + "attn_rel.weight");
    if (!wq || !wk || !wv || !wr) return;
    // each distinct input gets its own q8_1 rows: xq (the gate's as well), then scratch that the output
    // projection and the experts refill later, once these products have read it
    const int64_t row_q8 = kEmbd / 32 * 36;
    uint8_t* xq = scratch_q8_1_;
    uint8_t* kq = scratch_q8_1_ + (int64_t) T * row_q8 + (int64_t) T * (3 * kEmbd / 32 * 36);
    uint8_t* vq = scratch_q8_1_ + (int64_t) T * row_q8;
    sk::native_quantize_q8_1(in_q, xq, kEmbd, T, stream_);
    if (in_k != in_q) sk::native_quantize_q8_1(in_k, kq, kEmbd, T, stream_);
    if (in_v != in_q) sk::native_quantize_q8_1(in_v, vq, kEmbd, T, stream_);
    sk::native_mmvq(wq->type, wq->dev, xq, q_, kEmbd, kHeads * kHeadDim, T, stream_);
    sk::native_mmvq(wk->type, wk->dev, in_k == in_q ? xq : kq, k_, kEmbd, kEmbdK, T, stream_);
    sk::native_mmvq(wv->type, wv->dev, in_v == in_q ? xq : vq, v_, kEmbd, kEmbdK, T, stream_);
    sk::native_mmvq(wr->type, wr->dev, xq, r_, kEmbd, kRelLen * kHeads, T, stream_);

    // the K and V short convolutions (dynamic taps on the global layers)
    const Tensor* kconv_w = need(w_, b + "attn_k_conv.weight");
    const Tensor* vconv_w = need(w_, b + "attn_v_conv.weight");
    if (!kconv_w || !vconv_w) return;
    float* state = conv_state_ + (int64_t) il * (kConv - 1) * kConvChannels;
    float* kout = conv_kv_;
    float* vout = conv_kv_ + row;
    dump_f32("kx", il, k_, (int64_t) T * kEmbdK);                       // the conv input, pre-conv
    dump_f32("kstate", il, state, (kConv - 1) * kEmbdK);                 // the history the conv reads
    if (swa) {
        kernels::short_conv(k_, (const float*) kconv_w->dev, nullptr, state, kout, kEmbdK, 0, T, stream_);
        kernels::short_conv(v_, (const float*) vconv_w->dev, nullptr, state + kEmbdK, vout, kEmbdK, 0, T, stream_);
    } else {
        const Tensor* kproj = need(w_, b + "attn_k_conv_proj.weight");
        const Tensor* vproj = need(w_, b + "attn_v_conv_proj.weight");
        if (!kproj || !vproj) return;
        float* dyn_k = dyn_;
        float* dyn_v = dyn_ + (int64_t) T * kConv * kHeadsKv;
        kernels::gemv_f32(in_k, (const float*) kproj->dev, dyn_k, kEmbd, kConv * kHeadsKv, T, stream_);
        kernels::gemv_f32(in_v, (const float*) vproj->dev, dyn_v, kEmbd, kConv * kHeadsKv, T, stream_);
        kernels::short_conv(k_, (const float*) kconv_w->dev, dyn_k, state, kout, kEmbdK, kHeadsKv, T, stream_);
        kernels::short_conv(v_, (const float*) vconv_w->dev, dyn_v, state + kEmbdK, vout, kEmbdK, kHeadsKv, T,
                            stream_);
    }

    // the QK clip, the value norm and the log-position scale
    const Tensor* q_clip = need(w_, b + "attn_q_clip.weight");
    const Tensor* r_clip = need(w_, b + "attn_rel_clip.weight");
    if (!q_clip || !r_clip) return;
    kernels::clip_rows(q_, (const float*) q_clip->dev, swa ? nullptr : d_tau_, kHeads, kHeadDim, kQkClip, T, stream_);
    kernels::clip_rows(kout, nullptr, nullptr, kHeadsKv, kHeadDim, kQkClip, T, stream_);
    sk::rms_norm_weighted(vout, nullptr, (int64_t) T * kHeadsKv, kHeadDim, kRmsEps, stream_);
    kernels::clip_rows(r_, (const float*) r_clip->dev, swa ? nullptr : d_tau_, kHeads, kRelLen, kRelClip, T, stream_);
    dump_f32("q", il, q_, (int64_t) T * kHeads * kHeadDim);
    dump_f32("k", il, kout, (int64_t) T * kEmbdK);
    dump_f32("v", il, vout, (int64_t) T * kEmbdK);
    dump_f32("r", il, r_, (int64_t) T * kRelLen * kHeads);

    // the sinks and the gathered bias table
    const Tensor* sink_w = need(w_, b + "attn_sink_proj.weight");
    const Tensor* bias_w = need(w_, b + "attn_rel_bias.weight");
    if (!sink_w || !bias_w) return;
    kernels::gemv_f32(in_q, (const float*) sink_w->dev, sinks_, kEmbd, kHeads, T, stream_);
    dump_f32("sink", il, sinks_, (int64_t) T * kHeads);
    kernels::rel_bias_project((const float*) bias_w->dev, r_, sinks_, table_, kRelExtent, T, stream_);
    dump_f32("kqb", il, table_, (int64_t) T * (kRelExtent + 1) * kHeads);

    // the attention over this slot's pools
    kernels::kv_append(kout, vout, k_pools_ + pool_base_[(size_t) il], v_pools_ + pool_base_[(size_t) il], d_pos_,
                       caps_host_[(size_t) il], T, stream_);
    kernels::attn_forward(q_, k_pools_ + pool_base_[(size_t) il], v_pools_ + pool_base_[(size_t) il], table_, d_pos_,
                          swa ? kSwa : 0, caps_host_[(size_t) il], attn_out_, T, stream_);
    dump_f32("attn", il, attn_out_, (int64_t) T * kHeads * kHeadDim);

    // the sigmoid output gate and the output projection
    const Tensor* gate_w = need(w_, b + "attn_gate.weight");
    const Tensor* wo = need(w_, b + "attn_output.weight");
    if (!gate_w || !wo) return;
    sk::native_mmvq(gate_w->type, gate_w->dev, xq, gate_buf_, kEmbd, kHeads * kHeadDim, T, stream_);
    kernels::sigmoid_row(gate_buf_, gate_buf_, (int64_t) T * kHeads * kHeadDim, stream_);
    kernels::mul_inplace(attn_out_, gate_buf_, (int64_t) T * kHeads * kHeadDim, stream_);
    dump_f32("gated", il, attn_out_, (int64_t) T * kHeads * kHeadDim);
    uint8_t* aq = scratch_q8_1_ + (int64_t) T * (kEmbd / 32 * 36);
    sk::native_quantize_q8_1(attn_out_, aq, kHeads * kHeadDim, T, stream_);
    sk::native_mmvq(wo->type, wo->dev, aq, wo_buf_, kHeads * kHeadDim, kEmbd, T, stream_);
    dump_f32("wo", il, wo_buf_, row);

    // the post-attention convolution and the residual stream
    const Tensor* post_w = need(w_, b + "attn_post_conv.weight");
    if (!post_w) return;
    kernels::short_conv(wo_buf_, (const float*) post_w->dev, nullptr, state + 2 * kEmbdK, conv2304_, kEmbd, 0, T,
                        stream_);
    dump_f32("aconv", il, conv2304_, row);
    if (ip % kBlock == 0) {
        // a block starts: its pre-attention input joins the saved blocks and the output replaces the stream
        cudaMemcpy2DAsync(blocks_ + (int64_t) n_blocks_ * kEmbd, kEmbd * 4 * kMaxBlocks, inpSA_, kEmbd * 4, kEmbd * 4,
                          T, cudaMemcpyDeviceToDevice, s);
        ++n_blocks_;
        cudaMemcpyAsync(x_, conv2304_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);
    } else {
        cudaMemcpyAsync(x_, inpSA_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);
        sk::add_inplace(x_, conv2304_, row, stream_);
    }

    // the FFN input: mix, control, norm
    dump_f32("resid", il, x_, row);
    cudaMemcpyAsync(ffn_x_, x_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);
    const Tensor* ffn_q = need(w_, b + "ffn_res_query.weight");
    const Tensor* ffn_norm = need(w_, b + "ffn_norm.weight");
    if (!ffn_q || !ffn_norm) return;
    kernels::res_mix(blocks_, kMaxBlocks, n_blocks_, x_, (const float*) ffn_q->dev, wo_buf_, T, stream_);
    dump_f32("ffnmix", il, wo_buf_, row);
    if (looped) {
        const Tensor* down = need(w_, b + "loop_ffn_ctrl_down.weight");
        const Tensor* gain = need(w_, b + "loop_ffn_ctrl_gain.weight");
        const Tensor* forget = need(w_, b + "loop_ffn_ctrl_forget.weight");
        if (!down || !gain || !forget) return;
        kernels::loop_control(wo_buf_, msgs_ + (int64_t) 3 * row, (const float*) down->dev, (const float*) gain->dev,
                              (const float*) forget->dev, wo_buf_, T, stream_);
    }
    sk::rms_norm_weighted(wo_buf_, (const float*) ffn_norm->dev, T, kEmbd, kRmsEps, stream_);
    dump_f32("ffnx", il, wo_buf_, row);

    // the router and the experts
    const Tensor* router = need(w_, b + "ffn_gate_inp.weight");
    const Tensor* probs_b = w_.find(b + "exp_probs_b.bias");
    if (!router) return;
    kernels::gemv_f32(wo_buf_, (const float*) router->dev, router_l_, kEmbd, kExperts, T, stream_);
    kernels::route_topk(router_l_, probs_b ? (const float*) probs_b->dev : nullptr, expert_ids_, expert_w_, T,
                        stream_);
    const ExpertLayer& ex = w_.experts(phys);
    {
        // the expert input is quantized once per window; each (token, expert) pair runs one gate_up product,
        // SITU and one down product over the pack's [gate | up | down] blob
        uint8_t* eq = scratch_q8_1_ + (int64_t) T * (kEmbd / 32 * 36) + (int64_t) T * (3 * kEmbd / 32 * 36);
        uint8_t* hq = eq + (int64_t) T * (kEmbd / 32 * 36);
        sk::native_quantize_q8_1(wo_buf_, eq, kEmbd, T, stream_);
        cudaMemcpyAsync(h_ids_, expert_ids_, (size_t) T * kTopK * 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);            // the ids decide which blobs are read
        const int64_t gu_row = row_bytes(ex.gu_type, kEmbd);
        const int64_t xq_row = kEmbd / 32 * 36;
        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < kTopK; ++k) {
                const uint8_t* blob = ex.blob + (uint64_t) (h_ids_[(size_t) t * kTopK + k] & 0xff) * ex.blob_bytes;
                sk::native_mmvq(ex.gu_type, blob, eq + (int64_t) t * xq_row, gu_, kEmbd, 2 * kFf, 1, stream_);
                kernels::situ(gu_, h_, 1, kFf, stream_);
                sk::quantize_q8_1_rows(h_, 1, kFf, hq, stream_);
                sk::native_mmvq(ex.d_type, blob + 2 * kFf * gu_row, hq, parts_ + ((int64_t) t * kTopK + k) * kEmbd,
                                     kFf, kEmbd, 1, stream_);
            }
        }
        kernels::combine_parts(parts_, expert_w_, wo_buf_, kTopK, T, stream_);
        dump_i32_host("ids", il, h_ids_, (int64_t) T * kTopK);
        dump_f32("moe", il, wo_buf_, row);
    }

    // the FFN's post-convolution and the residual add
    const Tensor* fpost = need(w_, b + "ffn_post_conv.weight");
    if (!fpost) return;
    kernels::short_conv(wo_buf_, (const float*) fpost->dev, nullptr, state + 2 * kEmbdK + kEmbd, conv2304_ + row,
                        kEmbd, 0, T, stream_);
    dump_f32("fconv", il, conv2304_ + row, row);
    cudaMemcpyAsync(x_, ffn_x_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);
    sk::add_inplace(x_, conv2304_ + row, row, stream_);
    dump_f32("lout", il, x_, row);
}

model::Result Session::run_rows(const int32_t* tokens_host, const int64_t* pos_host, int T, bool wants_logits) {
    cudaStream_t s = (cudaStream_t) stream_;
    if (T <= 0 || T > cap_rows_) return {model::Status::Error, "a window of " + std::to_string(T) + " rows"};
    cudaMemcpyAsync(d_tokens_, tokens_host, (size_t) T * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_pos_, pos_host, (size_t) T * 8, cudaMemcpyHostToDevice, s);
    for (int t = 0; t < T; ++t) {
        const double p1 = (double) pos_host[t] + 1.0;
        const double a = p1 / (double) kLogFloor;
        h_tau_[t] = (float) (1.0 + kLogAlpha * (a > 1.0 ? std::log(a) : 0.0));
    }
    cudaMemcpyAsync(d_tau_, h_tau_, (size_t) T * 4, cudaMemcpyHostToDevice, s);

    // the embedding and its norm
    const Tensor* embd = need(w_, "token_embd.weight");
    const Tensor* embd_norm = need(w_, "token_embd_norm.weight");
    if (!embd || !embd_norm) return {model::Status::Error, "the pack misses the embedding"};
    kernels::gather_q8_0(embd->dev, d_tokens_, x_, kEmbd, T, stream_);
    sk::rms_norm_weighted(x_, (const float*) embd_norm->dev, T, kEmbd, kRmsEps, stream_);
    dump_f32("embd", -1, x_, (int64_t) T * kEmbd);

    n_blocks_ = 0;
    const int64_t row = (int64_t) T * kEmbd;
    for (int64_t n = 0; n < kSlots; ++n) {
        const int64_t il = slot_order(n);
        const bool pass_start = layer_pass(il) > 0 && layer_ip(il) == kCoreStart;
        if (pass_start) {
            // every pass after the first restarts from the prelude; the bank is the previous pass's block
            // after its first core block (saved block 2) followed by its output, each [T, kEmbd]
            cudaMemcpy2DAsync(bank_, kEmbd * 4, blocks_ + 2 * kEmbd, kEmbd * 4 * kMaxBlocks, kEmbd * 4, T,
                              cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(bank_ + row, x_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);
            const Tensor* mq = need(w_, "blk.6.loop_msg_query.weight");
            if (!mq) return {model::Status::Error, "the pack misses the loop message queries"};
            for (int col = 0; col < 4; ++col)
                kernels::res_mix(bank_, 1, 1, bank_ + row, (const float*) mq->dev + (int64_t) col * kEmbd,
                                 msgs_ + (int64_t) col * row, T, stream_);
            cudaMemcpyAsync(x_, prelude_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);
            n_blocks_ = 1;                          // the prelude's own block (layer 0's)
        }
        layer(il, T);
        if (il == kCoreStart - 1) cudaMemcpyAsync(prelude_, x_, (size_t) row * 4, cudaMemcpyDeviceToDevice, s);
    }

    if (!wants_logits) return {};
    const Tensor* outq = need(w_, "output_res_query.weight");
    const Tensor* onorm = need(w_, "output_norm.weight");
    const Tensor* head = need(w_, "output.weight");
    if (!outq || !onorm || !head) return {model::Status::Error, "the pack misses the output"};
    kernels::res_mix(blocks_, kMaxBlocks, n_blocks_, x_, (const float*) outq->dev, wo_buf_, T, stream_);
    dump_f32("outmix", -1, wo_buf_, row);
    sk::rms_norm_weighted(wo_buf_, (const float*) onorm->dev, T, kEmbd, kRmsEps, stream_);
    dump_f32("outnorm", -1, wo_buf_, row);
    uint8_t* xq = scratch_q8_1_;
    sk::native_quantize_q8_1(wo_buf_, xq, kEmbd, T, stream_);
    sk::native_mmvq(head->type, head->dev, xq, logits_, kEmbd, kVocabPadded, T, stream_);
    kernels::softcap30(logits_, (int64_t) T * kVocabPadded, stream_);
    return {};
}

model::Result Session::decode_window(const model::DecodeRequest& r) {
    if (r.n_tokens != 1) return {model::Status::Error, "this model decodes one row at a time"};
    if (r.pos != pos_next_) return {model::Status::Error, "decode at " + std::to_string(r.pos) + " but the conversation is at " +
                                                       std::to_string(pos_next_)};
    const int64_t pos = r.pos;
    const model::Result rr = run_rows(r.tokens, &pos, 1, true);
    if (!rr.ok()) return rr;
    pos_next_ = r.pos + 1;
    cudaStream_t s = (cudaStream_t) stream_;
    sk::SamplerParams p;
    if (r.sampling) p = *r.sampling;
    p.greedy = true;                                 // the parity path; a sampler rides in r.sampling later
    p.temperature = 0.0f;
    sk::sample_tokens(logits_, 1, (int) kVocab, nullptr, 0, p, (int*) d_out_, stream_);
    if (r.logits) {
        cudaMemcpyAsync(h_logits_, logits_, (size_t) kVocab * 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        std::memcpy(r.logits, h_logits_, (size_t) kVocab * 4);
    }
    if (r.next_token) {
        cudaMemcpyAsync(r.next_token, d_out_, 4, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
    }
    return {};
}

model::Result Session::prefill_chunk(const model::PrefillRequest& r) {
    if (r.n_tokens <= 0) return {model::Status::Error, "an empty prefill chunk"};
    if (r.pos != pos_next_ && pos_next_ != 0)
        return {model::Status::Error, "prefill at " + std::to_string(r.pos) + " but the conversation is at " +
                                          std::to_string(pos_next_)};
    int at = 0;
    while (at < r.n_tokens) {
        const int n = std::min(cap_rows_, r.n_tokens - at);
        std::vector<int64_t> pos(n);
        for (int t = 0; t < n; ++t) pos[(size_t) t] = r.pos + at + t;
        const model::Result rr = run_rows(r.tokens + at, pos.data(), n, false);
        if (!rr.ok()) return rr;
        at += n;
    }
    pos_next_ = r.pos + r.n_tokens;
    if (r.progress) r.progress(r.n_tokens);
    return {};
}

}  // namespace basalt::models::inkling2
