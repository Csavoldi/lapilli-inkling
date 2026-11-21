// include/basalt/models/inkling2/kernels.hpp - the model-local CUDA kernels of Inkling 2 Mini.
//
// Correctness first: plain kernels, one thread block per natural unit, no tensor cores and no CUDA graphs; the
// shared kernels (rms_norm_weighted, native_mmvq, quantize_act) carry the rest. Every kernel here has a
// double-precision CPU reference in tests/models/inkling2/kernel_parity.cpp.
#pragma once

#include <cstdint>

namespace basalt::models::inkling2::kernels {

/// The relative-bias projection: table[t][d][h] = (sum_r bias[r][d] * R[t][h][r]) - sink[t][h], over the
/// extent+1 rows of the zero-padded table (the pad row carries -sink too, so a distance-clamped key keeps its
/// sink offset). R is [T, 24, 16], bias [16, extent+1] row-major (ne0 = extent+1), sinks [T, 24],
/// table [T, extent+1, 24].
void rel_bias_project(const float* bias, const float* r, const float* sinks, float* table,
                      int64_t extent, int T, void* stream);

/// Causal attention over one slot's fp16 pools: q [T, 24, 128], pools [cap][8][128] (ring index key % cap),
/// table from rel_bias_project, pos [T] the queries' absolute positions. window = 0 attends every key at or
/// before the query (the global slots); otherwise only keys within `window` positions. The softmax denominator
/// carries one extra exp(0) term (the attention sink; its logit left the table above). out [T, 24, 128].
void attn_forward(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const float* table,
                  const int64_t* pos, int64_t window, int64_t cap, float* out, int T, void* stream);

/// The causal short convolution of one slot half: out = (x + conv(x)) * scale, conv a depthwise 4-tap causal
/// filter over the channel axis whose prologue is `state` (the last 3 input rows, oldest first). `dyn`, when
/// not null, adds the input-dependent taps of the global layers' K and V convolutions ([T, heads, 4], one tap
/// vector per kv head; the scale is then 1/3 instead of 1/sqrt(5)). state becomes the window's last 3 rows.
/// x and out are [T, c]; w is [4, c] (w[lag * c + ch], lag 0 the oldest tap).
void short_conv(const float* x, const float* w, const float* dyn, float* state, float* out,
                int64_t c, int dyn_heads, int T, void* stream);

/// The attention-residual mix: `d` block states, the i-th of token t at blocks + (t * stride + i) * 2304, plus
/// `cur` [T, 2304]; each RMS-normalized (no gain) and weighted by softmax(query . state / sqrt(2304)) over the
/// d + 1 states. query [2304]; out [T, 2304].
void res_mix(const float* blocks, int stride, int d, const float* cur, const float* query, float* out, int T,
             void* stream);

/// Per-row L2 clip after an optional per-channel scale: x [T, n_row, cols] in place; chan [cols, n_row] when
/// given (Q's per-channel scales, R's per-head clip scales); tau [T] multiplies each token's rows after the
/// clip (the global layers' log-position scale). limit is the row norm bound.
void clip_rows(float* x, const float* chan, const float* tau, int n_row, int cols, float limit, int T, void* stream);

/// The loop controllers' gate: out = (1 - tanh(F g)) * cur + tanh(G g) * msg, g = W_d [rank, 4608] applied to
/// concat(rms(cur), rms(msg)). down/gain/forget are [rank, 4608], [2304, rank], [2304, rank] row-major
/// (ne0 contiguous). cur, msg, out [T, 2304].
void loop_control(const float* cur, const float* msg, const float* down, const float* gain, const float* forget,
                  float* out, int T, void* stream);

/// The sigmoid router: logits [T, 128] in fp32; selection by logits + bias (when not null), weights the
/// unbiased sigmoid probabilities of the 8 selected, renormalized (sum + 1e-20) and scaled by 8.
/// ids [T, 8], w [T, 8].
void route_topk(const float* logits, const float* bias, int32_t* ids, float* w, int T, void* stream);

/// SITU activation: gu [rows, 2*n_ff] (gate half then up half per row) -> h [rows, n_ff]:
/// h = (beta * tanh(g / beta) * sigmoid(g)) * (lb * tanh(u / lb)), beta 4, lb 25.
void situ(const float* gu, float* h, int64_t rows, int64_t n_ff, void* stream);

/// A plain F32 GEMV over small dense weights: y [T, n_out] = x [T, n_in] . w^T, w [n_in, n_out] row-major.
void gemv_f32(const float* x, const float* w, float* y, int64_t n_in, int64_t n_out, int T, void* stream);

/// Rows of a Q8_0 embedding (row-major [rows, embd] blocks, ne0 = embd contiguous) gathered to f32 [T, embd].
void gather_q8_0(const void* matrix, const int32_t* ids, float* out, int64_t embd, int T, void* stream);

/// The final logit softcap, 30 * tanh(x / 30), in place.
void softcap30(float* x, int64_t n, void* stream);

/// Weighted expert combine: parts [T, topk, 2304], w [T, topk] -> out [T, 2304].
void combine_parts(const float* parts, const float* w, float* out, int topk, int T, void* stream);

/// out [n] = sigmoid(x [n]), the attention output gate precomputed (applied by mul in the caller when needed).
void sigmoid_row(const float* x, float* out, int64_t n, void* stream);

/// dst[i] *= src[i], the sigmoid gate applied to the attention output.
void mul_inplace(float* dst, const float* src, int64_t n, void* stream);

/// One window's K and V rows [T, 1024] (post-convolution, clip and norm) into a slot's fp16 pools at their
/// positions' ring cells.
void kv_append(const float* k, const float* v, uint16_t* k_pool, uint16_t* v_pool, const int64_t* pos,
               int64_t cap, int T, void* stream);

}  // namespace basalt::models::inkling2::kernels
