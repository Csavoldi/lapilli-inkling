// src/models/inkling2/kernels/ink_kernels.cu - the model-local kernels of Inkling 2 Mini. See the header.
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/models/inkling2/geometry.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstring>

namespace basalt::models::inkling2::kernels {
namespace {

constexpr float kMixEps = 1e-6f;
constexpr int kThreads = 256;

/// The block's sum of `v`, added in thread order so a run repeats bitwise; `tmp` is the block's shared scratch,
/// at least blockDim.x floats.
__device__ __forceinline__ float block_sum(float v, float* tmp) {
    tmp[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < (int) blockDim.x; ++i) s += tmp[i];
        tmp[0] = s;
    }
    __syncthreads();
    const float r = tmp[0];
    __syncthreads();
    return r;
}

// ------------------------------------------------------------------ relative-bias projection
__global__ void rel_bias_kernel(const float* __restrict__ bias, const float* __restrict__ r,
                                const float* __restrict__ sinks, float* __restrict__ table, int extent, int T) {
    const int h = threadIdx.x;                    // 24 heads
    const int d = blockIdx.x;                     // extent + 1 rows (the pad row carries -sink only)
    const int t = blockIdx.y;
    float sum = 0.0f;
    if (d < extent)                               // bias is [16, extent] with ne0 = 16 contiguous: (r, d) at d * 16 + r
        for (int i = 0; i < 16; ++i) sum += bias[(size_t) d * 16 + i] * r[((size_t) t * 24 + h) * 16 + i];
    table[((size_t) t * (extent + 1) + d) * 24 + h] = sum - sinks[(size_t) t * 24 + h];
}

// ------------------------------------------------------------------ attention
constexpr int kAttnThreads = 128;       // one block per (token, kv head), threads over the head dimension

__device__ __forceinline__ float attn_block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int off = kAttnThreads / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

__global__ void attn_kernel(const float* __restrict__ q, const uint16_t* __restrict__ k_pool,
                            const uint16_t* __restrict__ v_pool, const float* __restrict__ table,
                            const int64_t* __restrict__ pos, int64_t window, int64_t cap, float* __restrict__ out,
                            int T) {
    // one block per (token, kv head), 128 threads over the head dimension: the score is the dot of q and k over
    // the head dimension (a block reduction) plus the gathered bias; the softmax covers the keys and the sink's
    // zero logit (which the max, starting at 0, covers); two passes give the full-softmax numerics
    const int t = blockIdx.y;
    const int hkv = blockIdx.x;
    if (t >= T) return;
    const __half* kh = reinterpret_cast<const __half*>(k_pool);
    const __half* vh = reinterpret_cast<const __half*>(v_pool);
    const int64_t p = pos[t];
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : 0) : 0;
    constexpr int n_head = (int) (kHeads / kHeadsKv);          // 3 query heads per kv head
    const float* qt = q + ((size_t) t * kHeads + hkv * n_head) * kHeadDim;

    float qv[n_head];
    float o[n_head];
    for (int j = 0; j < n_head; ++j) { qv[j] = qt[j * kHeadDim + threadIdx.x]; o[j] = 0.0f; }
    const float scale = rsqrtf((float) kHeadDim);
    __shared__ float red[kAttnThreads];
    __shared__ float s_m[n_head], s_den[n_head];
    if (threadIdx.x == 0)
        for (int j = 0; j < n_head; ++j) { s_m[j] = 0.0f; s_den[j] = 0.0f; }   // the sink's logit is 0
    __syncthreads();

    for (int64_t kp = lo; kp <= p; ++kp) {                    // pass 1: the max score per query head
        const float kf = __half2float(kh[(((size_t) (kp % cap)) * kHeadsKv + hkv) * kHeadDim + threadIdx.x]);
        const int bd = (int) (p - kp < kRelExtent ? p - kp : kRelExtent);
        const float* trow = table + ((size_t) t * (kRelExtent + 1) + bd) * 24;
        for (int j = 0; j < n_head; ++j) {
            const float dot = attn_block_sum(qv[j] * kf, red);
            if (threadIdx.x == 0) s_m[j] = fmaxf(s_m[j], dot * scale + trow[hkv * n_head + j]);
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        for (int j = 0; j < n_head; ++j) s_den[j] = __expf(-s_m[j]);           // the sink's exp(0 - m)
    __syncthreads();
    for (int64_t kp = lo; kp <= p; ++kp) {                    // pass 2: the weighted value sum
        const float kf = __half2float(kh[(((size_t) (kp % cap)) * kHeadsKv + hkv) * kHeadDim + threadIdx.x]);
        const float vf = __half2float(vh[(((size_t) (kp % cap)) * kHeadsKv + hkv) * kHeadDim + threadIdx.x]);
        const int bd = (int) (p - kp < kRelExtent ? p - kp : kRelExtent);
        const float* trow = table + ((size_t) t * (kRelExtent + 1) + bd) * 24;
        for (int j = 0; j < n_head; ++j) {
            const float dot = attn_block_sum(qv[j] * kf, red);
            const float e = __expf(dot * scale + trow[hkv * n_head + j] - s_m[j]);
            if (threadIdx.x == 0) s_den[j] += e;
            o[j] += e * vf;
            __syncthreads();
        }
    }
    for (int j = 0; j < n_head; ++j)
        out[((size_t) t * kHeads + hkv * n_head + j) * kHeadDim + threadIdx.x] = o[j] / s_den[j];
}

// ------------------------------------------------------------------ short convolution
__global__ void conv_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ dyn,
                            const float* __restrict__ state, float* __restrict__ out, int64_t c, int dyn_heads, int T,
                            float scale, int64_t stride) {
    // one thread per (token, channel); w is [4, c] with ne0 = 4 contiguous (tap `lag` of channel ch at ch * 4 + lag).
    // The state is [3, all convolutions' channels] with this convolution's channels at the pointer's offset, a row
    // (one past position) spanning `stride` floats.
    const int64_t ch = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (ch >= c || t >= T) return;
    const int64_t per = dyn ? c / dyn_heads : 0;
    const int head = dyn ? (int) (ch / per) : 0;
    float sum = 0.0f;
    for (int lag = 0; lag < 4; ++lag) {
        const int rel = t - 3 + lag;
        const float v = rel >= 0 ? x[(size_t) rel * c + ch] : state[(size_t) (rel + 3) * stride + ch];
        sum += w[(size_t) ch * 4 + lag] * v;
        if (dyn) sum += v * dyn[((size_t) t * dyn_heads + head) * 4 + lag];
    }
    out[(size_t) t * c + ch] = (x[(size_t) t * c + ch] + sum) * scale;
}

__global__ void conv_commit_kernel(const float* __restrict__ x, float* __restrict__ state, int64_t c, int T,
                                   int64_t stride) {
    // after the window: the state becomes the last 3 input rows (its own tail when the window is shorter)
    const int64_t ch = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (ch >= c) return;
    for (int lag = 0; lag < 3; ++lag) {
        const int rel = T - 3 + lag;
        state[(size_t) lag * stride + ch] = rel >= 0 ? x[(size_t) rel * c + ch] : state[(size_t) (rel + 3) * stride + ch];
    }
}

// ------------------------------------------------------------------ attention-residual mix
__global__ void mix_kernel(const float* __restrict__ blocks, int stride, int d, const float* __restrict__ cur,
                           const float* __restrict__ query, float* __restrict__ out) {
    // one block per token over kEmbd values; the d block states at (t * stride + i) * kEmbd, then `cur`
    const int t = blockIdx.x;
    __shared__ float s_rms[11];
    __shared__ float s_score[11];
    __shared__ float s_q[kEmbd];
    __shared__ float s_tmp[kThreads];
    const int n = d + 1;
    for (int i = threadIdx.x; i < kEmbd; i += blockDim.x) s_q[i] = query[i];
    __syncthreads();
    for (int ds = 0; ds < n; ++ds) {
        const float* x = ds < d ? blocks + ((size_t) t * stride + ds) * kEmbd : cur + (size_t) t * kEmbd;
        float ms = 0.0f, dot = 0.0f;
        for (int i = threadIdx.x; i < kEmbd; i += blockDim.x) {
            const float v = x[i];
            ms += v * v;
            dot += s_q[i] * v;
        }
        const float ms_all = block_sum(ms, s_tmp);
        const float dot_all = block_sum(dot, s_tmp);
        if (threadIdx.x == 0) {
            const float rms = rsqrtf(ms_all / kEmbd + kMixEps);
            s_rms[ds] = rms;
            s_score[ds] = dot_all * rms * rsqrtf((float) kEmbd);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -CUDART_INF_F;
        for (int ds = 0; ds < n; ++ds) mx = fmaxf(mx, s_score[ds]);
        float den = 0.0f;
        for (int ds = 0; ds < n; ++ds) {
            s_score[ds] = __expf(s_score[ds] - mx);
            den += s_score[ds];
        }
        for (int ds = 0; ds < n; ++ds) s_score[ds] /= den;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kEmbd; i += blockDim.x) {
        float o = 0.0f;
        for (int ds = 0; ds < n; ++ds) {
            const float* x = ds < d ? blocks + ((size_t) t * stride + ds) * kEmbd : cur + (size_t) t * kEmbd;
            o += s_score[ds] * s_rms[ds] * x[i];
        }
        out[(size_t) t * kEmbd + i] = o;
    }
}

// ------------------------------------------------------------------ row L2 clip
__global__ void clip_kernel(float* __restrict__ x, const float* __restrict__ chan, const float* __restrict__ tau,
                            int n_row, int cols, float limit, int T) {
    // one block per (token, row); chan is [n_row, cols] (ne0 = cols contiguous)
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= T) return;
    float* xr = x + ((size_t) t * n_row + row) * cols;
    __shared__ float s_tmp[kThreads];
    float part = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = xr[i];
        if (chan) v *= chan[(size_t) row * cols + i];
        xr[i] = v;                                 // the channel scale stays: the clip applies to the scaled row
        part += v * v;
    }
    const float norm = sqrtf(block_sum(part, s_tmp)) + 1e-6f;
    const float factor = fminf(fmaxf(limit / norm, 0.0f), 1.0f);
    const float tv = tau ? tau[t] : 1.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] = (xr[i] * factor) * tv;
}

// ------------------------------------------------------------------ loop controllers
__global__ void control_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                               const float* __restrict__ down, const float* __restrict__ gain,
                               const float* __restrict__ forget, float* __restrict__ out) {
    // one block per token: g = down . concat(rms(cur), rms(msg)) [rank], then the tanh-gated blend
    const int t = blockIdx.x;
    __shared__ float s_cur[kEmbd], s_msg[kEmbd], s_g[kGateRank], s_tmp[kThreads];
    const float* c = cur + (size_t) t * kEmbd;
    const float* m = msg + (size_t) t * kEmbd;
    for (int i = threadIdx.x; i < kEmbd; i += blockDim.x) {
        s_cur[i] = c[i];
        s_msg[i] = m[i];
    }
    __syncthreads();
    float ms = 0.0f, dm = 0.0f;
    for (int i = threadIdx.x; i < kEmbd; i += blockDim.x) {
        ms += s_cur[i] * s_cur[i];
        dm += s_msg[i] * s_msg[i];
    }
    const float rms_c = rsqrtf(block_sum(ms, s_tmp) / kEmbd + kMixEps);
    const float rms_m = rsqrtf(block_sum(dm, s_tmp) / kEmbd + kMixEps);
    for (int r = 0; r < kGateRank; ++r) {
        const float* dr = down + (size_t) r * (2 * kEmbd);
        float part = 0.0f;
        for (int i = threadIdx.x; i < 2 * kEmbd; i += blockDim.x)
            part += dr[i] * (i < kEmbd ? s_cur[i] * rms_c : s_msg[i - kEmbd] * rms_m);
        const float g = block_sum(part, s_tmp);
        if (threadIdx.x == 0) s_g[r] = g;
        __syncthreads();
    }
    for (int i = threadIdx.x; i < kEmbd; i += blockDim.x) {
        float gn = 0.0f, fg = 0.0f;
        const float* gw = gain + (size_t) i * kGateRank;
        const float* fw = forget + (size_t) i * kGateRank;
        for (int r = 0; r < kGateRank; ++r) {
            gn += gw[r] * s_g[r];
            fg += fw[r] * s_g[r];
        }
        const float gain_v = tanhf(gn);
        const float forget_v = tanhf(fg);
        out[(size_t) t * kEmbd + i] = c[i] - forget_v * c[i] + gain_v * m[i];
    }
}

// ------------------------------------------------------------------ router
__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                             int32_t* __restrict__ ids, float* __restrict__ w) {
    const int t = blockIdx.x;
    __shared__ float s_prob[kExperts], s_sel[kExperts];
    const float* l = logits + (size_t) t * kExperts;
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x) {
        const float p = 1.0f / (1.0f + __expf(-l[e]));
        s_prob[e] = p;
        s_sel[e] = p + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool taken[kExperts] = {};
        float wsum = 0.0f;
        for (int k = 0; k < kTopK; ++k) {
            int best = -1;
            for (int e = 0; e < kExperts; ++e)
                if (!taken[e] && (best < 0 || s_sel[e] > s_sel[best])) best = e;
            taken[best] = true;
            ids[(size_t) t * kTopK + k] = best;
            w[(size_t) t * kTopK + k] = s_prob[best];
            wsum += s_prob[best];
        }
        const float inv = kRouteScale / (wsum + kRouteEps);
        for (int k = 0; k < kTopK; ++k) w[(size_t) t * kTopK + k] *= inv;
    }
}

// ------------------------------------------------------------------ SITU
__global__ void situ_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t n, int64_t n_ff) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / n_ff;
    const int64_t col = i - row * n_ff;
    const float g = gu[(size_t) row * (2 * n_ff) + col];
    const float u = gu[(size_t) row * (2 * n_ff) + n_ff + col];
    const float act = kSituBeta * tanhf(g / kSituBeta) * (1.0f / (1.0f + __expf(-g)));
    const float lin = kSituLinearBeta * tanhf(u / kSituLinearBeta);
    h[i] = act * lin;
}

// ------------------------------------------------------------------ plain F32 GEMV
__global__ void gemv_f32_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                int64_t n_in, int64_t n_out, int T) {
    const int64_t o = blockIdx.x;
    const int t = blockIdx.y;
    if (t >= T) return;
    const float* xr = x + (size_t) t * n_in;
    __shared__ float s_dot[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, n_warps = (int) (blockDim.x >> 5);
    float acc = 0.0f;
    for (int64_t i = threadIdx.x; i < n_in; i += blockDim.x) acc += xr[i] * w[(size_t) o * n_in + i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) s_dot[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int g = 0; g < n_warps; ++g) total += s_dot[g];
        y[(size_t) t * n_out + o] = total;
    }
}

// ------------------------------------------------------------------ Q8_0 row gather
__global__ void gather_q8_0_kernel(const uint8_t* __restrict__ matrix, const int32_t* __restrict__ ids,
                                   float* __restrict__ out, int64_t embd) {
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    const int64_t row_bytes = embd / 32 * 34;
    const uint8_t* r = matrix + (size_t) ids[t] * row_bytes + (size_t) b * 34;
    __half dh;
    memcpy(&dh, r, 2);                                    // the fp16 scale, at an arbitrary alignment
    const float d = __half2float(dh);
    const int8_t q = (int8_t) r[2 + threadIdx.x];
    out[(size_t) t * embd + b * 32 + threadIdx.x] = d * (float) q;
}

// ------------------------------------------------------------------ elementwise
__global__ void softcap_kernel(float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = kLogitSoftcap * tanhf(x[i] / kLogitSoftcap);
}

__global__ void sigmoid_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + __expf(-x[i]));
}

__global__ void combine_kernel(const float* __restrict__ parts, const float* __restrict__ w, float* __restrict__ out,
                               int topk, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * kEmbd) return;
    const int64_t t = i / kEmbd;
    float o = 0.0f;
    for (int k = 0; k < topk; ++k)
        o += w[(size_t) t * topk + k] * parts[((size_t) t * topk + k) * kEmbd + (i - t * kEmbd)];
    out[i] = o;
}

__global__ void mul_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] *= src[i];
}

// ------------------------------------------------------------------ KV append
__global__ void kv_append_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                 uint16_t* __restrict__ k_pool, uint16_t* __restrict__ v_pool,
                                 const int64_t* __restrict__ pos, int64_t cap) {
    // one block per token, 1024 threads over the kv rows
    const int i = threadIdx.x;
    const int t = blockIdx.x;
    if (i >= kEmbdK) return;
    const size_t row = ((size_t) (pos[t] % cap)) * kEmbdK;
    k_pool[row + i] = __float2half_rn(k[(size_t) t * kEmbdK + i]);
    v_pool[row + i] = __float2half_rn(v[(size_t) t * kEmbdK + i]);
}

}  // namespace

#define LAUNCH(fn, grid, block, ...) fn<<<grid, block, 0, (cudaStream_t) stream>>>(__VA_ARGS__)

void rel_bias_project(const float* bias, const float* r, const float* sinks, float* table,
                      int64_t extent, int T, void* stream) {
    dim3 grid((unsigned) (extent + 1), (unsigned) T);
    LAUNCH(rel_bias_kernel, grid, 24, bias, r, sinks, table, (int) extent, T);
}

void attn_forward(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const float* table,
                  const int64_t* pos, int64_t window, int64_t cap, float* out, int T, void* stream) {
    dim3 grid((unsigned) kHeadsKv, (unsigned) T);
    LAUNCH(attn_kernel, grid, 128, q, k_pool, v_pool, table, pos, window, cap, out, T);
}

void short_conv(const float* x, const float* w, const float* dyn, float* state, float* out,
                int64_t c, int dyn_heads, int T, void* stream) {
    // the state's row stride: all four convolutions share the [3, kConvChannels] state, this one at its offset
    const float scale = dyn ? 1.0f / 3.0f : rsqrtf((float) kConv + 1);
    const unsigned nx = (unsigned) ((c + 255) / 256);
    LAUNCH(conv_kernel, dim3(nx, (unsigned) T), 256, x, w, dyn, state, out, c, dyn_heads, T, scale,
           (int64_t) kConvChannels);
    LAUNCH(conv_commit_kernel, dim3(nx), 256, x, state, c, T, (int64_t) kConvChannels);
}

void res_mix(const float* blocks, int stride, int d, const float* cur, const float* query, float* out, int T,
             void* stream) {
    LAUNCH(mix_kernel, dim3((unsigned) T), kThreads, blocks, stride, d, cur, query, out);
}

void clip_rows(float* x, const float* chan, const float* tau, int n_row, int cols, float limit, int T, void* stream) {
    dim3 grid((unsigned) n_row, (unsigned) T);
    const int threads = cols >= 128 ? 128 : 32;
    LAUNCH(clip_kernel, grid, threads, x, chan, tau, n_row, cols, limit, T);
}

void loop_control(const float* cur, const float* msg, const float* down, const float* gain, const float* forget,
                  float* out, int T, void* stream) {
    LAUNCH(control_kernel, dim3((unsigned) T), kThreads, cur, msg, down, gain, forget, out);
}

void route_topk(const float* logits, const float* bias, int32_t* ids, float* w, int T, void* stream) {
    LAUNCH(route_kernel, dim3((unsigned) T), 128, logits, bias, ids, w);
}

void situ(const float* gu, float* h, int64_t rows, int64_t n_ff, void* stream) {
    const int64_t n = rows * n_ff;
    LAUNCH(situ_kernel, dim3((unsigned) ((n + 255) / 256)), 256, gu, h, n, n_ff);
}

void gemv_f32(const float* x, const float* w, float* y, int64_t n_in, int64_t n_out, int T, void* stream) {
    dim3 grid((unsigned) n_out, (unsigned) T);
    LAUNCH(gemv_f32_kernel, grid, 256, x, w, y, n_in, n_out, T);
}

void gather_q8_0(const void* matrix, const int32_t* ids, float* out, int64_t embd, int T, void* stream) {
    dim3 grid((unsigned) (embd / 32), (unsigned) T);
    LAUNCH(gather_q8_0_kernel, grid, 32, (const uint8_t*) matrix, ids, out, embd);
}

void softcap30(float* x, int64_t n, void* stream) {
    LAUNCH(softcap_kernel, dim3((unsigned) ((n + 255) / 256)), 256, x, n);
}

void combine_parts(const float* parts, const float* w, float* out, int topk, int T, void* stream) {
    LAUNCH(combine_kernel, dim3((unsigned) ((T * kEmbd + 255) / 256)), 256, parts, w, out, topk, T);
}

void sigmoid_row(const float* x, float* out, int64_t n, void* stream) {
    LAUNCH(sigmoid_kernel, dim3((unsigned) ((n + 255) / 256)), 256, x, out, n);
}

void mul_inplace(float* dst, const float* src, int64_t n, void* stream) {
    LAUNCH(mul_kernel, dim3((unsigned) ((n + 255) / 256)), 256, dst, src, n);
}

void kv_append(const float* k, const float* v, uint16_t* k_pool, uint16_t* v_pool, const int64_t* pos,
               int64_t cap, int T, void* stream) {
    LAUNCH(kv_append_kernel, dim3((unsigned) T), 1024, k, v, k_pool, v_pool, pos, cap);
}

}  // namespace basalt::models::inkling2::kernels
