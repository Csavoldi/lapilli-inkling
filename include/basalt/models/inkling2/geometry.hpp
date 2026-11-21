// include/basalt/models/inkling2/geometry.hpp - the constants of Inkling 2 Mini and its looped layer slots.
//
// The GGUF carries 24 physical layers; the looped core (layers 6..17) runs three passes, so the engine walks 48
// layer applications ("slots") that share the physical weights but each keep their own KV pool and short-conv
// state. Slot `il < 24` is the physical layer; above that, pass p >= 1 revisits core layer `6 + (il - 24) % 12`.
#pragma once

#include <cstdint>

namespace basalt::models::inkling2 {

constexpr int64_t kEmbd = 2304;          // hidden width
constexpr int64_t kVocab = 200058;       // real vocabulary (the padded head is 201024)
constexpr int64_t kVocabPadded = 201024;
constexpr int64_t kHeads = 24;           // query heads
constexpr int64_t kHeadsKv = 8;          // key/value heads
constexpr int64_t kHeadDim = 128;
constexpr int64_t kEmbdK = kHeadsKv * kHeadDim;             // 1024
constexpr int64_t kRelLen = 16;          // relative-position query width per head
constexpr int64_t kRelExtent = 1024;     // bias table extent (SWA window and global clamp)
constexpr int64_t kSwa = 1024;           // sliding window of the 20 SWA layers
constexpr int64_t kLayersPhys = 24;
constexpr int64_t kCoreStart = 6;
constexpr int64_t kCoreEnd = 18;
constexpr int64_t kPasses = 3;
constexpr int64_t kGateRank = 64;        // loop controller rank
constexpr int64_t kBlock = 6;            // attention-residual block size
constexpr int64_t kConv = 4;             // short-convolution taps
constexpr int64_t kExperts = 128;
constexpr int64_t kTopK = 8;
constexpr int64_t kFf = 1152;            // expert intermediate width
constexpr float kRmsEps = 1e-6f;
constexpr float kQkClip = 22.627416997969522f;   // 16*sqrt(2)
constexpr float kRelClip = 8.0f;
constexpr int64_t kLogFloor = 8192;
constexpr float kLogAlpha = 0.1f;
constexpr float kSituBeta = 4.0f;
constexpr float kSituLinearBeta = 25.0f;
constexpr float kRouteScale = 8.0f;
constexpr float kRouteEps = 1e-20f;
constexpr float kLogitSoftcap = 30.0f;
constexpr int64_t kEos = 200006;

constexpr int64_t kSlots = kLayersPhys + (kPasses - 1) * (kCoreEnd - kCoreStart);   // 48
constexpr int64_t kConvChannels = 2 * kEmbdK + 2 * kEmbd;                           // 6656: K | V | attn | ffn
constexpr int64_t kMaxStates = kSlots / kBlock + 2;                                 // attn-res blocks + cur (8)

/// The physical layer of slot `il` (the weights it reads).
inline int64_t layer_phys(int64_t il) { return il < kLayersPhys ? il : kCoreStart + (il - kLayersPhys) % (kCoreEnd - kCoreStart); }

/// The pass of slot `il` (0 for the physical run of the core).
inline int64_t layer_pass(int64_t il) { return il < kLayersPhys ? 0 : 1 + (il - kLayersPhys) / (kCoreEnd - kCoreStart); }

/// The core index (ip) of slot `il` inside [kCoreStart, kCoreEnd); used for the block-boundary rule.
inline int64_t layer_ip(int64_t il) {
    return il < kLayersPhys ? il : kCoreStart + (il - kLayersPhys) % (kCoreEnd - kCoreStart);
}

/// The four global-attention physical layers are the last of each attention-residual block.
inline bool layer_is_swa(int64_t phys) { return (phys + 1) % kBlock != 0; }

/// The slot the reference runs n-th: the prelude (0..5), then the loop's passes back to back (each revisiting
/// core layers 6..17 on its own slots: 6..17, 24..35, 36..47), and only then the coda layers (18..23). The slot
/// numbers are not the execution order.
inline int64_t slot_order(int64_t n) {
    constexpr int64_t core = kCoreEnd - kCoreStart;
    if (n < kCoreStart) return n;
    if (n < kCoreStart + kPasses * core) {
        const int64_t k = n - kCoreStart;
        const int64_t pass = k / core;
        const int64_t ip = kCoreStart + k % core;
        return pass == 0 ? ip : kLayersPhys + (pass - 1) * core + (ip - kCoreStart);
    }
    return n - (kPasses - 1) * core;
}

/// The KV pool capacity of slot `il`: the SWA slots hold one window, the global ones the whole context.
inline int64_t slot_kv_cap(int64_t il, int64_t max_context) {
    return layer_is_swa(layer_phys(il)) ? kSwa : max_context;
}

}  // namespace basalt::models::inkling2
