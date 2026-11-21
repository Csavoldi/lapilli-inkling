// include/basalt/models/inkling2/model.hpp - the loaded weights of an Inkling 2 Mini pack and its sessions' seams.
#pragma once

#include "basalt/core/weights.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/models/inkling2/geometry.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

/// One tensor as the engine reads it: a device pointer with its ggml type and [ne0 (contiguous), ne1] shape.
struct Tensor {
    const void* dev = nullptr;
    int type = 0;               // ggml type id (0 = F32)
    int64_t ne0 = 0, ne1 = 0;
};

/// The routed experts of one layer: the device blobs in the pack's [gate | up | down] layout.
struct ExpertLayer {
    const uint8_t* blob = nullptr;   // n_expert blobs
    int gu_type = 0, d_type = 0;     // gate/up rows' type and the down rows'
    uint64_t blob_bytes = 0;
};

/// Everything a session needs: the dense F32 pool, the native Q8_0 projections on the device and the expert blobs.
class Weights {
public:
    ~Weights();
    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;

    /// Loads from a .basalt pack (its GGUF view must be registered). max_context sizes nothing here.
    static bool load(const std::string& pack_path, int64_t max_context, std::unique_ptr<Weights>& out,
                     std::string& err);

    const Tensor* find(const std::string& name) const;
    const ExpertLayer& experts(int64_t layer) const { return expert_layers_[(size_t) layer]; }
    /// The context the loaded model runs with (what its fp16 pools were sized for).
    int64_t context() const { return context_; }

private:
    Weights() = default;
    core::WeightTable table_;               // the dense F32 pool's arena
    void* arena_ = nullptr;
    void* native_ = nullptr;                // the Q8_0 GGUF tensors' device slab
    void* expert_slab_ = nullptr;           // every layer's expert blobs
    uint64_t expert_slab_bytes_ = 0;
    std::vector<ExpertLayer> expert_layers_;
    std::vector<Tensor> tensors_;           // dense F32 and native entries alike
    std::vector<std::string> names_;
    int64_t context_ = 0;

public:
    uint64_t dense_bytes() const;
    uint64_t native_bytes() const { return native_bytes_; }
    uint64_t expert_bytes() const { return expert_slab_bytes_; }

private:
    uint64_t native_bytes_ = 0;
};

/// The loaded model: owns the weights, makes one session per conversation.
class LoadedInkling : public model::LoadedModel {
public:
    explicit LoadedInkling(std::unique_ptr<Weights> w) : w_(std::move(w)) {}
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return w_->context(); }
    model::Result make_session(std::unique_ptr<model::ModelSession>& out) override;

    model::ModelDesc desc_;
    std::unique_ptr<Weights> w_;
    std::string dump_dir;                 // --diag-dump: the session writes its per-step trace there
};

}  // namespace basalt::models::inkling2
