// src/models/inkling2/model.cpp - loading an Inkling 2 Mini pack: the dense F32 pool, the native Q8_0
// projections and the routed experts' blobs, all onto the main card.
#include "basalt/models/inkling2/model.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/session.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <set>

namespace basalt::models::inkling2 {
namespace {

constexpr uint64_t kUploadChunk = 16ull << 20;

bool upload_file_range(const std::string& path, uint64_t off, uint64_t bytes, void* dst, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    void* stage = nullptr;
    if (cudaHostAlloc(&stage, kUploadChunk, cudaHostAllocDefault) != cudaSuccess) {
        std::fclose(f);
        err = "cudaHostAlloc for the pack upload failed";
        return false;
    }
    std::fseek(f, (long) off, SEEK_SET);
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t n = bytes - done < kUploadChunk ? bytes - done : kUploadChunk;
        if (std::fread(stage, 1, n, f) != n) {
            cudaFreeHost(stage);
            std::fclose(f);
            err = "short read from " + path;
            return false;
        }
        if (cudaMemcpy((uint8_t*) dst + done, stage, n, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFreeHost(stage);
            std::fclose(f);
            err = "the pack upload to the device failed";
            return false;
        }
        done += n;
    }
    cudaFreeHost(stage);
    std::fclose(f);
    return true;
}

/// The experts v5/v7 layout lines: per layer (gu_type, d_type, offset, blob_bytes), with the total and count.
struct ExpertLayoutText {
    int n_layers = 0, n_expert = 0;
    std::vector<int> gu, dn;
    std::vector<uint64_t> off, blob;
    uint64_t total = 0;
};

bool parse_expert_layout(const std::string& text, ExpertLayoutText& out, std::string& err) {
    int version = 0;
    const char* p = text.c_str();
    char line[512];
    while (*p) {
        const char* e = std::strchr(p, '\n');
        const size_t n = e ? (size_t) (e - p) : std::strlen(p);
        if (n >= sizeof line) {
            err = "experts.layout: a line too long";
            return false;
        }
        std::memcpy(line, p, n);
        line[n] = 0;
        p = e ? e + 1 : p + n;
        if (line[0] == '#') {
            if (std::strncmp(line, "# experts v", 11) == 0) version = std::atoi(line + 11);
            const char* at = std::strstr(line, "(n_expert ");
            if (at) out.n_expert = std::atoi(at + 10);
            continue;
        }
        if (!line[0]) continue;
        int l = 0, gt = 0, dt = 0;
        unsigned long long o = 0, b = 0;
        if (std::sscanf(line, "%d %d %d %llu %llu", &l, &gt, &dt, &o, &b) != 5) {
            err = "experts.layout: a malformed line: " + std::string(line);
            return false;
        }
        if (l != out.n_layers) {
            err = "experts.layout: the layers are not contiguous";
            return false;
        }
        out.gu.push_back(gt);
        out.dn.push_back(dt);
        out.off.push_back(o);
        out.blob.push_back(b);
        out.n_layers++;
    }
    if (version > 7) {
        err = "experts.layout v" + std::to_string(version) + " is newer than this engine reads";
        return false;
    }
    if (!out.n_layers || out.n_layers != kLayersPhys || out.n_expert != kExperts) {
        err = "experts.layout: " + std::to_string(out.n_layers) + " layers of " + std::to_string(out.n_expert) +
              " experts; the engine reads " + std::to_string(kLayersPhys) + " of " + std::to_string(kExperts);
        return false;
    }
    out.total = out.off.back() + out.blob.back() * (uint64_t) out.n_expert;
    return true;
}

}  // namespace

Weights::~Weights() {
    if (arena_) cudaFree(arena_);
    if (native_) cudaFree(native_);
    if (expert_slab_) cudaFree(expert_slab_);
}

uint64_t Weights::dense_bytes() const {
    return table_.report().arena_bytes;
}

const Tensor* Weights::find(const std::string& name) const {
    for (size_t i = 0; i < names_.size(); ++i)
        if (names_[i] == name) return &tensors_[i];
    return nullptr;
}

bool Weights::load(const std::string& pack_path, int64_t max_context, std::unique_ptr<Weights>& out, std::string& err) {
    core::ModelFile mf;
    if (!mf.open(pack_path, err) || !mf.load_meta(err) || !mf.register_gguf_view(err)) return false;
    core::ModelSources srcs;
    core::model_sources(mf, srcs);
    core::PackFiles pack = core::PackFiles::from(srcs);

    std::unique_ptr<Weights> w(new Weights());

    // The GGUF view: the tensors kept in their GGUF form (the Q8_0 projections, the embedding and the head). They
    // are uploaded below as they are, so the dense pool's loader leaves exactly those out (its other rows are the
    // small float tensors, which the pack carries in both places).
    const GgufFile gguf(pack_path);
    std::set<std::string> skip;
    for (const TensorInfo& t : gguf.tensors())
        if (t.type == 8) skip.insert(t.name);

    // the dense F32 pool
    uint64_t pool = 0;
    if (!core::WeightTable::pool_bytes(pack, pool, err, &skip)) return false;
    if (pool > 0) {
        if (cudaMalloc(&w->arena_, pool) != cudaSuccess) {
            err = "the dense pool (" + std::to_string(pool >> 20) + " MiB) does not fit on the device";
            return false;
        }
        if (!w->table_.load(pack, w->arena_, pool, err, &skip)) return false;
    }
    struct NativeTodo {
        const TensorInfo* t;
        uint64_t at;      // where in the native slab
    };
    std::vector<NativeTodo> todo;
    uint64_t native_total = 0;
    for (const TensorInfo& t : gguf.tensors()) {
        if (t.type != 8) continue;                   // only Q8_0 rows are mmvq-native here
        const uint64_t bytes = tensor_payload_bytes(t);
        if (bytes == 0) {
            err = "tensor " + t.name + " has no payload";
            return false;
        }
        todo.push_back({&t, native_total});
        native_total += (bytes + 4095) / 4096 * 4096;
    }
    if (native_total > 0) {
        if (cudaMalloc(&w->native_, native_total) != cudaSuccess) {
            err = "the native projections (" + std::to_string(native_total >> 20) + " MiB) do not fit on the device";
            return false;
        }
        for (const NativeTodo& n : todo) {
            const uint64_t bytes = tensor_payload_bytes(*n.t);
            if (!upload_file_range(pack_path, gguf.data_start() + (uint64_t) n.t->offset, bytes,
                                   (uint8_t*) w->native_ + n.at, err))
                return false;
            w->tensors_.push_back({(const uint8_t*) w->native_ + n.at, 8, n.t->shape[0],
                                   n.t->shape.size() > 1 ? n.t->shape[1] : 0});
            w->names_.push_back(n.t->name);
        }
    }
    w->native_bytes_ = native_total;

    // the routed experts: the whole section as one device slab
    std::string layout_text;
    {
        std::vector<uint8_t> bytes;
        if (!mf.read("experts.layout", bytes, err)) return false;
        layout_text.assign((const char*) bytes.data(), bytes.size());
    }
    ExpertLayoutText lay;
    if (!parse_expert_layout(layout_text, lay, err)) return false;
    const core::FileRange exp = mf.range("experts");
    if (exp.bytes != lay.total) {
        err = "the experts section is " + std::to_string(exp.bytes) + " B but its layout wants " + std::to_string(lay.total);
        return false;
    }
    if (cudaMalloc(&w->expert_slab_, lay.total) != cudaSuccess) {
        err = "the expert blobs (" + std::to_string(lay.total >> 30) + " GiB) do not fit on the device; this build keeps "
              "every expert in VRAM";
        return false;
    }
    if (!upload_file_range(pack_path, exp.offset, exp.bytes, w->expert_slab_, err)) return false;
    w->expert_slab_bytes_ = lay.total;
    w->expert_layers_.resize(kLayersPhys);
    for (int l = 0; l < kLayersPhys; ++l) {
        w->expert_layers_[l] = {(const uint8_t*) w->expert_slab_ + lay.off[(size_t) l], lay.gu[(size_t) l],
                                lay.dn[(size_t) l], lay.blob[(size_t) l]};
    }

    // the dense F32 tensors as the session reads them
    for (const auto& [name, ref] : w->table_.all()) {
        if (ref.bytes == 0) continue;              // the native rows carry no pool bytes
        w->tensors_.push_back({ref.data, 0, ref.ne0, ref.ne1});
        w->names_.push_back(name);
    }

    w->context_ = max_context > 0 ? max_context : 65536;
    if (w->context_ > 1048576) w->context_ = 1048576;
    log::info("basalt inkling2", "dense %llu MiB, native %llu MiB, experts %llu GiB, context %lld",
              (unsigned long long) (w->dense_bytes() >> 20), (unsigned long long) (native_total >> 20),
              (unsigned long long) (lay.total >> 30), (long long) w->context_);
    out = std::move(w);
    return true;
}

model::Result LoadedInkling::make_session(std::unique_ptr<model::ModelSession>& out) {
    std::string err;
    std::unique_ptr<Session> s(new Session(*w_, dump_dir));
    if (!s->init(err)) return {model::Status::Error, err};
    out = std::move(s);
    return {};
}

}  // namespace basalt::models::inkling2
