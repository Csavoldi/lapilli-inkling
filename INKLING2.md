# Inkling 2 Mini on V100 (this fork's current work)

Inkling 2 Mini (the `inkling2_loop` looped-MoE architecture, 26.2B total / 3.4B active) running on
NVIDIA V100 (sm_70), via a port of upstream Basalt v0.5.5's Inkling 2 support into this Lapilli fork.

Upstream v0.5.5 targets Blackwell (sm_120a: fp4/fp8 tensor cores, clusters, TMA). This port keeps
the fork's Volta lanes alive — the `basalt::volta::` shims, dp4a MMQ, emulated mma — so the whole
Inkling 2 pipeline runs on sm_70.

## Status (validated 2026-11-21, 2×V100 PCIe 32GB, one card in use)

| gate | result |
|---|---|
| Logits vs llama.cpp reference | 5/5 top-1 match, KL 0.0001–0.0005, corr ≥ 0.9997 |
| Decode | 69 tok/s (empty ctx) → 60.5 (42K) → 44.4 (165K) |
| Prefill | 850 t/s (short) → 439 t/s (42K) → 211 t/s (165K) — the bottleneck lane |
| Context | 512K configured and benchmarked (model is native 1M) |
| VRAM | ~16 GB weights (fits one V100 beside its KV) |

Weights are the official release: experts NVFP4 **bit-for-bit** (no requant), dense MXFP8→Q8_0,
embedding BF16 pinned in host RAM.

## Branches

- **`inkling2-port`** (default) — the port: upstream v0.5.5 merged onto the fork's `7588701` base,
  13 hand-resolved conflicts keeping the fork's Volta paths, plus sm_70 fixes to the inkling2
  kernels (attn cluster→split path, attn_prompt ldmatrix/mma/cp.async via Volta shims,
  L2-prefetch no-op, tf32 tiles) and GGUF-dialect pack support (`inkling2_loop` arch alias,
  dual-prefix metadata keys, per-layer array unwrap).
- **`inkling2-scratch`** — the from-scratch engine attempt that preceded the port (loads, ran at
  10 tok/s, top-1 2/5 vs reference; superseded, kept for its converter work: an NVFP4→Q4_K/Q8_0
  requantizer validated bitwise against the ggml reference).
- **`base-7588701`** — the fork point, matching the production image.

## Known gaps (all speed, not correctness)

- The DSpark speculative drafter needs `window_gemm` kernels that are sm_120a-only; `--spec 1`
  skips it (draftless decode numbers above).
- Prefill on NVFP4 goes through emulated-mma MMQ; the Q4_K/Q8_0 expert lane (dp4a) is the planned
  fix — requant code exists on `inkling2-scratch`.
- `--kv fp8` runs as fp16 on Volta.

## Build

Same as the fork, with the inkling2 model enabled:

```
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=inkling2
cmake --build build -j <n>
```

CUDA 12.6+ (12.9 tested). Pack conversion from the official checkpoint:
`tools/convert/convert.py --checkpoint DIR --out MODEL.basalt` (needs `numpy regex`).

Credits: Thinking Machines Lab for Inkling 2 Mini; jesdga95 for Basalt and its Inkling 2
implementation; chris83-dev for the Lapilli V100 fork this builds on. MIT, as upstream.
