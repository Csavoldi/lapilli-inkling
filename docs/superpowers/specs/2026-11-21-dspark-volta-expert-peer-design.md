# DSpark on Volta and a two-card expert tier for Inkling 2 Mini

Date: 2026-11-21
Branch: `inkling2-port`
Status: draft, awaiting review

## Goal

Two changes, in this order, each measured before the next starts:

1. Make the pack's DSpark drafter run on V100 (sm_70) and find out whether it raises decode speed.
2. Let Inkling 2 Mini use a second V100 for its experts, with peer access (P2P) on.

Target: 100 tok/s or more at decode, on two V100 32 GB cards and on two V100 16 GB cards.

The 100 tok/s target is a goal, not a promise. Nothing in this document has been measured on two
cards or with a drafter. Each phase ends with a measurement that decides whether the next is worth
doing.

## Where we start

Measured on one V100 32 GB, this build, the official NVFP4 pack (2026-11-21):

| Context fill | Decode | Prefill |
|---|---|---|
| empty | 69 tok/s | up to 850 tok/s |
| 42K tokens | 60.5 tok/s | 439 tok/s |
| 165K tokens | 44.4 tok/s | 211 tok/s |

Correctness: 5 of 5 top-1 tokens agree with the llama.cpp reference, KL 0.0001 to 0.0005.

Weights: experts 15.8 GB (20 layers NVFP4, 4 layers Q8_0), dense 1.1 GB, embedding 0.9 GB in host
RAM, drafter 0.48 GB.

P2P works on this pair: with `BASALT_PEER_NO_P2P=0` the engine logs `peer access: 2 of 2 ordered
pairs enabled`. Without that variable peer access stays off on Volta, because upstream found
card-to-card copies returning wrong data on some Volta pair.

## Phase 1: the drafter on Volta

### What blocks it today

With a drafter in the pack the engine fails at start with
`window gemm: occupancy: no kernel image is available for execution on the device`. The drafter
computes its products with the `window_gemm` library, and that library is compiled for sm_120a only
(`CMakeLists.txt`, `basalt_arch_120a(window_gemm)`).

### What is already there

- The pack stores the drafter's matrices as Q8_0. At load, `dspark.cpp` rounds each one to NVFP4
  in place, as a size and speed optimisation for Blackwell. Its comment says why this is allowed:
  the rounding affects drafts only, and every draft is verified exactly.
- `window_gemm.cuh` already goes through the Volta shims (`mma_16816_f32_bf16`, `cp_async_commit`
  and `cp_async_wait`, `cvt_rn_bf16x2`).
- The Q8 and BF16 kernels live in `window_gemm_f8.cu` and are plain template code
  (`make_kernel<Q8, ...>`). Only `window_gemm_nvfp4.cu` is Blackwell-only by content.

### Change

1. **Split the compile target.** `window_gemm_nvfp4.cu` stays at sm_120a. `window_gemm.cu` and
   `window_gemm_f8.cu` build for sm_70 as well. This is a CMake change plus a small source for the
   NVFP4 entry points on sm_70 that refuses with a clear error.
2. **Keep the drafter's matrices as Q8_0 on Volta.** Skip the NVFP4 rounding at load when the
   device is not sm_120, and build the drafter's `wg::Call` objects with `Format::Q8` instead of
   `Format::Nvfp4`. The decision is one device query at load, kept in one place. The sm_120 path
   is untouched.
3. **Guard the other Blackwell-only formats.** `Format::F8` refuses on anything below sm_89 unless
   a caller asks for it deliberately. Nothing in the Inkling path uses it.

Cost: the drafter grows from 0.48 GB to about 0.9 GB of VRAM.

The FP8 kernels in `window_gemm_f8.cu` are compiled by the same unit as the Q8 kernels. If they do
not compile for sm_70, they move into the sm_120a unit with the NVFP4 file. That is found at build
time and does not change the design.

### Not in this phase

- Writing new drafter kernels. capy-flash runs its DSpark drafter with plain BF16 CUDA kernels and
  no tensor cores, which shows that route exists, but the Q8_0 route needs no new kernels and is
  tried first.
- The fine-tuned drafter on Hugging Face. capy-flash measured +16 to 23% on Italian and about no
  change on English. Not worth a pack change now.

### Gates

| Gate | Pass condition |
|---|---|
| Build | `cuobjdump --list-elf` shows the Q8 `window_gemm` kernel for sm_70 and the NVFP4 one for sm_120a only |
| Text | Greedy text for a fixed 300-token prompt is identical with the drafter on and off, and identical to today's build |
| Draftless unchanged | The existing logits comparison still gives 5 of 5 and KL under 0.001 |
| Speed | A table of tok/s and acceptance rate for draft counts 1 to 5, at empty, 8K and 42K fill |

The speed gate has no pass number. Its result decides the next step:

- Drafter gives 20% or more at K=3 or K=5: keep it, make it the default for this pack on Volta.
- Drafter gives less than 20%: keep the code, leave the default off, and say so in the README.

A round costs one forward at T = K+1 plus the drafter. On Volta that forward is 48 layer visits,
so the drafter can lose money. If it does, that is the result.

## Phase 2: experts on the second card

### Why expert tier and not tensor parallel

capy-flash runs this model on two 16 GB cards with tensor parallelism: every expert split across
both ranks, about 10 GB per board. That works without P2P, which consumer boards lack. It costs an
all-reduce per mixture layer, dense work replicated on both cards, and a decode graph that spans
two devices.

This pair has P2P, and the engine already has an expert tier: each expert is read from an address
in a table. Putting experts on card 1 needs no all-reduce and no cross-device graph.

### The seam

inkling2 keeps each layer's experts behind `blob_table[e]`, a device array of one address per
expert (`model.hpp`, `LayerWeights::blob_table`). Decode reads an expert through that address
(`native_expert_grouped`). Experts spilled to host RAM already use this: their entries point into
host memory instead of the card.

So the second card is a third place an entry can point.

### Change

1. **Place the expert pool on the peer.** In `model.cpp`, where each layer's expert range is
   allocated (`device_buf(eb + past)`), allocate on the peer card when one is in use. Reads load
   straight into peer memory.
2. **Fill the table with peer addresses.** The entries become peer addresses, which P2P makes
   dereferenceable from the main card.
3. **Budget.** The placement sum in `model.cpp` (experts + session bytes + spare against free VRAM)
   counts the expert pool against the peer's memory and everything else against the main card's.
4. **P2P on by default for this model on Volta.** Production already runs `BASALT_PEER_NO_P2P=0`
   on these cards. The stack's compose file sets it; the engine default does not change.

### What has to be measured before trusting it

No one has run inkling2's kernels against peer memory. Three things can go wrong, and each has a
check:

| Risk | Check |
|---|---|
| Peer reads are slow at decode sizes (one expert is about 16 KB) | Compare decode tok/s with the pool on the main card and on the peer, same context |
| Kernels assume a single device pointer space | Logits comparison gives 5 of 5 and KL under 0.001 with the pool on the peer |
| Prefill's MMQ path reads experts through a different route than decode | Read `prompt.cpp` and `moe_mmq` for how they address experts before starting; prefill benchmark at 2K and 42K |

The third row is the main unknown. If prefill's MMQ takes a base pointer and computes offsets, the
pool on the peer works for decode and not for prefill. That is read first, before any code.

### Gates

| Gate | Pass condition |
|---|---|
| Logits | 5 of 5 top-1 and KL under 0.001, pool on the peer |
| Decode | tok/s at empty, 8K, 42K, 165K fill, against the one-card numbers |
| Prefill | tok/s at 2K and 42K, against the one-card numbers |
| Long run | 20 minutes of continuous requests, no error, text stays coherent |

## Phase 3: fitting two 16 GB cards

Nothing here can be measured on this machine. There are no 16 GB cards in it.

The experts are 15.8 GB. They do not fit on one 16 GB card next to anything else. So 16 GB needs
both cards to hold experts, about 8 GB each, with the dense weights and caches on the main card
taking the rest of its room, and the elastic spill to host RAM taking what is left over.

1. **Placement by budget.** One function decides, from the free memory of each card: all experts on
   the peer (two 32 GB cards), or split across both cards (two 16 GB cards). Phase 2's table entries
   already allow a per-expert choice of card.
2. **Cache size.** `--max-context` is chosen from what is left. Written down as a table in the
   README: context against free VRAM, per card type.
3. **Drafter.** The drafter's 0.9 GB counts against the main card. On 16 GB cards it is off unless
   the budget allows it.

Phase 3 is code and a budget table. Its measurement happens on real 16 GB cards, when they exist.
Until then the README says 16 GB support is untested.

## Credits

- **Basalt** (jesdga95), MIT: the engine and its Inkling 2 implementation. This fork's base.
- **Lapilli** (chris83-dev), MIT: the V100 port this branch builds on.
- **capy-flash** (ValerioDolci), Apache-2.0, itself descending from NInfer and five other credited
  projects: the reference for how DSpark and two-card Inkling run, and for the drafter and
  tensor-parallel numbers quoted here.

Nothing in Phases 1 to 3 copies code from capy-flash. Its design and its measurements are cited.
If a later step does port a kernel or a function from it, that file carries its copyright and
licence header, and `THIRD_PARTY_LICENSES` gains an entry that keeps the upstream NOTICE chain
intact, as the Apache licence requires.

## Out of scope

- Prefill speed. The dp4a fast path on sm_70 covers Q8_0, Q5_1, Q4_K and Q5_K, not NVFP4, so it
  would need a Q4_K pack. Separate piece of work.
- `--kv fp8`, which runs as fp16 on Volta.
- Vision.
- Tensor parallelism. Revisited only if Phase 3 cannot reach a useful context on 16 GB cards.

## Production

The GPU VM runs the user's main model on both cards. Work on this spec may stop it any time, and
must start it again when a measurement window ends. Every window uses a watchdog that restarts it
automatically. A restart is confirmed by the server log line `weights loaded`, not by the HTTP
port answering.

## Estimates

| Phase | Time | Confidence |
|---|---|---|
| 1, drafter | 1 to 2 days | High on the code, none on whether it helps |
| 2, expert tier | 2 to 4 days | Medium; depends on what the prefill read finds |
| 3, 16 GB placement | 1 day of code | Cannot be measured here |
| Tuning and write-up | 1 day | High |
