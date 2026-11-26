# DSpark drafter on Volta

Date: 2026-11-21
Branch: `inkling2-port`
Status: draft, reviewed three times, awaiting the user
Scope: this spec covers the drafter only. A second card for the experts is a separate spec.

## Goal

Make the pack's DSpark drafter run on a V100 (sm_70), then measure whether it raises decode
speed. Today, on one V100 32 GB with the official NVFP4 pack, decode is:

| Context fill | Decode |
|---|---|
| empty | 69 tok/s |
| 42K tokens | 60.5 tok/s |
| 165K tokens | 44.4 tok/s |

The user's target is 100 tok/s or more. This spec does not promise it. Its result is a measured
table, and a decision based on that table.

## What blocks the drafter today

With a drafter in the pack the engine fails at start:

    window gemm: occupancy: no kernel image is available for the device

The drafter computes its matrix products with the `window_gemm` library, compiled for sm_120a only
(`CMakeLists.txt`, `basalt_arch_120a(window_gemm)`). `--spec 1` avoids it by never loading the
drafter.

## Facts the design rests on

Each was read from the source on `inkling2-port`.

1. The pack stores the drafter's seven matrices per layer as BF16. They are converted to Q8_0
   while loading. Then `dspark.cpp` rounds each to NVFP4 in place. The comment there gives the
   reason this is allowed: the rounding affects drafts only, and every draft is verified exactly.
2. `window_gemm` has four formats: `F8`, `Nvfp4`, `Bf16` (`[rows][K]`), and `Q8`. `Q8` means Q8_0
   repacked into planes: int8 codes `[rows][K]`, then fp16 scales `[rows][K/32]`
   (`include/basalt/kernels/window_gemm.hpp`). It is not the ggml block layout.
3. The `Bf16` and `Q8` kernels are templates in `window_gemm_f8.cu` (`make_kernel<...>`), built
   over `window_gemm.cuh`.
4. `kernel_for` (`src/kernels/cuda/window_gemm.cu`) refuses SwiGLU for every format but `Nvfp4`.
   The drafter's gate/up call uses SwiGLU (`dspark.cpp`, `gu.epi = wg::Epilogue::SwiGLU`).
5. `window_gemm.cuh` uses `bf16x2_mul`, which is Blackwell-only. Below sm_100 it compiles to
   `__trap()` (`include/basalt/kernels/volta_sm100.hpp`, `BASALT_SM100_ASM`).
6. The dense stack already converts Q8_0 blocks to planes at load with `wg::q8_planes`
   (`src/dense/model.cpp`).
7. The drafter's buffers are allocated at Q8_0 size before the rounding step, so keeping Q8_0 adds
   no VRAM. Q8_0 blocks and Q8 planes both take `rows * (K/32) * 34` bytes.
8. `ws_need` (`dspark.cpp`) sizes the drafter's workspace. It builds its calls with `Format::Nvfp4`
   and runs at load, before any weight is read.
9. The model verifies every draft. So the output text is the same whether the drafter is right,
   weak, or broken. A broken drafter shows only as a low acceptance rate.
10. `--spec N` caps the window at N tokens, so at most N-1 drafts per round. Without the flag a
    pack with a DSpark drafter starts at `--spec 8`, which is 7 drafts
    (`dspark.cpp:740`, `kDsBlock = 8`). The help text says "6 with a DSpark or DFly2 one"
    (`src/program/cli/help.cpp:113`). The help text is wrong about this pack.
11. `--spec N` is a cap. A policy picks each round's window from the drafter's own confidences
    (`src/spec/draft_policy.cpp`), so two caps can end up running the same windows.

## Design

### 1. Split the compile target

`window_gemm_nvfp4.cu` stays on sm_120a. `window_gemm.cu` and `window_gemm_f8.cu` build for sm_70
too. No stub source is needed: the NVFP4 kernels are not in those two files, and step 6 refuses
the format before any lookup.

`basalt_arch_120a` builds for the chosen architecture on CUDA 12.6 and 12.7, and for sm_120a on
12.8 and later. This fork is built inside the CUDA 12.9 image, which is the case the design is
written for.

The FP8 kernels compile for sm_70 and must never run there. They reach a different instruction
from the rest of the file: `e4m3x2_bf16x2` in `include/basalt/kernels/narrow_bf16.cuh` goes
through `cvt.rn.f16x2.e4m3x2`, which is Blackwell-only and becomes a trap below sm_100. So a
successful compile proves nothing about FP8. Step 6 is what makes it safe.

### 2. A Volta multiply for `window_gemm.cuh`

`bf16x2_mul` is the only Blackwell-only instruction in `window_gemm.cuh` and in the Q8 and BF16
kernels. A grep of `window_gemm.cuh`, `window_gemm_sliced.cuh`, `window_gemm.cu` and
`window_gemm_f8.cu` for `BASALT_SM100_ASM`, `ldmatrix`, `mma.sync`, `tcgen`, `cp.async.bulk` and
`griddep` finds one hit, `window_gemm.cuh:96`. The FP8 path's conversion helpers are the
exception, described in step 1. Everything else in these files is ordinary CUDA that already goes
through the Volta shims.

Give `bf16x2_mul` a sm_70 path: two bf16 values widened to fp32, multiplied, rounded back to bf16.
Exact against the instruction, because a product of two bf16 values rounded once to bf16 is the
same either way. The Blackwell instruction stays for sm_100 and later.

### 3. Separate the gate and up products

SwiGLU as one fused call exists only for NVFP4. On Volta the drafter issues the gate and up
matrices as two `Format::Q8` calls with `Epilogue::Store`, into two scratch rows, then applies the
activation to the two results with one small elementwise kernel next to the existing `dspark_*`
kernels.

Not `Epilogue::Add`. `Add` means "add into the residual", which is right for the output and down
projections and wrong here.

The activation is plain `silu(gate) * up`, with `silu(x) = x * sigmoid(x)`. That is what the fused
epilogue computes (`include/basalt/kernels/window_gemm.hpp`, `Epilogue::SwiGLU`;
`src/kernels/cuda/window_gemm.cuh:383`). It is not the SITU form the model's own experts use, with
its softcaps. The two are easy to confuse, and the wrong one would only lower the acceptance rate,
which looks like a slow drafter rather than a broken one.

The fused path rounds its output to bf16. The elementwise kernel applies the activation in fp32
and rounds once to bf16, so the rounding matches.

### 4. One switch for the drafter's format

`ws_need` (`src/models/inkling2/dspark.cpp`) sizes the split-K workspace, and it builds its
`wg::Call` objects with `Format::Nvfp4` and one SwiGLU call. It runs at load, before any weight
is read, and reaches the kernel table through `resident()`. Left as it is, the engine still exits
at start with the same error after every other step is done.

So both places change together: `ws_need` and the per-layer call sites in `stage`. Each takes the
format and the gate/up shape from one place, a small function that returns the `wg::Call` layout
for the device. One device query at load decides it. The sm_120 path is unchanged.

The Q8 calls pass a null scale. The `sc + i` argument the NVFP4 path passes to `mat()` is one
multiplier per matrix (`wg::Matrix::scale`), and the kernel multiplies every product by it for
every format (`src/kernels/cuda/window_gemm.cuh:390`). It is not part of the NVFP4 data. It
exists because 4-bit codes cannot hold the weights' range: `from_q8` divides the matrix by a
global scale before rounding and stores it in `L.fp4_scale[i]`, and the kernel multiplies it back
(`include/basalt/kernels/nvfp4_planes.hpp`). Q8 holds the weights at their real values, so the
multiplier is exactly 1.0, which is what a null scale gives.

Passing the old pointer instead would read an uninitialised float, because only `from_q8` writes
`L.fp4_scale` (`dspark.cpp:264`). The drafter would then produce garbage drafts, and the only
symptom would be a low acceptance rate.

### 5. Keep the drafter's matrices as Q8_0 planes on Volta

At load, when the device is below sm_100: skip the NVFP4 rounding, run `wg::q8_planes` on each
matrix into a scratch buffer, and copy back into the block buffer. Both layouts take
`rows * (K/32) * 34` bytes, so it fits. The dense stack already does this conversion at load
(`src/dense/model.cpp`).

### 6. Refuse, do not trap

`kernel_for` (`src/kernels/cuda/window_gemm.cu`) refuses `Format::Nvfp4` and `Format::F8` below
sm_100 through the library's existing `refuse()`, which logs the reason and calls `exit(1)`. That
is how it already treats every other call it cannot run. The drafter never reaches these formats
after step 4, so this protects the next caller, and it is what keeps the compiled FP8 kernels from
being launched by accident.

## Gates

Run in this order. A failed gate stops the work and the result is reported. GPU gates run on one
V100 with production stopped.

| # | Gate | How it runs | Pass condition |
|---|---|---|---|
| 1 | Build | `cuobjdump --list-elf bin/basalt` | Q8 and BF16 `window_gemm` kernels for sm_70, NVFP4 for sm_120a only |
| 2 | Refusal | A small new test, `window_gemm_refuse`, built like `window_gemm_parity`. It takes a format name, makes one `window_gemm` call of that format, and is run twice as a child process, once with `f8` and once with `nvfp4` | Each child logs the refusal naming the format and exits with status 1. Neither hangs in a trap |
| 3 | Kernel parity | `window_gemm_parity` on the V100 | Its Q8 and BF16 cases pass. F8 cases report "skipped" |
| 4 | Activation | A small new test beside the kernel | Each output is within one bf16 step of a CPU double reference over 1M random fp32 pairs in [-40, 40] |
| 5 | Drafter parity | `inkling2_dspark_parity` on the V100 | Passes |
| 6 | Draftless unchanged | The existing logits run: `--diag-logits FILE --diag-serial-prompt --max-new 0 --spec 1`, then the comparison script against `ref5.bin` | 5 of 5 top-1 and KL under 0.001 |
| 7 | Text | Generate mode, the fixed prompt, 300 tokens, once with `--spec 1` and once with `--spec 4` | Identical token sequences |
| 8 | The drafter drafts | Serve mode, `--spec 4`, a 2000-token wikitext prompt, 300 tokens generated | The request's log line shows 50% or more of offered drafts accepted |

Gate 2 needs a new test because `refuse()` ends the process, so a test that made the call in its
own process would never get to report. The engine's `--exp` flag is no help: it has one probe,
a logit hash (`src/program/cli/help.cpp:45`), and nothing that reaches `window_gemm`. The test
therefore links the library itself. Its parent process is what checks the exit status and the
message, and is what the test runner sees.

Gate 3 needs a change to the test. `window_gemm_parity` runs its F8 shapes first
(`tests/kernels/window_gemm_parity.cpp:597`). With step 6 in place those calls would exit the test
before the Q8 case runs. The test asks the library whether a format is runnable before each shape,
and reports "skipped" for one that is not. A shape that fails for any other reason still fails the
test. The question is one new query function in the library, next to the refusal, so the two
cannot disagree.

The test's only multi-matrix case is an F8 one, so it is skipped on a V100. The drafter makes a
three-matrix Q8 call (`dspark.cpp:517`). That call shape is not covered by gate 3. Gate 5 and gate
8 cover it as part of the drafter.

Gate 4 uses "within one bf16 step" rather than bit-identity. The kernel computes the exponential
in single precision and the reference in double, and they differ in the last bit for a few values
in a million. A correct kernel fails a bit-identity test.

Gate 5 covers the drafter's own kernels. It does not run `stage`, the function this change edits,
so it cannot catch a wiring error there. Gate 8 is the check for that.

Gate 8 exists because gate 7 cannot fail on a broken drafter. The model verifies every draft, so
identical text is what a correct drafter and a broken one both produce. A broken drafter shows up
only as low acceptance. The floor of 50% comes from capy-flash's 77% at K=3 on RTX 5070 Ti cards,
with room for the V100's different rounding. It separates a working drafter from a mis-wired one.
It does not prove the drafter is exact. The engine's `--draft-log FILE` adds one line per window
with its size, the number accepted and the drafter's confidences
(`src/program/serve/session_engine.cpp:561`). The log from this run is kept with the results.

Gate 6 is a regression check on the whole build, because step 2 changes a file that other kernels
include.

## Measurement

With gates 1 to 8 passed, run the engine in `--serve` on one V100, production stopped.

Drafts are capped by `--spec N`, which allows at most N-1 drafts per round. Run `--spec` 2, 3, 4,
5 and 6, which is 1 to 5 drafts, and also what the engine does with no `--spec` flag, which is 7
drafts for this pack. That last row is what a user gets by default, so it has to be in the table.

| `--spec` | Most drafts per round | Context fill | Record |
|---|---|---|---|
| 2, 3, 4, 5, 6, none | 1, 2, 3, 4, 5, 7 | empty, 8K, 42K | tok/s, tokens per round, acceptance rate |

The baseline is measured in the same session, the same way: `--spec 1` at each of the three fills,
three runs, median. Today's numbers (69, 60.5, 44.4 tok/s) were single runs at other fills, and
there is no 8K figure among them. The table is read against the measured baseline, not against
them.

Prompts: fixed wikitext prefixes, 300 generated tokens each, temperature 0, three runs each, take
the median. The serve log prints tok/s and "drafts accepted N of M" at the end of each request
(`src/program/serve/request_finish.cpp`).

Two `--spec` caps can produce the same windows, because the policy chooses each round's window
from the drafter's confidences. If two rows come out identical, the table says so and keeps both.

## Decision rule

Read the table two ways. First against the measured baseline without a drafter at the same fill.
Second against the user's target of 100 tok/s. Both go in the README.

| Result at the best setting | Action |
|---|---|
| 20% or more over the baseline | Drafter on by default for this pack on Volta, at that setting. README gets the table. |
| Under 20% | Code stays. This pack on Volta starts with the drafter off. README says the drafter costs more than it gives on V100, with the table. |
| Any gate fails and the fix is larger than a day | Stop. Report which gate and why. |

Today the drafter is on by default for a pack that has one: 7 drafts unless `--spec` says
otherwise. "Off by default" in the second row means changing that for this pack on Volta. It is
one condition at the place the default is chosen.

20% is the smallest gain a user would notice without measuring. It is a judgement, not a derived
number.

Why the drafter may lose: one round costs a forward at T = K+1 (48 layer visits) plus the
drafter's own three layers. capy-flash measured this drafter on two RTX 5070 Ti at 149 against
127.5 tok/s at K=3, and 166.6 against 112.8 at 12.8K context. A V100 has roughly one eighth the
tensor throughput, so its rounds cost more. Nothing has been measured.

## Not in this scope

- A second card. Separate spec.
- The fine-tuned drafter published on Hugging Face. capy-flash measured +16 to 23% on Italian and
  about no change on English.
- New BF16 drafter kernels in the style of capy-flash. The Q8_0 route needs less new code. If gate
  5, gate 8 or the measurement fails, that route is the next thing to try and gets its own spec.
- Prefill speed. The dp4a fast path on sm_70 covers Q8_0, Q5_1, Q4_K and Q5_K, not NVFP4.

## Credits

- Basalt (jesdga95), MIT, and Lapilli (chris83-dev), MIT: the engine, the `window_gemm` library
  and the DSpark drafter this change builds on.
- capy-flash (ValerioDolci), Apache-2.0: the measurements quoted above, and the demonstration that
  this drafter runs without tensor cores.

No code is copied from capy-flash. If a later step ports a kernel or a function from it, that file
carries its copyright and licence header and `THIRD_PARTY_LICENSES` gains an entry that keeps its
NOTICE chain, as the Apache licence requires. capy-flash's NOTICE lists six upstream projects of
its own (NInfer and five forks); an entry that drops those is not a valid credit.

## Production

The GPU VM runs the user's main model on both cards. The user has allowed stopping it for this
work at any time, with the condition that it is started again after each measurement window. Every
window runs under a watchdog that restarts it automatically.

A restart is confirmed by the server's own log line, `[server] weights  loaded, ctx=...`, and then
by a real request completing. The HTTP port answers before the engine is ready, so a 200 from the
port proves nothing.

## Estimate

| Step | Time | Confidence |
|---|---|---|
| Build split, Volta multiply, refusal and its test, the parity test's skip | half a day | High |
| Gate/up split and silu kernel, with its test | half a day | High |
| One format switch (`ws_need` and `stage`), Q8 planes at load | half a day | Medium: it touches the function that reads every draft |
| Gates on the V100, and the baseline measurement | half a day | Medium: none of these tests has run for this model on a V100 |
| Measurement and write-up | half a day | High |

Total: about two and a half days. The estimate does not include what happens if the measurement
says the drafter does not help.
