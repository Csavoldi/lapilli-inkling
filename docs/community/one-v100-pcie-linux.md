# One V100 PCIe on Linux: a user's report

Reported by **u/nebod99** on Reddit, November 2026, a few hours after this fork was published. It is the first run of
Lapilli outside the machine it was developed on, and it covers three things the port had not tried: **Linux**, a **single
V100**, and a **PCIe card**.

## Thank you, nebod99

Before this report, Lapilli had only ever run on one machine. nebod99 built it on a platform it had never seen, ran it on a
single card next to the engine already in use there, measured both the same way, and wrote down everything someone else
needs to do the same: the build flags, the Docker image, the pack revision and how to convert it. Thanks to that work, every
V100 owner now knows the port runs on Linux, on one card and on PCIe, at about 100 tokens/s.

Thank you, sincerely, for the time, the care and the generosity of sharing it. This page exists because of you.

The figures below are copied as reported. They are initial measurements, taken from each engine's own timing output.

## The machine

| | |
|---|---|
| GPU | 1 x Tesla V100 PCIe 32 GB |
| CPU | Intel Core i9-10940X |
| System RAM | 96 GB |
| OS | Linux, in a CUDA 12.6.3 Docker image |
| Pack | Qwen3.8-Flash-Next GSQ-RCO `IQ3_S` (revision v1, converted locally to v2) |
| Lapilli build | commit `7588701`, `-DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=qwen4exp` |
| Lapilli options | int8 KV cache, memory-mapped and resident experts |

## The test

Thinking enabled. The long prompt is 7,242 input tokens, and every run generates 1,024 tokens. The long prompt fills about
7.2K tokens of the context window.

| Engine, configured context | Decode, short prompt | Decode, long prompt | First token, long prompt | Launch to ready |
|---|---:|---:|---:|---:|
| Strata, 128K | 57.7 tokens/s | 55.6 tokens/s | 22.6 s | 70 s |
| **Lapilli, 16K** | **112.2 tokens/s** | **107.3 tokens/s** | **8.8 s** | 124 s |
| **Lapilli, 128K** | **99.2 tokens/s** | **100.9 tokens/s** | **8.7 s** | 90 s |

## What it shows

- **One V100 is enough.** The full Qwen3.8-Flash-Next runs at about 100 tokens/s on a single PCIe card, with a 128K context.
- **The first token comes 2.6 times sooner** than with Strata on the same machine: under 9 seconds instead of about 23, once
  the model is loaded.
- **Long-prompt decode is about 1.8 times faster** than Strata at the same configured context.
- **Startup is slower than Strata's** on this machine (90 to 124 s against 70 s). It is paid once per launch. Launch time
  depends a lot on the disk and on what the OS already has in its file cache, so the order of the runs matters here.

## Building it on Linux

The port was only developed on Windows; this report shows that the same tree builds and runs on Linux without changes:

```sh
cmake -S . -B build70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=qwen4exp
cmake --build build70
```

with CUDA 12.6 (here the 12.6.3 Docker image). A pack published in Basalt's v1 format is converted once with Basalt's own
tool, `tools/migrations/basalt_v1_to_v2.py` (see [tools/migrations/README.md](../../tools/migrations/README.md)).

Back to the [README](../../README.md).
