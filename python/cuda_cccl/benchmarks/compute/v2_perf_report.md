# cuda.compute v2 (HostJIT) — Performance Report

**Date:** 2026-05-30  **GPU:** NVIDIA RTX 6000 Ada  **Fix commit:** `5697955ac0`

## Summary

Benchmarking the new HostJIT (`USING_V2`) backend against the legacy NVRTC/LTO
backend (v1) surfaced one severe regression — heavy compute operators were
**~5.6× slower** on v2. Root cause was a missing loop-unroll/SROA step in the
HostJIT optimization pipeline that left user-operator local arrays in local
memory. A pipeline fix restores **parity** with v1 on that kernel and is neutral
elsewhere.

## Initial results

Ran the Python `cuda.compute` benchmark suite (`benchmarks/compute/`) on both
backends via `compare_v1_v2.py`. Most kernels were at parity (they are
memory-bandwidth bound), but a handful of compute/custom-operator kernels
regressed on v2. The outlier was `transform/heavy` (a unary transform whose
operator does ~200 ops over a 64-element local array):

| Benchmark | v2 vs v1 |
|---|---|
| transform/heavy | **~5.6× slower** |
| segmented_reduce/variable_sum | ~1.8× slower |
| transform/complex_cmp | ~7% slower |
| reduce/custom | ~6% slower |
| (most others) | parity |

## Identified problem

Using a HostJIT IR/PTX dump hook (`CCCL_HOSTJIT_DUMP_DIR`) plus
`cuobjdump -res-usage`, the heavy kernel showed:

- v1: `REG:80 STACK:0` — the operator's `uint32[64]` array lives in **registers**.
- v2: `REG:95 STACK:256` + ~670 local load/store (`LDL`/`STL`) instructions — the
  array spilled to **local memory**.

The optimized IR confirmed the cause: HostJIT's `buildPerModuleDefaultPipeline(O2)`
only **partially** unrolled the operator's fixed-trip-count loops, so the backing
`alloca [64 x i32]` kept a dynamic index and SROA could not promote it to
registers. On the v1 path, `ptxas` performs this promotion; the LLVM-NVPTX path
needs full-unroll-then-SROA at the IR level. Result: every operator iteration
became local-memory traffic.

## Resolution

In `c/parallel.v2/src/hostjit/compiler.cpp`, raise LLVM's loop-unroll thresholds
so the operator's small constant-trip loops fully unroll, then re-run the
optimization pipeline so its SROA promotes the now-constant-indexed arrays to
registers. (Committed as `5697955ac0`; also adds the `CCCL_HOSTJIT_DUMP_DIR`
debug hook.)

## New vs old results (Elements = 2²⁸)

Same suite re-run with v1 as the stable baseline, comparing v2 without the fix
vs v2 with the fix (times in µs; ratio = v2/v1, **<1.0 = v2 slower**):

| Benchmark | v1 | v2 (before) | v2 (after) | Effect |
|---|--:|--:|--:|---|
| **transform/heavy** | 16,549 | 93,139 (0.18×) | **17,167 (0.96×)** | **5.4× faster — restored to parity** |
| (all other kernels) | — | — | — | unchanged within noise |

Geomean v2-vs-v1 across the suite improved from **~9% slower → ~3% slower**
(essentially all of the gain from `transform/heavy`).

## Remaining work (separate root causes — not addressed by this fix)

- `segmented_reduce/variable_sum` ~1.8× slower — largest remaining gap; likely the
  v2 segmented-reduce dispatch/signature path, not codegen.
- `segmented_sort/keys [small]` ~10%, `transform/complex_cmp` ~7%,
  `reduce/custom` ~6%.
- `segmented_sort/keys [power]` slipped ~4% *after* the fix (the unroll threshold
  is currently a process-wide setting) — argues for refining the fix to a scoped,
  single appended SROA pass (also halves the doubled JIT compile time).
