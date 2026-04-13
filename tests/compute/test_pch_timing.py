#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
PCH (Pre-Compiled Header) timing benchmark.

Measures build time for three scenarios:
  1. No PCH  — baseline, compiles from scratch every time
  2. PCH cold — first call with PCH enabled: generates PCH then compiles
  3. PCH warm — second call with PCH enabled: reuses the cached PCH on disk

Usage:
    python test_pch_timing.py           # run benchmark (uses existing PCH cache)
    python test_pch_timing.py --clear   # delete cached PCH files first, then run
    python test_pch_timing.py --verbose # show compiler diagnostics (PCH usage lines)
"""

import argparse
import glob
import os
import time

import cupy as cp
import numpy as np

PCH_DIR = "/tmp/clangjit_pch"
N = 1 << 20  # 1M elements, large enough that kernel launch cost is irrelevant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_reduce(d_in: cp.ndarray, d_out: cp.ndarray, build_config) -> float:
    """
    Time the *build* step of a device-wide int32 sum reduction.
    Returns wall-clock seconds.
    """
    from cuda.compute._bindings import Determinism, DeviceReduceBuildResult, OpKind
    from cuda.compute._cccl_interop import (
        call_build,
        get_value_type,
        to_cccl_input_iter,
        to_cccl_output_iter,
        to_cccl_value,
    )
    from cuda.compute.op import make_op_adapter

    h_init = np.zeros(1, dtype=d_in.dtype)

    d_in_cccl = to_cccl_input_iter(d_in)
    d_out_cccl = to_cccl_output_iter(d_out)
    h_init_cccl = to_cccl_value(h_init)

    op_adapter = make_op_adapter(OpKind.PLUS)
    value_type = get_value_type(h_init)
    op_cccl = op_adapter.compile((value_type, value_type), value_type)

    t0 = time.perf_counter()
    call_build(
        DeviceReduceBuildResult,
        d_in_cccl,
        d_out_cccl,
        op_cccl,
        h_init_cccl,
        Determinism.RUN_TO_RUN,
        build_config=build_config,
    )
    return time.perf_counter() - t0


def clear_pch_cache():
    removed = glob.glob(os.path.join(PCH_DIR, "*.pch"))
    for f in removed:
        os.remove(f)
    return removed


def list_pch_files():
    files = sorted(glob.glob(os.path.join(PCH_DIR, "*.pch")))
    if not files:
        print("    (none)")
        return
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(f)))
        print(f"    {os.path.basename(f):44s}  {size_kb:7.1f} KB  @ {mtime}")


def run_scenario(label: str, enable_pch: bool, verbose: bool = False) -> float:
    from cuda.compute._bindings import BuildConfig

    cfg = BuildConfig(enable_pch=enable_pch, verbose=verbose)
    d_in = cp.ones(N, dtype=np.int32)
    d_out = cp.zeros(1, dtype=np.int32)
    t = _build_reduce(d_in, d_out, cfg)
    tag = "PCH" if enable_pch else "   "
    print(f"  [{tag}] {label:<38s}  {t:.2f}s")
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--clear", action="store_true", help="Remove cached *.pch files before running"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print compiler diagnostics (PCH hit/miss lines)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("PCH timing benchmark  —  device int32 sum-reduce, N=2^20")
    print("=" * 65)

    if args.clear:
        removed = clear_pch_cache()
        if removed:
            print(f"\nCleared {len(removed)} PCH file(s) from {PCH_DIR}")
        else:
            print(f"\nNothing to clear in {PCH_DIR}")

    print("\nPCH cache before benchmark:")
    list_pch_files()

    # ------------------------------------------------------------------
    # Warmup: initialise LLVM, load Python modules, resolve include paths.
    # This run is discarded so it doesn't inflate the baseline.
    # ------------------------------------------------------------------
    print("\nInitialising (discarded) ...")
    run_scenario("init (discarded)", enable_pch=False)

    # ------------------------------------------------------------------
    # Actual scenarios
    # ------------------------------------------------------------------
    print()
    t_no_pch = run_scenario("No PCH (baseline)", enable_pch=False)
    t_cold = run_scenario(
        "PCH cold (generate + compile)", enable_pch=True, verbose=args.verbose
    )
    t_warm = run_scenario(
        "PCH warm #1 (reuse cached PCH)", enable_pch=True, verbose=args.verbose
    )
    t_warm2 = run_scenario(
        "PCH warm #2 (confirm)", enable_pch=True, verbose=args.verbose
    )

    print("\nPCH cache after benchmark:")
    list_pch_files()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  No PCH (baseline)        {t_no_pch:6.2f}s")
    print(
        f"  PCH cold (gen+compile)   {t_cold:6.2f}s  ({t_cold - t_no_pch:+.2f}s vs baseline)"
    )
    print(
        f"  PCH warm #1              {t_warm:6.2f}s  ({t_warm - t_no_pch:+.2f}s vs baseline)"
    )
    print(
        f"  PCH warm #2              {t_warm2:6.2f}s  ({t_warm2 - t_no_pch:+.2f}s vs baseline)"
    )
    if t_warm > 0 and t_no_pch > 0:
        speedup = t_no_pch / t_warm
        print(f"\n  Warm PCH speedup         {speedup:.1f}x faster than no-PCH")
        print(f"  One-time cold overhead   +{max(0, t_cold - t_no_pch):.2f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
