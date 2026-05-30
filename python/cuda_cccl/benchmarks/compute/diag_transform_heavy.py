#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Minimal standalone repro for the transform/heavy v1-vs-v2 regression.

Builds the same heavy unary transform the benchmark uses, times it, and dumps
the final *fused* cubin (op inlined into the CUB kernel) so its register/local-
memory usage can be inspected -- the heavy op packs a 64-elem uint32 local array,
so a big v2 slowdown most likely means that array is spilling to local memory
instead of living in registers (i.e. the op isn't being optimized after inlining).

Run it once in the v1 env and once in the v2 env, then diff the cubins:
    python diag_transform_heavy.py            # writes heavy_v1.cubin / heavy_v2.cubin
    cuobjdump -res-usage heavy_v1.cubin
    cuobjdump -res-usage heavy_v2.cubin
    cuobjdump -sass heavy_v2.cubin | grep -cE '\\bLD[LG]?\\b|\\bST[LG]?\\b'

If cuobjdump is on PATH this script prints -res-usage for you.

Also dump the op's pre-fusion LLVM IR (v2 only) with:
    CCCL_JIT_DEBUG=./ir_v2 python diag_transform_heavy.py
"""

import argparse
import shutil
import subprocess
import sys

import cupy as cp
import numba
import numpy as np
from numba import cuda as lang

import cuda.compute

try:
    from cuda.compute._build_info import USING_V2
except Exception:  # noqa: BLE001
    USING_V2 = False


# Heavy ops copied verbatim from transform/heavy.py so this script is standalone.
def _heavy_op_32(data):
    reg = lang.local.array(shape=32, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 32):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(32):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(32):
        reg[i] = reg[32 - i - 1] * reg[i]
    out = data - data
    for i in range(32):
        out += reg[i]
    return out


def _heavy_op_64(data):
    reg = lang.local.array(shape=64, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 64):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(64):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(64):
        reg[i] = reg[64 - i - 1] * reg[i]
    out = data - data
    for i in range(64):
        out += reg[i]
    return out


def _heavy_op_128(data):
    reg = lang.local.array(shape=128, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 128):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(128):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(128):
        reg[i] = reg[128 - i - 1] * reg[i]
    out = data - data
    for i in range(128):
        out += reg[i]
    return out


def _heavy_op_256(data):
    reg = lang.local.array(shape=256, dtype=numba.uint32)
    reg[0] = data
    for i in range(1, 256):
        x = reg[i - 1]
        reg[i] = x * x + 1
    for i in range(256):
        x = reg[i]
        reg[i] = (x * x) % 19
    for i in range(256):
        reg[i] = reg[256 - i - 1] * reg[i]
    out = data - data
    for i in range(256):
        out += reg[i]
    return out


_HEAVY_OPS = {
    32: _heavy_op_32,
    64: _heavy_op_64,
    128: _heavy_op_128,
    256: _heavy_op_256,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--heaviness", type=int, default=64, choices=[32, 64, 128, 256])
    ap.add_argument("--elements", type=int, default=1 << 16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--out", default=None, help="cubin output path [default: heavy_{v1,v2}.cubin]"
    )
    args = ap.parse_args()

    tag = "v2" if USING_V2 else "v1"
    out = args.out or f"heavy_{tag}.cubin"
    n = args.elements
    op = _HEAVY_OPS[args.heaviness]

    d_in = cp.arange(n, dtype=np.uint32)
    d_out = cp.empty(n, dtype=np.uint32)

    transform = cuda.compute.make_unary_transform(d_in=d_in, d_out=d_out, op=op)

    # Dump the fused cubin (build is eager, so this is populated already).
    cubin = transform.build_result._get_cubin()
    with open(out, "wb") as f:
        f.write(cubin)
    print(f"backend: {tag} (USING_V2={USING_V2})")
    print(f"config:  heaviness={args.heaviness}  elements={n}")
    print(f"cubin:   {len(cubin)} bytes -> {out}")

    # Warm up + time.
    for _ in range(10):
        transform(d_in=d_in, d_out=d_out, op=op, num_items=n)
    cp.cuda.runtime.deviceSynchronize()

    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(args.iters):
        transform(d_in=d_in, d_out=d_out, op=op, num_items=n)
    end.record()
    end.synchronize()
    us = cp.cuda.get_elapsed_time(start, end) / args.iters * 1e3
    print(f"time:    {us:.2f} us/iter (mean of {args.iters})")

    exe = shutil.which("cuobjdump")
    if exe:
        print("\n=== cuobjdump -res-usage ===", flush=True)
        subprocess.run([exe, "-res-usage", out], check=False)
    else:
        print(
            f"\ncuobjdump not on PATH; inspect with: cuobjdump -res-usage {out}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
