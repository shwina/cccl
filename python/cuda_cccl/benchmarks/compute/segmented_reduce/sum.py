# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Python benchmark for segmented_reduce (sum) with fixed-size segments.

C++ equivalent: cub/benchmarks/bench/segmented_reduce/sum.cu (uses base.cuh)

Notes:
- Implements three sub-benchmarks: small, medium, large (by SegmentSize)
- The C++ equivalent uses DispatchFixedSizeSegmentedReduce; the Python API uses
  the variable dispatch path with uniform fixed-size offsets (functionally equivalent
  for benchmarking throughput).
- OffsetT axis is omitted because the Python API does not expose offset type.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cupy as cp
import numpy as np
from utils import (
    FUNDAMENTAL_TYPES as TYPE_MAP,
)
from utils import (
    as_cupy_stream,
    generate_data_with_entropy,
    generate_fixed_segment_offsets,
)

import cuda.bench as bench
from cuda.compute import OpKind, make_segmented_reduce


def run_segmented_reduce(
    state: bench.State,
    d_in,
    d_out,
    h_init,
    start_offsets,
    end_offsets,
    num_segments,
    segment_size,
):
    reducer = make_segmented_reduce(
        d_in,
        d_out,
        start_offsets,
        end_offsets,
        OpKind.PLUS,
        h_init,
    )

    temp_storage_bytes = reducer(
        None,
        d_in,
        d_out,
        OpKind.PLUS,
        num_segments,
        start_offsets,
        end_offsets,
        h_init,
        max_segment_size=segment_size,
    )
    alloc_stream = as_cupy_stream(state.get_stream())
    with alloc_stream:
        temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)

    def launcher(launch: bench.Launch):
        reducer(
            temp_storage,
            d_in,
            d_out,
            OpKind.PLUS,
            num_segments,
            start_offsets,
            end_offsets,
            h_init,
            segment_size,
            launch.get_stream(),
        )

    state.exec(launcher, batched=False, sync=True)


def bench_segmented_reduce_fixed(state: bench.State):
    type_str = state.get_string("T{ct}")
    dtype = TYPE_MAP[type_str]
    num_elements = int(state.get_int64("Elements{io}"))
    segment_size = int(state.get_int64("SegmentSize"))

    alloc_stream = as_cupy_stream(state.get_stream())
    start_offsets, end_offsets, num_segments, num_elements = (
        generate_fixed_segment_offsets(num_elements, segment_size, alloc_stream)
    )
    h_init = np.zeros(1, dtype=dtype)
    d_in = generate_data_with_entropy(num_elements, dtype, "1.000", alloc_stream)
    with alloc_stream:
        d_out = cp.empty(num_segments, dtype=dtype)

    alloc_stream.synchronize()

    state.add_element_count(num_elements)
    state.add_global_memory_reads(num_elements * d_in.dtype.itemsize)
    state.add_global_memory_writes(num_segments * d_out.dtype.itemsize)
    state.add_global_memory_reads((num_segments + 1) * start_offsets.dtype.itemsize)

    run_segmented_reduce(
        state,
        d_in,
        d_out,
        h_init,
        start_offsets,
        end_offsets,
        num_segments,
        segment_size,
    )


if __name__ == "__main__":
    b_small = bench.register(bench_segmented_reduce_fixed)
    b_small.set_name("small")
    b_small.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_small.add_int64_power_of_two_axis("Elements{io}", range(16, 28, 4))
    b_small.add_int64_power_of_two_axis("SegmentSize", range(0, 5, 1))

    b_medium = bench.register(bench_segmented_reduce_fixed)
    b_medium.set_name("medium")
    b_medium.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_medium.add_int64_power_of_two_axis("Elements{io}", range(16, 28, 4))
    b_medium.add_int64_power_of_two_axis("SegmentSize", range(5, 9, 1))

    b_large = bench.register(bench_segmented_reduce_fixed)
    b_large.set_name("large")
    b_large.add_string_axis("T{ct}", list(TYPE_MAP.keys()))
    b_large.add_int64_power_of_two_axis("Elements{io}", range(16, 28, 4))
    b_large.add_int64_power_of_two_axis("SegmentSize", range(9, 17, 2))

    bench.run_all_benchmarks(sys.argv)
