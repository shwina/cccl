# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from os import PathLike
from pathlib import Path

from ._binary_format import read_cclb


def load_algorithm(path: str | PathLike):
    """Load a pre-compiled algorithm from a file saved with ``.save()``.

    No NVRTC or nvJitLink is invoked; the compiled cubin is loaded directly.

    Files are stored in the CCLB binary format (magic ``b"CCLB"``).

    Args:
        path: Path to a file previously written by ``algorithm.save()``.

    Returns:
        A callable algorithm object identical to the one that was saved.

    Example:
        .. code-block:: python

            reducer = cuda.compute.make_reduce_into(d_in, d_out, op, h_init)
            reducer.save("my_reduce.alg")

            # In a fresh process:
            reducer = cuda.compute.load_algorithm("my_reduce.alg")
    """
    path = Path(path)
    algorithm, data = read_cclb(path)

    if algorithm == "reduce":
        from .algorithms._reduce import _Reduce

        return _Reduce._from_serialized(data)
    elif algorithm == "merge_sort":
        from .algorithms._sort._merge_sort import _MergeSort

        return _MergeSort._from_serialized(data)
    elif algorithm == "segmented_reduce":
        from .algorithms._segmented_reduce import _SegmentedReduce

        return _SegmentedReduce._from_serialized(data)
    elif algorithm == "unary_transform":
        from .algorithms._transform import _UnaryTransform

        return _UnaryTransform._from_serialized(data)
    elif algorithm == "binary_transform":
        from .algorithms._transform import _BinaryTransform

        return _BinaryTransform._from_serialized(data)
    elif algorithm == "radix_sort":
        from .algorithms._sort._radix_sort import _RadixSort

        return _RadixSort._from_serialized(data)
    elif algorithm == "binary_search":
        from .algorithms._binary_search import _BinarySearch

        return _BinarySearch._from_serialized(data)
    elif algorithm == "scan":
        from .algorithms._scan import _Scan

        return _Scan._from_serialized(data)
    elif algorithm == "unique_by_key":
        from .algorithms._unique_by_key import _UniqueByKey

        return _UniqueByKey._from_serialized(data)
    elif algorithm == "histogram":
        from .algorithms._histogram import _Histogram

        return _Histogram._from_serialized(data)
    elif algorithm == "three_way_partition":
        from .algorithms._three_way_partition import _ThreeWayPartition

        return _ThreeWayPartition._from_serialized(data)
    elif algorithm == "segmented_sort":
        from .algorithms._sort._segmented_sort import _SegmentedSort

        return _SegmentedSort._from_serialized(data)
    else:
        raise ValueError(f"Unknown algorithm type in file: {algorithm!r}")
