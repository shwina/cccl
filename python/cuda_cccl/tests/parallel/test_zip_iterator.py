# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def test_zip_iterator_basic():
    @parallel.gpu_struct
    class Pair:
        first: np.int32
        second: np.float32

    def sum_pairs(p1, p2):
        return Pair(p1.first + p2.first, p1.second + p2.second)

    d_input1 = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_input2 = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    zip_it = parallel.ZipIterator(d_input1, d_input2)

    d_output = cp.empty(1, dtype=Pair.dtype)
    h_init = Pair(0, 0.0)

    parallel.reduce_into(zip_it, d_output, sum_pairs, len(d_input1), h_init)

    expected_first = sum(d_input1.get())  # 1+2+3+4+5 = 15
    expected_second = sum(d_input2.get())  # 1.0+2.0+3.0+4.0+5.0 = 15.0

    result = d_output.get()[0]
    assert result["first"] == expected_first
    assert result["second"] == expected_second


def test_zip_iterator_with_counting_iterator():
    """Test ZipIterator with two counting iterators."""

    @parallel.gpu_struct
    class IndexValuePair:
        index: np.int32
        value: np.int32

    def max_by_value(p1, p2):
        # Return the pair with the larger value
        return p1 if p1.value > p2.value else p2

    counting_it1 = parallel.CountingIterator(np.int32(0))  # 0, 1, 2, 3, 4
    counting_it2 = parallel.CountingIterator(np.int32(10))  # 10, 11, 12, 13, 14

    zip_it = parallel.ZipIterator(counting_it1, counting_it2)

    num_items = 5
    d_output = cp.empty(1, dtype=IndexValuePair.dtype)
    h_init = IndexValuePair(-1, -1)

    parallel.reduce_into(zip_it, d_output, max_by_value, num_items, h_init)

    result = d_output.get()[0]

    assert result["index"] == 4
    assert result["value"] == 14


def test_zip_iterator_with_counting_iterator_and_array():
    """Test ZipIterator with counting iterator and array."""
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental as parallel

    @parallel.gpu_struct
    class IndexValuePair:
        index: np.int32
        value: np.int32

    def max_by_value(p1, p2):
        # Return the pair with the larger value
        return p1 if p1.value > p2.value else p2

    counting_it = parallel.CountingIterator(np.int32(0))  # 0, 1, 2, 3, 4
    arr = cp.asarray([0, 1, 2, 4, 7, 3, 5, 6], dtype=np.int32)

    zip_it = parallel.ZipIterator(counting_it, arr)

    num_items = 8
    d_output = cp.empty(1, dtype=IndexValuePair.dtype)
    h_init = IndexValuePair(-1, -1)

    parallel.reduce_into(zip_it, d_output, max_by_value, num_items, h_init)

    result = d_output.get()[0]

    assert result["index"] == 4
    assert result["value"] == 7


def test_zip_iterator_with_counting_iterator_and_transform():
    @parallel.gpu_struct
    class IndexValuePair:
        index: np.int32
        value: np.int64

    def max_by_value(p1, p2):
        return p1 if p1.value > p2.value else p2

    counting_it = parallel.CountingIterator(np.int32(0))  # 0, 1, 2, 3, 4
    arr = cp.asarray([0, 1, 2, 4, 7, 3, 5, 6], dtype=np.int32)

    def double_op(x):
        return x * 2

    transform_it = parallel.TransformIterator(
        arr, double_op
    )  # 0, 2, 4, 8, 14, 6, 10, 12

    zip_it = parallel.ZipIterator(counting_it, transform_it)

    num_items = 8

    d_output = cp.empty(1, dtype=IndexValuePair.dtype)

    result = d_output.get()[0]
    h_init = IndexValuePair(-1, -1)

    parallel.reduce_into(zip_it, d_output, max_by_value, num_items, h_init)

    result = d_output.get()[0]

    assert result["index"] == 4
    assert result["value"] == 14


def test_zip_iterator_n_iterators():
    """Test generalized ZipIterator with N iterators (3 in this case)."""

    @parallel.gpu_struct
    class Triple:
        first: np.int32
        second: np.float32
        third: np.int64

    def sum_triples(t1, t2):
        return Triple(t1.first + t2.first, t1.second + t2.second, t1.third + t2.third)

    d_input1 = cp.array([1, 2, 3, 4], dtype=np.int32)
    d_input2 = cp.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    counting_it = parallel.CountingIterator(np.int64(10))  # 10, 11, 12, 13

    zip_it = parallel.ZipIterator(d_input1, d_input2, counting_it)

    num_items = 4
    d_output = cp.empty(1, dtype=Triple.dtype)
    h_init = Triple(0, 0.0, 0)

    parallel.reduce_into(zip_it, d_output, sum_triples, num_items, h_init)

    result = d_output.get()[0]

    # Expected results:
    # first: 1 + 2 + 3 + 4 = 10
    # second: 1.5 + 2.5 + 3.5 + 4.5 = 12.0
    # third: 10 + 11 + 12 + 13 = 46

    assert result["first"] == 10
    assert result["second"] == 12.0
    assert result["third"] == 46


def test_zip_iterator_four_iterators():
    """Test generalized ZipIterator with 4 iterators."""

    @parallel.gpu_struct
    class Quad:
        a: np.int32
        b: np.float32
        c: np.int64
        d: np.float64

    def sum_quads(q1, q2):
        return Quad(q1.a + q2.a, q1.b + q2.b, q1.c + q2.c, q1.d + q2.d)

    arr1 = cp.array([1, 2, 3], dtype=np.int32)
    arr2 = cp.array([0.1, 0.2, 0.3], dtype=np.float32)
    counting_it1 = parallel.CountingIterator(np.int64(100))  # 100, 101, 102
    counting_it2 = parallel.CountingIterator(np.float64(5.0))  # 5.0, 6.0, 7.0

    zip_it = parallel.ZipIterator(arr1, arr2, counting_it1, counting_it2)

    num_items = 3
    d_output = cp.empty(1, dtype=Quad.dtype)
    h_init = Quad(0, 0.0, 0, 0.0)

    parallel.reduce_into(zip_it, d_output, sum_quads, num_items, h_init)

    result = d_output.get()[0]

    assert result["a"] == 6
    np.testing.assert_allclose(result["b"], 0.6)
    assert result["c"] == 303
    np.testing.assert_allclose(result["d"], 18.0)
