# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import (
    CountingIterator,
    OpKind,
    TransformIterator,
    TransformOutputIterator,
    gpu_struct,
)


@pytest.fixture(params=["i4", "u4", "i8", "u8"])
def offset_dtype(request):
    return np.dtype(request.param)


def test_segmented_reduce(input_array, offset_dtype):
    "Test for all supported input types and for some offset types"

    def binary_op(a, b):
        return a + b

    assert input_array.ndim == 1
    sz = input_array.size
    rng = cp.random
    n_segments = 16
    h_offsets = cp.zeros(n_segments + 1, dtype="int64")
    h_offsets[1:] = rng.multinomial(sz, [1 / n_segments] * n_segments)

    offsets = cp.cumsum(cp.asarray(h_offsets, dtype=offset_dtype), dtype=offset_dtype)

    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]

    assert offsets.dtype == np.dtype(offset_dtype)
    assert cp.all(start_offsets <= end_offsets)
    assert end_offsets[-1] == sz

    d_in = cp.asarray(input_array)
    d_out = cp.empty(n_segments, dtype=d_in.dtype)

    h_init = np.zeros(tuple(), dtype=input_array.dtype)

    if input_array.dtype == np.float16:
        reduce_op = OpKind.PLUS
    else:
        reduce_op = binary_op

    # Call single-phase API directly with num_segments parameter
    cuda.compute.segmented_reduce(
        d_in, d_out, start_offsets, end_offsets, reduce_op, h_init, n_segments
    )

    d_expected = cp.empty_like(d_out)
    for i in range(n_segments):
        d_expected[i] = cp.sum(d_in[start_offsets[i] : end_offsets[i]])

    assert cp.all(d_out == d_expected)


def test_segmented_reduce_struct_type():
    import cupy as cp
    import numpy as np

    @gpu_struct
    class Pixel:
        r: np.int32
        g: np.int32
        b: np.int32

    def max_g_value(x, y):
        return x if x.g > y.g else y

    def align_up(n, m):
        return ((n + m - 1) // m) * m

    segment_size = 64
    n_pixels = align_up(4000, 64)
    offsets = cp.arange(n_pixels + segment_size - 1, step=segment_size, dtype=np.int64)
    start_offsets = offsets[:-1]
    end_offsets = offsets[1:]
    n_segments = start_offsets.size

    d_rgb = cp.random.randint(0, 256, (n_pixels, 3), dtype=np.int32).view(Pixel.dtype)
    d_out = cp.empty(n_segments, Pixel.dtype)

    h_init = Pixel(0, 0, 0)

    # Call single-phase API directly with n_segments parameter
    cuda.compute.segmented_reduce(
        d_rgb, d_out, start_offsets, end_offsets, max_g_value, h_init, n_segments
    )

    h_rgb = np.reshape(d_rgb.get(), (n_segments, -1))
    expected = h_rgb[np.arange(h_rgb.shape[0]), h_rgb["g"].argmax(axis=-1)]

    np.testing.assert_equal(expected["g"], d_out.get()["g"])


@pytest.mark.large
def test_large_num_segments_uniform_segment_sizes_nonuniform_input():
    """
    This test builds input iterator as transformation
    over counting iterator by a function
    k -> (F(k + 1) - F(k)) % 7

    Segmented reduction with fixed size is performed
    using add modulo 7. Expected result is known to be
    F(end_offset[k] + 1) - F(start_offset[k]) % 7
    """

    pytest.xfail("Requires host_advance for iterators when num_segments > int32")


@pytest.mark.large
def test_large_num_segments_nonuniform_segment_sizes_uniform_input():
    """
    Test with large num_segments > INT_MAX

    Input is constant iterator with value 1.

    offset positions are computed as transformation
    over counting iterator with `n -> sum(min + (k % p), k=0..n)`.
    The closed form value of the sum is coded in `offset_value`
    function.

    Result of segmented reduction is known, and is
    given by transformed iterator over counting iterator
    transformed by `k -> min + (k % p)` function.
    """

    pytest.xfail("Requires host_advance for iterators when num_segments > int32")


def test_segmented_reduce_well_known_plus():
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)

    # Create segmented data: [1, 2, 3] | [4, 5] | [6, 7, 8, 9]
    d_input = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
    d_starts = cp.array([0, 3, 5], dtype=np.int32)
    d_ends = cp.array([3, 5, 9], dtype=np.int32)
    d_output = cp.empty(3, dtype=dtype)

    cuda.compute.segmented_reduce(
        d_input, d_output, d_starts, d_ends, OpKind.PLUS, h_init, 3
    )

    expected = np.array([6, 9, 30])
    np.testing.assert_equal(d_output.get(), expected)


def test_segmented_reduce_well_known_maximum():
    dtype = np.int32
    h_init = np.array([-100], dtype=dtype)

    # Create segmented data: [1, 9, 3] | [4, 2] | [6, 7, 1, 8]
    d_input = cp.array([1, 9, 3, 4, 2, 6, 7, 1, 8], dtype=dtype)
    d_starts = cp.array([0, 3, 5], dtype=np.int32)
    d_ends = cp.array([3, 5, 9], dtype=np.int32)
    d_output = cp.empty(3, dtype=dtype)

    cuda.compute.segmented_reduce(
        d_input, d_output, d_starts, d_ends, OpKind.MAXIMUM, h_init, 3
    )

    expected = np.array([9, 4, 8])  # max of each segment
    np.testing.assert_equal(d_output.get(), expected)


def test_segmented_reduce_transform_output_iterator(floating_array):
    """Test segmented reduce with TransformOutputIterator."""
    dtype = floating_array.dtype
    h_init = np.array([0], dtype=dtype)

    # Use the floating_array fixture which provides random floating-point data of size 1000
    d_input = floating_array

    # Create 2 segments of roughly equal size
    segment_size = d_input.size // 2
    d_output = cp.empty(2, dtype=dtype)
    start_offsets = cp.array([0, segment_size], dtype=np.int32)
    end_offsets = cp.array([segment_size, d_input.size], dtype=np.int32)

    def sqrt(x: dtype) -> dtype:
        return x**0.5

    d_out_it = TransformOutputIterator(d_output, sqrt)

    cuda.compute.segmented_reduce(
        d_input, d_out_it, start_offsets, end_offsets, OpKind.PLUS, h_init, 2
    )

    expected = cp.sqrt(
        cp.array(
            [
                cp.sum(d_input[0:segment_size]),
                cp.sum(d_input[segment_size : d_input.size]),
            ]
        )
    )
    np.testing.assert_allclose(d_output.get(), expected.get(), atol=1e-6)


def test_device_segmented_reduce_for_rowwise_sum():
    def add_op(a, b):
        return a + b

    n_rows, n_cols = 67, 12345
    rng = cp.random.default_rng()
    mat = rng.integers(low=-31, high=32, dtype=np.int32, size=(n_rows, n_cols))

    def make_scaler(step):
        def scale(row_id):
            return row_id * step

        return scale

    zero = np.int32(0)
    row_offset = make_scaler(np.int32(n_cols))
    start_offsets = TransformIterator(CountingIterator(zero), row_offset)
    end_offsets = TransformIterator(CountingIterator(np.int32(1)), row_offset)

    d_input = mat
    h_init = np.zeros(tuple(), dtype=np.int32)
    d_output = cp.empty(n_rows, dtype=d_input.dtype)

    cuda.compute.segmented_reduce(
        d_input, d_output, start_offsets, end_offsets, add_op, h_init, n_rows
    )

    expected = cp.sum(mat, axis=-1)
    assert cp.all(d_output == expected)


def test_segmented_reduce_with_lambda():
    """Test segmented_reduce with a lambda function as the reducer."""
    dtype = np.int32
    h_init = np.array([0], dtype=dtype)

    # Create segmented data: [1, 2, 3] | [4, 5] | [6, 7, 8, 9]
    d_input = cp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=dtype)
    d_starts = cp.array([0, 3, 5], dtype=np.int32)
    d_ends = cp.array([3, 5, 9], dtype=np.int32)
    d_output = cp.empty(3, dtype=dtype)

    # Use a lambda function directly as the reducer
    cuda.compute.segmented_reduce(
        d_input, d_output, d_starts, d_ends, lambda a, b: a + b, h_init, 3
    )

    expected = np.array([6, 9, 30])  # sum of each segment
    np.testing.assert_equal(d_output.get(), expected)
