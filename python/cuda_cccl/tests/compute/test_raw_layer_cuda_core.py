# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Integration tests for the Raw LTOIR layer using cuda.core for compilation.

These tests demonstrate the "bring your own compiler" workflow where:
1. C++ source code is compiled to LTOIR using cuda.core.Program
2. The LTOIR is wrapped in RawOp or RawIterator
3. These are used with cuda.compute algorithms

See: https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Program.html
"""

import numpy as np
import pytest


def get_current_device_arch():
    """Get the current device's compute capability as a string like 'sm_80'."""
    try:
        from cuda.core import Device
    except ImportError:
        from cuda.core.experimental import Device

    cc = Device().compute_capability
    return f"sm_{cc[0]}{cc[1]}"


# ============================================================================
# Test: RawOp with cuda.core compiled C++ binary operator
# ============================================================================


@pytest.fixture
def cuda_core_program():
    """Skip test if cuda.core is not available."""
    try:
        from cuda.core import Program
    except ImportError:
        try:
            from cuda.core.experimental import Program
        except ImportError:
            pytest.skip("cuda.core.Program not available")
    return Program


def compile_cpp_to_ltoir(program_cls, source: str, arch: str) -> bytes:
    """Compile C++ source to LTOIR using cuda.core.Program.

    Args:
        program_cls: The Program class from cuda.core
        source: C++ source code
        arch: Target architecture (e.g., 'sm_80')

    Returns:
        LTOIR bytecode
    """
    try:
        from cuda.core import ProgramOptions
    except ImportError:
        from cuda.core.experimental import ProgramOptions

    options = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,  # Required for LTO/LTOIR
        link_time_optimization=True,
    )

    program = program_cls(source, "c++", options)
    obj_code = program.compile("ltoir")

    # Get the LTOIR bytes from ObjectCode via .code attribute
    return obj_code.code


@pytest.mark.skipif(
    not pytest.importorskip("numba.cuda").is_available(),
    reason="CUDA not available",
)
def test_raw_op_with_cuda_core_binary_add(cuda_core_program):
    """Test RawOp with a C++ binary add function compiled via cuda.core."""
    from numba import cuda as numba_cuda

    import cuda.compute as cc

    arch = get_current_device_arch()

    # C++ source for a binary add operator
    # Must match the LTOIR ABI: extern "C" void name(void* a, void* b, void* result)
    cpp_source = """
    extern "C" __device__ void my_add(void* a_ptr, void* b_ptr, void* result_ptr) {
        int a = *reinterpret_cast<int*>(a_ptr);
        int b = *reinterpret_cast<int*>(b_ptr);
        *reinterpret_cast<int*>(result_ptr) = a + b;
    }
    """

    # Compile to LTOIR using cuda.core
    ltoir = compile_cpp_to_ltoir(cuda_core_program, cpp_source, arch)
    assert len(ltoir) > 0, "LTOIR compilation failed"

    # Create RawOp from the LTOIR
    add_op = cc.RawOp(
        ltoir=ltoir,
        name="my_add",
        arg_types=(cc.int32, cc.int32),
        return_type=cc.int32,
    )

    # Test with reduce_into
    d_in = numba_cuda.to_device(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    d_out = numba_cuda.to_device(np.array([0], dtype=np.int32))
    h_init = np.array([0], dtype=np.int32)

    cc.reduce_into(d_in, d_out, add_op, len(d_in), h_init)

    result = d_out.copy_to_host()[0]
    expected = 1 + 2 + 3 + 4 + 5
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.skipif(
    not pytest.importorskip("numba.cuda").is_available(),
    reason="CUDA not available",
)
def test_raw_op_with_cuda_core_max(cuda_core_program):
    """Test RawOp with a C++ max function compiled via cuda.core."""
    from numba import cuda as numba_cuda

    import cuda.compute as cc

    arch = get_current_device_arch()

    # C++ source for a binary max operator
    cpp_source = """
    extern "C" __device__ void my_max(void* a_ptr, void* b_ptr, void* result_ptr) {
        float a = *reinterpret_cast<float*>(a_ptr);
        float b = *reinterpret_cast<float*>(b_ptr);
        *reinterpret_cast<float*>(result_ptr) = (a > b) ? a : b;
    }
    """

    # Compile to LTOIR using cuda.core
    ltoir = compile_cpp_to_ltoir(cuda_core_program, cpp_source, arch)

    # Create RawOp from the LTOIR
    max_op = cc.RawOp(
        ltoir=ltoir,
        name="my_max",
        arg_types=(cc.float32, cc.float32),
        return_type=cc.float32,
    )

    # Test with reduce_into
    input_data = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float32)
    d_in = numba_cuda.to_device(input_data)
    d_out = numba_cuda.to_device(np.array([0.0], dtype=np.float32))
    h_init = np.array([float("-inf")], dtype=np.float32)

    cc.reduce_into(d_in, d_out, max_op, len(d_in), h_init)

    result = d_out.copy_to_host()[0]
    expected = 9.0
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.skipif(
    not pytest.importorskip("numba.cuda").is_available(),
    reason="CUDA not available",
)
def test_raw_op_with_cuda_core_transform(cuda_core_program):
    """Test RawOp with a C++ unary transform function."""
    from numba import cuda as numba_cuda

    import cuda.compute as cc

    arch = get_current_device_arch()

    # C++ source for a unary square function
    cpp_source = """
    extern "C" __device__ void my_square(void* input_ptr, void* result_ptr) {
        double x = *reinterpret_cast<double*>(input_ptr);
        *reinterpret_cast<double*>(result_ptr) = x * x;
    }
    """

    # Compile to LTOIR using cuda.core
    ltoir = compile_cpp_to_ltoir(cuda_core_program, cpp_source, arch)

    # Create RawOp from the LTOIR
    square_op = cc.RawOp(
        ltoir=ltoir,
        name="my_square",
        arg_types=(cc.float64,),
        return_type=cc.float64,
    )

    # Test with unary_transform
    input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    d_in = numba_cuda.to_device(input_data)
    d_out = numba_cuda.to_device(np.zeros(5, dtype=np.float64))

    cc.unary_transform(d_in, d_out, square_op, len(d_in))

    result = d_out.copy_to_host()
    expected = input_data**2
    np.testing.assert_array_almost_equal(result, expected)


# ============================================================================
# Test: RawIterator with cuda.core compiled C++ advance/dereference
# ============================================================================


@pytest.mark.skipif(
    not pytest.importorskip("numba.cuda").is_available(),
    reason="CUDA not available",
)
def test_raw_iterator_counting_with_cuda_core(cuda_core_program):
    """Test RawIterator implementing a counting iterator via cuda.core C++."""
    from numba import cuda as numba_cuda

    import cuda.compute as cc

    arch = get_current_device_arch()

    # C++ source for counting iterator advance
    # State is a single int64 value (the current count)
    # Note: offset is passed as void* pointer to uint64, must be dereferenced
    advance_source = """
    extern "C" __device__ void counting_advance(void* state_ptr, void* offset_ptr) {
        // Load offset from pointer (it's a uint64)
        unsigned long long offset = *reinterpret_cast<unsigned long long*>(offset_ptr);
        // State is the current count value directly, add offset to advance
        *reinterpret_cast<long long*>(state_ptr) += static_cast<long long>(offset);
    }
    """

    # C++ source for counting iterator dereference
    deref_source = """
    extern "C" __device__ void counting_deref(void* state_ptr, void* result_ptr) {
        // State is the current count value directly
        long long value = *reinterpret_cast<long long*>(state_ptr);
        *reinterpret_cast<long long*>(result_ptr) = value;
    }
    """

    # Compile to LTOIR
    advance_ltoir = compile_cpp_to_ltoir(cuda_core_program, advance_source, arch)
    deref_ltoir = compile_cpp_to_ltoir(cuda_core_program, deref_source, arch)

    # Create state: starting count value
    start_value = np.int64(10)
    state = start_value.tobytes()

    # Create RawIterator
    counting_iter = cc.RawIterator(
        state=state,
        state_alignment=8,
        value_type=cc.int64,
        advance=("counting_advance", advance_ltoir),
        input_dereference=("counting_deref", deref_ltoir),
    )

    # Test with reduce_into to sum counting values
    # This should sum: 10 + 11 + 12 + 13 + 14 = 60
    d_out = numba_cuda.to_device(np.array([0], dtype=np.int64))
    h_init = np.array([0], dtype=np.int64)

    cc.reduce_into(counting_iter, d_out, cc.OpKind.PLUS, 5, h_init)

    result = d_out.copy_to_host()[0]
    expected = 10 + 11 + 12 + 13 + 14
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.skipif(
    not pytest.importorskip("numba.cuda").is_available(),
    reason="CUDA not available",
)
def test_raw_iterator_constant_with_cuda_core(cuda_core_program):
    """Test RawIterator implementing a constant iterator via cuda.core C++."""
    from numba import cuda as numba_cuda

    import cuda.compute as cc

    arch = get_current_device_arch()

    # Constant iterator: advance does nothing, deref returns the constant
    # Note: offset is passed as void* pointer to uint64
    advance_source = """
    extern "C" __device__ void constant_advance(void* state_ptr, void* offset_ptr) {
        // Do nothing - constant iterator doesn't advance
        (void)state_ptr;
        (void)offset_ptr;
    }
    """

    deref_source = """
    extern "C" __device__ void constant_deref(void* state_ptr, void* result_ptr) {
        // State holds the constant value
        float value = *reinterpret_cast<float*>(state_ptr);
        *reinterpret_cast<float*>(result_ptr) = value;
    }
    """

    # Compile to LTOIR
    advance_ltoir = compile_cpp_to_ltoir(cuda_core_program, advance_source, arch)
    deref_ltoir = compile_cpp_to_ltoir(cuda_core_program, deref_source, arch)

    # State: the constant value (e.g., 7.5)
    constant_value = np.float32(7.5)
    state = constant_value.tobytes()

    # Create RawIterator
    constant_iter = cc.RawIterator(
        state=state,
        state_alignment=4,
        value_type=cc.float32,
        advance=("constant_advance", advance_ltoir),
        input_dereference=("constant_deref", deref_ltoir),
    )

    # Sum 5 constant values: 7.5 * 5 = 37.5
    d_out = numba_cuda.to_device(np.array([0.0], dtype=np.float32))
    h_init = np.array([0.0], dtype=np.float32)

    cc.reduce_into(constant_iter, d_out, cc.OpKind.PLUS, 5, h_init)

    result = d_out.copy_to_host()[0]
    expected = 7.5 * 5
    assert abs(result - expected) < 1e-5, f"Expected {expected}, got {result}"


# ============================================================================
# Test: Combined RawOp and RawIterator
# ============================================================================


@pytest.mark.skipif(
    not pytest.importorskip("numba.cuda").is_available(),
    reason="CUDA not available",
)
def test_raw_op_and_iterator_combined(cuda_core_program):
    """Test using both RawOp and RawIterator together in a reduction."""
    from numba import cuda as numba_cuda

    import cuda.compute as cc

    arch = get_current_device_arch()

    # Custom multiply operator
    multiply_source = """
    extern "C" __device__ void my_multiply(void* a_ptr, void* b_ptr, void* result_ptr) {
        int a = *reinterpret_cast<int*>(a_ptr);
        int b = *reinterpret_cast<int*>(b_ptr);
        *reinterpret_cast<int*>(result_ptr) = a * b;
    }
    """

    # Counting iterator (values 1, 2, 3, 4, 5)
    # Note: offset is passed as void* pointer to uint64
    advance_source = """
    extern "C" __device__ void count_advance(void* state_ptr, void* offset_ptr) {
        unsigned long long offset = *reinterpret_cast<unsigned long long*>(offset_ptr);
        *reinterpret_cast<int*>(state_ptr) += static_cast<int>(offset);
    }
    """

    deref_source = """
    extern "C" __device__ void count_deref(void* state_ptr, void* result_ptr) {
        int value = *reinterpret_cast<int*>(state_ptr);
        *reinterpret_cast<int*>(result_ptr) = value;
    }
    """

    # Compile all sources
    multiply_ltoir = compile_cpp_to_ltoir(cuda_core_program, multiply_source, arch)
    advance_ltoir = compile_cpp_to_ltoir(cuda_core_program, advance_source, arch)
    deref_ltoir = compile_cpp_to_ltoir(cuda_core_program, deref_source, arch)

    # Create RawOp for multiply
    multiply_op = cc.RawOp(
        ltoir=multiply_ltoir,
        name="my_multiply",
        arg_types=(cc.int32, cc.int32),
        return_type=cc.int32,
    )

    # Create RawIterator starting at 1
    state = np.int32(1).tobytes()
    counting_iter = cc.RawIterator(
        state=state,
        state_alignment=4,
        value_type=cc.int32,
        advance=("count_advance", advance_ltoir),
        input_dereference=("count_deref", deref_ltoir),
    )

    # Compute factorial: 1 * 2 * 3 * 4 * 5 = 120
    d_out = numba_cuda.to_device(np.array([0], dtype=np.int32))
    h_init = np.array([1], dtype=np.int32)  # Identity for multiplication

    cc.reduce_into(counting_iter, d_out, multiply_op, 5, h_init)

    result = d_out.copy_to_host()[0]
    expected = 1 * 2 * 3 * 4 * 5
    assert result == expected, f"Expected {expected}, got {result}"
