# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for the Raw LTOIR layer.

These tests verify that RawOp and RawIterator work correctly with cuda.compute
algorithms, enabling the "bring your own compiler" use case.
"""

import numpy as np
import pytest

# Test imports work correctly


def test_raw_layer_imports():
    """Test that all Raw layer exports are accessible."""
    from cuda.compute import (
        TypeInfo,
        float64,
        int32,
        uint8,
    )

    # Verify type helpers are TypeInfo instances
    assert isinstance(int32, TypeInfo)
    assert isinstance(float64, TypeInfo)
    assert isinstance(uint8, TypeInfo)


def test_type_info_properties():
    """Test TypeInfo constants have correct properties."""
    from cuda.compute import float32, float64, int32, int64, uint8

    # int32: 4 bytes, 4-byte aligned
    assert int32.size == 4
    assert int32.alignment == 4

    # int64: 8 bytes, 8-byte aligned
    assert int64.size == 8
    assert int64.alignment == 8

    # float32: 4 bytes, 4-byte aligned
    assert float32.size == 4
    assert float32.alignment == 4

    # float64: 8 bytes, 8-byte aligned
    assert float64.size == 8
    assert float64.alignment == 8

    # uint8: 1 byte, 1-byte aligned
    assert uint8.size == 1
    assert uint8.alignment == 1


def test_storage_factory():
    """Test the storage() factory function."""
    from cuda.compute import TypeEnum, storage

    # Default alignment = size
    s1 = storage(16)
    assert s1.size == 16
    assert s1.alignment == 16
    assert s1.typenum == TypeEnum.STORAGE

    # Explicit alignment
    s2 = storage(16, 8)
    assert s2.size == 16
    assert s2.alignment == 8


def test_raw_op_validation():
    """Test RawOp constructor validation."""
    from cuda.compute import RawOp, int32

    # Valid construction
    ltoir = b"\x00" * 100  # Dummy LTOIR
    op = RawOp(ltoir=ltoir, name="test_op", arg_types=(int32, int32), return_type=int32)
    assert op.name == "test_op"
    assert op.ltoir == ltoir
    assert len(op.arg_types) == 2

    # Invalid: empty ltoir
    with pytest.raises(ValueError, match="non-empty bytes"):
        RawOp(ltoir=b"", name="test", arg_types=(int32,), return_type=int32)

    # Invalid: ltoir not bytes
    with pytest.raises(ValueError, match="non-empty bytes"):
        RawOp(ltoir="not bytes", name="test", arg_types=(int32,), return_type=int32)

    # Invalid: empty name
    with pytest.raises(ValueError, match="non-empty string"):
        RawOp(ltoir=ltoir, name="", arg_types=(int32,), return_type=int32)

    # Invalid: arg_types not tuple
    with pytest.raises(TypeError, match="tuple"):
        RawOp(ltoir=ltoir, name="test", arg_types=[int32], return_type=int32)


def test_raw_op_cache_key():
    """Test RawOp cache key generation."""
    from cuda.compute import RawOp, int32

    ltoir1 = b"\x00" * 100
    ltoir2 = b"\x01" * 100

    op1 = RawOp(ltoir=ltoir1, name="add", arg_types=(int32, int32), return_type=int32)
    op2 = RawOp(ltoir=ltoir1, name="add", arg_types=(int32, int32), return_type=int32)
    op3 = RawOp(ltoir=ltoir2, name="add", arg_types=(int32, int32), return_type=int32)

    # Same LTOIR and name should have same cache key
    assert op1.get_cache_key() == op2.get_cache_key()

    # Different LTOIR should have different cache key
    assert op1.get_cache_key() != op3.get_cache_key()


def test_raw_op_to_cccl_op():
    """Test RawOp conversion to low-level Op."""
    from cuda.compute import RawOp, int32
    from cuda.compute._bindings import Op

    ltoir = b"\x00" * 100
    raw_op = RawOp(
        ltoir=ltoir, name="my_add", arg_types=(int32, int32), return_type=int32
    )

    cccl_op = raw_op.to_cccl_op()
    assert isinstance(cccl_op, Op)
    assert cccl_op.name == "my_add"
    assert cccl_op.ltoir == ltoir


def test_raw_iterator_validation():
    """Test RawIterator constructor validation."""
    from cuda.compute import RawIterator, int32

    state = b"\x00" * 8
    advance = ("advance_fn", b"\x00" * 100)
    input_deref = ("input_deref_fn", b"\x00" * 100)

    # Valid: input iterator
    it = RawIterator(
        state=state,
        state_alignment=8,
        value_type=int32,
        advance=advance,
        input_dereference=input_deref,
    )
    assert it.is_input_iterator
    assert not it.is_output_iterator

    # Valid: output iterator
    output_deref = ("output_deref_fn", b"\x00" * 100)
    it2 = RawIterator(
        state=state,
        state_alignment=8,
        value_type=int32,
        advance=advance,
        output_dereference=output_deref,
    )
    assert not it2.is_input_iterator
    assert it2.is_output_iterator

    # Valid: bidirectional iterator
    it3 = RawIterator(
        state=state,
        state_alignment=8,
        value_type=int32,
        advance=advance,
        input_dereference=input_deref,
        output_dereference=output_deref,
    )
    assert it3.is_input_iterator
    assert it3.is_output_iterator

    # Invalid: no dereference provided
    with pytest.raises(ValueError, match="At least one"):
        RawIterator(
            state=state,
            state_alignment=8,
            value_type=int32,
            advance=advance,
        )

    # Invalid: advance tuple format
    with pytest.raises(ValueError, match="advance"):
        RawIterator(
            state=state,
            state_alignment=8,
            value_type=int32,
            advance=("only_name",),  # Missing ltoir
            input_dereference=input_deref,
        )


def test_raw_iterator_ltoir_access():
    """Test RawIterator LTOIR accessor methods."""
    from cuda.compute import RawIterator, int32

    state = b"\x00" * 8
    advance = ("advance_fn", b"advance_ltoir")
    input_deref = ("input_deref_fn", b"input_deref_ltoir")
    output_deref = ("output_deref_fn", b"output_deref_ltoir")

    it = RawIterator(
        state=state,
        state_alignment=8,
        value_type=int32,
        advance=advance,
        input_dereference=input_deref,
        output_dereference=output_deref,
    )

    # Test accessor methods
    assert it.get_advance_ltoir() == advance
    assert it.get_input_dereference_ltoir() == input_deref
    assert it.get_output_dereference_ltoir() == output_deref

    # Test input-only iterator raises for output access
    input_only = RawIterator(
        state=state,
        state_alignment=8,
        value_type=int32,
        advance=advance,
        input_dereference=input_deref,
    )
    with pytest.raises(AttributeError, match="not an output iterator"):
        input_only.get_output_dereference_ltoir()


def test_make_op_adapter_with_raw_op():
    """Test that make_op_adapter correctly handles RawOp."""
    from cuda.compute import RawOp, int32
    from cuda.compute.op import _RawOpAdapter, make_op_adapter

    ltoir = b"\x00" * 100
    raw_op = RawOp(
        ltoir=ltoir, name="my_add", arg_types=(int32, int32), return_type=int32
    )

    adapter = make_op_adapter(raw_op)
    assert isinstance(adapter, _RawOpAdapter)
    assert adapter.raw_op is raw_op

    # Compile should return the underlying Op
    cccl_op = adapter.compile((int32, int32))
    assert cccl_op.name == "my_add"


# GPU-dependent tests require a GPU
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_raw_op_from_numba_compilation(dtype):
    """Test creating RawOp from Numba-compiled LTOIR and using it with reduce."""
    numba = pytest.importorskip("numba")
    cuda_module = pytest.importorskip("numba.cuda")

    # Skip if no GPU available
    if not cuda_module.is_available():
        pytest.skip("CUDA not available")

    import cuda.compute as cc
    from cuda.compute._numba import compile_op

    # Define a simple add function
    def add(x, y):
        return x + y

    # Get the numba type for the dtype
    numba_type = numba.from_dtype(np.dtype(dtype))

    # Compile to RawOp
    raw_op = compile_op(add, (numba_type, numba_type), numba_type)

    assert isinstance(raw_op, cc.RawOp)
    assert len(raw_op.ltoir) > 0
    assert "wrapped_add" in raw_op.name

    # Test with reduce_into
    d_in = cuda_module.to_device(np.array([1, 2, 3, 4, 5], dtype=dtype))
    d_out = cuda_module.to_device(np.array([0], dtype=dtype))
    h_init = np.array([0], dtype=dtype)

    cc.reduce_into(d_in, d_out, raw_op, len(d_in), h_init)

    result = d_out.copy_to_host()[0]
    expected = np.sum(np.array([1, 2, 3, 4, 5], dtype=dtype))
    assert result == expected


def test_numba_layer_imports():
    """Test that _numba layer is correctly structured."""
    from cuda.compute._numba import (
        compile_iterator_advance,
        compile_op,
        numba_type_to_type_info,
    )

    # All functions should be callable
    assert callable(compile_op)
    assert callable(compile_iterator_advance)
    assert callable(numba_type_to_type_info)


def test_backward_compatible_imports():
    """Test that old import paths still work for backward compatibility."""
    # These should still work after the refactor
    from cuda.compute._odr_helpers import (
        create_op_void_ptr_wrapper,
    )
    from cuda.compute.numba_utils import (
        get_inferred_return_type,
    )

    assert callable(create_op_void_ptr_wrapper)
    assert callable(get_inferred_return_type)
