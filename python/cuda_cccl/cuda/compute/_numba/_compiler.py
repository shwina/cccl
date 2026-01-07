# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba compiler for cuda.compute.

This module provides functions to compile Python callables and iterator methods
to the Raw LTOIR layer using Numba. This is the default compiler implementation
that can be swapped out for other LTOIR-producing compilers.
"""

from __future__ import annotations

import functools
import uuid
from typing import TYPE_CHECKING

from numba import cuda, types

from .._bindings import TypeEnum, TypeInfo
from ..raw import RawOp
from ._odr_helpers import (
    create_advance_void_ptr_wrapper,
    create_input_dereference_void_ptr_wrapper,
    create_op_void_ptr_wrapper,
    create_output_dereference_void_ptr_wrapper,
)
from ._utils import get_inferred_return_type, signature_from_annotations

if TYPE_CHECKING:
    pass

__all__ = [
    "compile_op",
    "compile_iterator_advance",
    "compile_iterator_input_dereference",
    "compile_iterator_output_dereference",
    "numba_type_to_type_info",
]

# Mapping from numba types to TypeEnum
_TYPE_TO_ENUM = {
    types.int8: TypeEnum.INT8,
    types.int16: TypeEnum.INT16,
    types.int32: TypeEnum.INT32,
    types.int64: TypeEnum.INT64,
    types.uint8: TypeEnum.UINT8,
    types.uint16: TypeEnum.UINT16,
    types.uint32: TypeEnum.UINT32,
    types.uint64: TypeEnum.UINT64,
    types.float16: TypeEnum.FLOAT16,
    types.float32: TypeEnum.FLOAT32,
    types.float64: TypeEnum.FLOAT64,
}


def _type_to_enum(numba_type: types.Type) -> TypeEnum:
    """Convert a numba type to a TypeEnum."""
    if numba_type in _TYPE_TO_ENUM:
        return _TYPE_TO_ENUM[numba_type]
    return TypeEnum.STORAGE


@functools.lru_cache(maxsize=None)
def numba_type_to_type_info(numba_type: types.Type) -> TypeInfo:
    """Convert a numba type to a TypeInfo.

    Args:
        numba_type: A Numba type.

    Returns:
        TypeInfo with size, alignment, and type enum.
    """
    context = cuda.descriptor.cuda_target.target_context
    value_type = context.get_value_type(numba_type)
    if isinstance(numba_type, types.Record):
        # then `value_type` is a pointer and we need the
        # alignment of the pointee.
        value_type = value_type.pointee
    size = value_type.get_abi_size(context.target_data)
    alignment = value_type.get_abi_alignment(context.target_data)
    return TypeInfo(size, alignment, _type_to_enum(numba_type))


def compile_op(
    func,
    arg_numba_types: tuple[types.Type, ...],
    return_numba_type: types.Type | None = None,
) -> RawOp:
    """Compile a Python callable to RawOp using Numba.

    Args:
        func: Python callable to compile.
        arg_numba_types: Tuple of Numba types for each argument.
        return_numba_type: Optional Numba type for return value.
            If None, will attempt to infer from annotations or compilation.

    Returns:
        RawOp containing the compiled LTOIR.
    """
    # Try to get signature from annotations first
    try:
        sig = signature_from_annotations(func)
    except ValueError:
        # Infer signature from input/output types
        if return_numba_type is None:
            return_numba_type = get_inferred_return_type(func, arg_numba_types)
        sig = return_numba_type(*arg_numba_types)

    wrapped, wrapper_sig = create_op_void_ptr_wrapper(func, sig)
    ltoir, _ = cuda.compile(wrapped, sig=wrapper_sig, output="ltoir")

    return RawOp(
        ltoir=ltoir,
        name=wrapped.__name__,
        arg_types=tuple(numba_type_to_type_info(t) for t in sig.args),
        return_type=numba_type_to_type_info(sig.return_type),
    )


def _get_abi_suffix() -> str:
    """Generate a unique ABI suffix for iterator methods."""
    return uuid.uuid4().hex


@functools.lru_cache(maxsize=256)
def _cached_compile(func, sig, abi_name=None, **kwargs):
    """Cached compilation of a function."""
    return cuda.compile(func, sig, abi_info={"abi_name": abi_name}, **kwargs)


def compile_iterator_advance(
    advance_fn, state_ptr_type: types.CPointer
) -> tuple[str, bytes]:
    """Compile an iterator advance method to LTOIR.

    Args:
        advance_fn: The advance method (state_ptr, distance) -> None.
        state_ptr_type: Numba CPointer type for the state.

    Returns:
        Tuple of (abi_name, ltoir_bytes).
    """
    abi_name = f"advance_{_get_abi_suffix()}"
    wrapped_advance, wrapper_sig = create_advance_void_ptr_wrapper(
        advance_fn, state_ptr_type
    )
    ltoir, _ = _cached_compile(
        wrapped_advance,
        wrapper_sig,
        output="ltoir",
        abi_name=abi_name,
    )
    return (abi_name, ltoir)


def compile_iterator_input_dereference(
    deref_fn, state_ptr_type: types.CPointer, value_type: types.Type
) -> tuple[str, bytes]:
    """Compile an iterator input dereference method to LTOIR.

    Args:
        deref_fn: The dereference method (state_ptr, result_ptr) -> None.
        state_ptr_type: Numba CPointer type for the state.
        value_type: Numba type for the dereferenced value.

    Returns:
        Tuple of (abi_name, ltoir_bytes).
    """
    abi_name = f"input_dereference_{_get_abi_suffix()}"
    wrapped_deref, wrapper_sig = create_input_dereference_void_ptr_wrapper(
        deref_fn, state_ptr_type, value_type
    )
    ltoir, _ = _cached_compile(
        wrapped_deref,
        wrapper_sig,
        output="ltoir",
        abi_name=abi_name,
    )
    return (abi_name, ltoir)


def compile_iterator_output_dereference(
    deref_fn, state_ptr_type: types.CPointer, value_type: types.Type
) -> tuple[str, bytes]:
    """Compile an iterator output dereference method to LTOIR.

    Args:
        deref_fn: The dereference method (state_ptr, value) -> None.
        state_ptr_type: Numba CPointer type for the state.
        value_type: Numba type for the value to write.

    Returns:
        Tuple of (abi_name, ltoir_bytes).
    """
    abi_name = f"output_dereference_{_get_abi_suffix()}"
    wrapped_deref, wrapper_sig = create_output_dereference_void_ptr_wrapper(
        deref_fn, state_ptr_type, value_type
    )
    ltoir, _ = _cached_compile(
        wrapped_deref,
        wrapper_sig,
        output="ltoir",
        abi_name=abi_name,
    )
    return (abi_name, ltoir)
