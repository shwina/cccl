# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Raw LTOIR Layer for cuda.compute.

This module provides the Raw LTOIR layer, which sits directly above the C bindings
and enables compiler-agnostic LTOIR injection. Users can provide pre-compiled LTOIR
from any source (Numba, nvcc, MLIR, Triton, custom compilers) to use with
cuda.compute algorithms.

The main classes are:
    - RawOp: Pre-compiled LTOIR operation (for reduce, transform, etc.)
    - RawIterator: Pre-compiled LTOIR iterator (advance + dereference)

Type helpers are provided in the `types` submodule for specifying type signatures
without depending on Numba.

Example:
    >>> from cuda.compute.raw import RawOp, RawIterator
    >>> from cuda.compute.raw.types import int32, float64
    >>>
    >>> # Create operation from LTOIR compiled elsewhere
    >>> add_op = RawOp(
    ...     ltoir=my_add_ltoir,
    ...     name="my_add",
    ...     arg_types=(int32, int32),
    ...     return_type=int32,
    ... )
    >>>
    >>> # Use with cuda.compute algorithms
    >>> import cuda.compute as cc
    >>> cc.reduce_into(d_in, d_out, add_op, num_items, h_init)

LTOIR ABI Contract:
    All LTOIR functions must use extern "C" linkage with void* parameters:

    Binary operator:
        extern "C" void op_name(void* arg0, void* arg1, void* result);

    Unary operator:
        extern "C" void op_name(void* arg0, void* result);

    Iterator advance:
        extern "C" void advance_name(void* state_ptr, uint64_t offset);

    Iterator input dereference:
        extern "C" void deref_name(void* state_ptr, void* result_ptr);

    Iterator output dereference:
        extern "C" void deref_name(void* state_ptr, void* value_ptr);
"""

from ._raw_iterator import RawIterator
from ._raw_op import RawOp
from .types import (
    TypeEnum,
    TypeInfo,
    bool_,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    storage,
    uint8,
    uint16,
    uint32,
    uint64,
)

__all__ = [
    # Core classes
    "RawOp",
    "RawIterator",
    # Type helpers
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool_",
    "storage",
    # Re-exports from bindings
    "TypeInfo",
    "TypeEnum",
]
