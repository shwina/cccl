# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Type helpers for the Raw LTOIR layer.

This module provides pre-built TypeInfo constants for common types,
enabling compiler-agnostic type specification without Numba dependency.

Example:
    >>> from cuda.compute.raw.types import int32, float64, storage
    >>> from cuda.compute.raw import RawOp
    >>>
    >>> # Use with RawOp
    >>> add_op = RawOp(ltoir=my_ltoir, name="add", arg_types=(int32, int32), return_type=int32)
    >>>
    >>> # Create custom struct type
    >>> my_struct = storage(size=16, alignment=8)
"""

from .._bindings import TypeEnum, TypeInfo

__all__ = [
    # Integer types
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    # Floating point types
    "float16",
    "float32",
    "float64",
    # Other types
    "bool_",
    # Factory function
    "storage",
    # Re-exports
    "TypeInfo",
    "TypeEnum",
]

# Signed integer types
int8 = TypeInfo(1, 1, TypeEnum.INT8)
int16 = TypeInfo(2, 2, TypeEnum.INT16)
int32 = TypeInfo(4, 4, TypeEnum.INT32)
int64 = TypeInfo(8, 8, TypeEnum.INT64)

# Unsigned integer types
uint8 = TypeInfo(1, 1, TypeEnum.UINT8)
uint16 = TypeInfo(2, 2, TypeEnum.UINT16)
uint32 = TypeInfo(4, 4, TypeEnum.UINT32)
uint64 = TypeInfo(8, 8, TypeEnum.UINT64)

# Floating point types
float16 = TypeInfo(2, 2, TypeEnum.FLOAT16)
float32 = TypeInfo(4, 4, TypeEnum.FLOAT32)
float64 = TypeInfo(8, 8, TypeEnum.FLOAT64)

# Boolean type
bool_ = TypeInfo(1, 1, TypeEnum.BOOLEAN)


def storage(size: int, alignment: int | None = None) -> TypeInfo:
    """Create TypeInfo for opaque storage types (custom structs).

    Use this for custom struct types where you only need to specify
    size and alignment, without semantic type information.

    Args:
        size: Size of the type in bytes.
        alignment: Alignment requirement in bytes. Defaults to size if not specified.

    Returns:
        TypeInfo with STORAGE type enum.

    Example:
        >>> # 16-byte struct with 8-byte alignment
        >>> my_struct = storage(16, 8)
    """
    return TypeInfo(
        size, alignment if alignment is not None else size, TypeEnum.STORAGE
    )
