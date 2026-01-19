# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Internal type descriptors for cuda.compute.

This module provides TypeDescriptor, a lightweight representation of types
used throughout the library. TypeDescriptor maps directly to the underlying
CCCL TypeInfo structure with size, alignment, and type enumeration.

This is an internal module - TypeDescriptor is used throughout the library
but is not part of the public API.
"""

from __future__ import annotations

import functools

import numpy as np

from ._bindings import TypeEnum, TypeInfo

# Map TypeDescriptor name -> numpy dtype for structured dtypes
_STRUCT_DTYPE_BY_NAME: dict[str, np.dtype] = {}


class TypeDescriptor:
    """
    A type descriptor that wraps size, alignment, and type enumeration.

    This is used to specify types for operations and iterators without
    requiring Numba's type system. It maps directly to CCCL's TypeInfo.

    Attributes:
        size: Size of the type in bytes
        alignment: Alignment of the type in bytes
        type_enum: The TypeEnum classification (INT32, FLOAT64, STORAGE, etc.)
        name: Human-readable name for debugging
    """

    __slots__ = ("size", "alignment", "type_enum", "name")

    def __init__(self, size: int, alignment: int, type_enum: TypeEnum, name: str):
        self.size = size
        self.alignment = alignment
        self.type_enum = type_enum
        self.name = name

    def to_type_info(self) -> TypeInfo:
        """Convert this type descriptor to a CCCL TypeInfo object."""
        return TypeInfo(self.size, self.alignment, self.type_enum)

    def __repr__(self) -> str:
        return f"TypeDescriptor({self.name})"

    def __eq__(self, other) -> bool:
        if isinstance(other, TypeDescriptor):
            return (
                self.size == other.size
                and self.alignment == other.alignment
                and self.type_enum == other.type_enum
            )
        return False

    def __hash__(self) -> int:
        return hash((self.size, self.alignment, self.type_enum))


# =============================================================================
# Standard type descriptors
# =============================================================================

# Signed integer types
int8 = TypeDescriptor(1, 1, TypeEnum.INT8, "int8")
int16 = TypeDescriptor(2, 2, TypeEnum.INT16, "int16")
int32 = TypeDescriptor(4, 4, TypeEnum.INT32, "int32")
int64 = TypeDescriptor(8, 8, TypeEnum.INT64, "int64")

# Unsigned integer types
uint8 = TypeDescriptor(1, 1, TypeEnum.UINT8, "uint8")
uint16 = TypeDescriptor(2, 2, TypeEnum.UINT16, "uint16")
uint32 = TypeDescriptor(4, 4, TypeEnum.UINT32, "uint32")
uint64 = TypeDescriptor(8, 8, TypeEnum.UINT64, "uint64")

# Floating point types
float16 = TypeDescriptor(2, 2, TypeEnum.FLOAT16, "float16")
float32 = TypeDescriptor(4, 4, TypeEnum.FLOAT32, "float32")
float64 = TypeDescriptor(8, 8, TypeEnum.FLOAT64, "float64")

# Boolean
boolean = TypeDescriptor(1, 1, TypeEnum.BOOLEAN, "bool")

# Mapping from numpy dtype to TypeEnum
_NUMPY_DTYPE_TO_ENUM = {
    np.dtype("int8"): TypeEnum.INT8,
    np.dtype("int16"): TypeEnum.INT16,
    np.dtype("int32"): TypeEnum.INT32,
    np.dtype("int64"): TypeEnum.INT64,
    np.dtype("uint8"): TypeEnum.UINT8,
    np.dtype("uint16"): TypeEnum.UINT16,
    np.dtype("uint32"): TypeEnum.UINT32,
    np.dtype("uint64"): TypeEnum.UINT64,
    np.dtype("float16"): TypeEnum.FLOAT16,
    np.dtype("float32"): TypeEnum.FLOAT32,
    np.dtype("float64"): TypeEnum.FLOAT64,
    np.dtype("bool"): TypeEnum.BOOLEAN,
}

# Mapping from numpy dtype to pre-defined TypeDescriptor
_NUMPY_DTYPE_TO_DESCRIPTOR = {
    np.dtype("int8"): int8,
    np.dtype("int16"): int16,
    np.dtype("int32"): int32,
    np.dtype("int64"): int64,
    np.dtype("uint8"): uint8,
    np.dtype("uint16"): uint16,
    np.dtype("uint32"): uint32,
    np.dtype("uint64"): uint64,
    np.dtype("float16"): float16,
    np.dtype("float32"): float32,
    np.dtype("float64"): float64,
    np.dtype("bool"): boolean,
}


@functools.lru_cache(maxsize=256)
def from_numpy_dtype(dtype: np.dtype) -> TypeDescriptor:
    """
    Create a TypeDescriptor from a numpy dtype.

    This handles both scalar dtypes (int32, float64, etc.) and structured
    dtypes (record types). For structured dtypes, a custom type with the
    appropriate size and alignment is returned.

    Args:
        dtype: A numpy dtype

    Returns:
        A TypeDescriptor for the dtype

    Example:
        from cuda.compute._types import from_numpy_dtype
        import numpy as np

        int_type = from_numpy_dtype(np.dtype('int32'))
        struct_type = from_numpy_dtype(np.dtype([('x', 'i4'), ('y', 'f8')]))
    """
    dtype = np.dtype(dtype)  # Ensure it's a dtype object

    # Check for pre-defined types first
    if dtype in _NUMPY_DTYPE_TO_DESCRIPTOR:
        return _NUMPY_DTYPE_TO_DESCRIPTOR[dtype]

    # Handle complex types (kind 'c' = complex floating)
    if dtype.kind == "c":
        return TypeDescriptor(
            dtype.itemsize, dtype.alignment, TypeEnum.STORAGE, dtype.name
        )

    # For structured/record types, use STORAGE enum
    if dtype.fields is not None:
        type_desc = TypeDescriptor(
            dtype.itemsize, dtype.alignment, TypeEnum.STORAGE, f"struct({dtype})"
        )
        _STRUCT_DTYPE_BY_NAME[type_desc.name] = dtype
        return type_desc

    # Fallback for any other type
    type_enum = _NUMPY_DTYPE_TO_ENUM.get(dtype, TypeEnum.STORAGE)
    return TypeDescriptor(dtype.itemsize, dtype.alignment, type_enum, dtype.name)


def custom_type(size: int, alignment: int, name: str = "custom") -> TypeDescriptor:
    """
    Create a custom TypeDescriptor for user-defined struct types.

    This is useful for pre-compiled operations that work with custom
    struct types where the exact layout is known.

    Args:
        size: Size of the type in bytes
        alignment: Alignment of the type in bytes
        name: Optional name for debugging

    Returns:
        A TypeDescriptor for the custom type

    Example:
        # A custom struct with 16 bytes and 8-byte alignment
        my_struct_type = custom_type(16, 8, "MyStruct")
    """
    return TypeDescriptor(size, alignment, TypeEnum.STORAGE, name)


def dtype_from_type_descriptor(type_desc: TypeDescriptor) -> np.dtype | None:
    """Return the numpy dtype for a structured TypeDescriptor, if known."""
    return _STRUCT_DTYPE_BY_NAME.get(type_desc.name)


def pointer_type() -> TypeDescriptor:
    """
    Return a TypeDescriptor for a device pointer.

    Device pointers are 64-bit (8 bytes) with 8-byte alignment.
    """
    return uint64  # Pointers are uint64 on device
