# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
RawOp: Pre-compiled LTOIR operation for the Raw LTOIR layer.

This module provides RawOp, which represents a pre-compiled LTOIR operation
that can be used directly with cuda.compute algorithms, bypassing Numba compilation.

Example:
    >>> from cuda.compute.raw import RawOp
    >>> from cuda.compute.raw.types import int32
    >>>
    >>> # Create from pre-compiled LTOIR (e.g., from nvcc -dlto or custom compiler)
    >>> add_op = RawOp(
    ...     ltoir=my_compiled_ltoir,
    ...     name="my_add",
    ...     arg_types=(int32, int32),
    ...     return_type=int32,
    ... )
    >>>
    >>> # Use with algorithms
    >>> cuda.compute.reduce_into(d_in, d_out, add_op, num_items, h_init)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Hashable

from .._bindings import Op, OpKind, TypeInfo

if TYPE_CHECKING:
    pass

__all__ = ["RawOp"]


class RawOp:
    """A pre-compiled LTOIR operation, ready for use with cuda.compute algorithms.

    RawOp wraps pre-compiled LTOIR bytecode along with its type signature,
    allowing users to bring their own compiler output to cuda.compute.

    The LTOIR must define a function with the following ABI:
        - extern "C" linkage
        - void return type
        - Arguments as void* pointers:
            - For N-ary operators: void(void* arg0, ..., void* argN-1, void* result)

    Example LTOIR function signatures:
        // Binary operator (e.g., for reduce)
        extern "C" void my_add(void* a, void* b, void* result);

        // Unary operator (e.g., for transform)
        extern "C" void my_transform(void* input, void* result);

    Attributes:
        ltoir: The pre-compiled LTOIR bytecode.
        name: The ABI function name in the LTOIR.
        arg_types: Tuple of TypeInfo for each input argument.
        return_type: TypeInfo for the return value.
    """

    __slots__ = ["_ltoir", "_name", "_arg_types", "_return_type"]

    def __init__(
        self,
        ltoir: bytes,
        name: str,
        arg_types: tuple[TypeInfo, ...],
        return_type: TypeInfo,
    ):
        """Create a RawOp from pre-compiled LTOIR.

        Args:
            ltoir: Pre-compiled LTOIR bytecode. Must be non-empty bytes.
            name: The ABI function name in the LTOIR (must use extern "C").
            arg_types: Tuple of TypeInfo for each input argument.
            return_type: TypeInfo for the return value.

        Raises:
            ValueError: If ltoir is not non-empty bytes.
            TypeError: If arg_types is not a tuple or return_type is not TypeInfo.
        """
        if not isinstance(ltoir, bytes) or len(ltoir) == 0:
            raise ValueError("ltoir must be non-empty bytes")
        if not isinstance(name, str) or len(name) == 0:
            raise ValueError("name must be a non-empty string")
        if not isinstance(arg_types, tuple):
            raise TypeError("arg_types must be a tuple of TypeInfo")
        if not isinstance(return_type, TypeInfo):
            raise TypeError("return_type must be a TypeInfo")

        self._ltoir = ltoir
        self._name = name
        self._arg_types = arg_types
        self._return_type = return_type

    @property
    def ltoir(self) -> bytes:
        """The pre-compiled LTOIR bytecode."""
        return self._ltoir

    @property
    def name(self) -> str:
        """The ABI function name in the LTOIR."""
        return self._name

    @property
    def arg_types(self) -> tuple[TypeInfo, ...]:
        """Tuple of TypeInfo for each input argument."""
        return self._arg_types

    @property
    def return_type(self) -> TypeInfo:
        """TypeInfo for the return value."""
        return self._return_type

    def to_cccl_op(self) -> Op:
        """Convert to low-level Op for C library interop.

        Returns:
            Op object suitable for passing to the C bindings.
        """
        return Op(
            operator_type=OpKind.STATELESS,
            name=self._name,
            ltoir=self._ltoir,
        )

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operation.

        The cache key is based on the function name and LTOIR content hash,
        allowing algorithms to cache build results for identical operations.
        """
        return (self._name, hash(self._ltoir))

    def __repr__(self) -> str:
        return (
            f"RawOp(name={self._name!r}, "
            f"arg_types={self._arg_types}, "
            f"return_type={self._return_type})"
        )
