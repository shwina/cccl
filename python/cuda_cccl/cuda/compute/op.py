# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Operator adapters for cuda.compute algorithms.

This module provides adapters that unify different operator types
(well-known operations, user callables, and pre-compiled RawOp)
into a common interface for use with cuda.compute algorithms.
"""

from typing import Callable, Hashable

from ._bindings import Op, OpKind
from ._caching import CachableFunction


def _is_well_known_op(op: OpKind) -> bool:
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


class _OpAdapter:
    """
    Provides a unified interface for operators, whether they are:
    - Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
    - Stateless user-provided callables (compiled via Numba)
    - Pre-compiled RawOp (from any LTOIR source)
    """

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        raise NotImplementedError("Subclasses must implement this method")

    def compile(self, input_types, output_type=None) -> Op:
        """
        Compile this operator to an Op for CCCL interop.

        Args:
            input_types: Tuple of numba types for input arguments
            output_type: Optional numba type for return value (inferred if None)

        Returns:
            Compiled Op object for C++ interop
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def func(self) -> Callable | None:
        """The underlying callable, if any."""
        return None


class _WellKnownOp(_OpAdapter):
    """Internal wrapper for well-known OpKind values."""

    __slots__ = ["_kind"]

    def __init__(self, kind: OpKind):
        if not _is_well_known_op(kind):
            raise ValueError(
                f"OpKind.{kind.name} is not a well-known operation. "
                "Use OpKind.PLUS, OpKind.MAXIMUM, etc."
            )
        self._kind = kind

    def get_cache_key(self) -> Hashable:
        return (self._kind.name, self._kind.value)

    def compile(self, input_types, output_type=None) -> Op:
        return Op(
            operator_type=self._kind,
            name="",
            ltoir=b"",
            state_alignment=1,
            state=b"",
        )

    @property
    def kind(self) -> OpKind:
        """The underlying OpKind."""
        return self._kind


class _StatelessOp(_OpAdapter):
    """Internal wrapper for stateless callables, compiled via Numba."""

    __slots__ = ["_func", "_cachable"]

    def __init__(self, func: Callable):
        self._func = func
        self._cachable = CachableFunction(func)

    def get_cache_key(self) -> Hashable:
        return self._cachable

    def compile(self, input_types, output_type=None) -> Op:
        from ._numba import compile_op

        # compile_op handles annotation inference internally
        raw_op = compile_op(self._func, input_types, output_type)
        return raw_op.to_cccl_op()

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func


class _RawOpAdapter(_OpAdapter):
    """Adapter for pre-compiled RawOp (LTOIR from any source)."""

    __slots__ = ["_raw_op"]

    def __init__(self, raw_op):
        from .raw import RawOp

        if not isinstance(raw_op, RawOp):
            raise TypeError(f"Expected RawOp, got {type(raw_op)}")
        self._raw_op = raw_op

    def get_cache_key(self) -> Hashable:
        return self._raw_op.get_cache_key()

    def compile(self, input_types, output_type=None) -> Op:
        # RawOp is already compiled, just convert to C binding Op
        return self._raw_op.to_cccl_op()

    @property
    def raw_op(self):
        """Access the wrapped RawOp."""
        return self._raw_op


# Public aliases
OpAdapter = _OpAdapter


def make_op_adapter(op) -> OpAdapter:
    """
    Create an OpAdapter from a callable, well-known OpKind, or RawOp.

    This function provides a unified interface for creating operator adapters
    from various sources:
    - Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
    - Python callables (compiled to LTOIR via Numba)
    - Pre-compiled RawOp (from any LTOIR source)

    Args:
        op: Callable, OpKind, or RawOp

    Returns:
        An OpAdapter instance

    Example:
        >>> from cuda.compute import OpKind
        >>> from cuda.compute.raw import RawOp
        >>>
        >>> # Well-known operation
        >>> adapter = make_op_adapter(OpKind.PLUS)
        >>>
        >>> # Python callable (uses Numba)
        >>> adapter = make_op_adapter(lambda x, y: x + y)
        >>>
        >>> # Pre-compiled RawOp
        >>> raw_op = RawOp(ltoir=..., name="my_op", ...)
        >>> adapter = make_op_adapter(raw_op)
    """
    from .raw import RawOp

    # Already an _OpAdapter instance:
    if isinstance(op, _OpAdapter):
        return op

    # Pre-compiled RawOp
    if isinstance(op, RawOp):
        return _RawOpAdapter(op)

    # Well-known operation
    if isinstance(op, OpKind):
        return _WellKnownOp(op)

    # Assume it's a callable
    return _StatelessOp(op)


__all__ = [
    "OpAdapter",
    "OpKind",
    "make_op_adapter",
]
