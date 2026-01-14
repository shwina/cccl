# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Operator adapters for cuda.compute algorithms.

This module provides the unified interface for operators used in algorithms like
reduce_into, scan, etc. Operators can be:

- Well-known operations (OpKind.PLUS, OpKind.MAXIMUM, etc.)
- User-provided Python callables (compiled to LTOIR at runtime)
- Pre-compiled LTOIR (CompiledOp)
"""

from __future__ import annotations

from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Callable,
    Hashable,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ._bindings import Op, OpKind
from ._caching import CachableFunction

if TYPE_CHECKING:
    from ._types import TypeDescriptor


class OpContext(Enum):
    """
    Internal: Context in which an operator is compiled.

    Different contexts require different void* wrapper signatures when
    compiling callables to LTOIR. This is determined internally by algorithms
    and iterators - users should not need to specify this.
    """

    BINARY_OP = auto()  # void(a*, b*, result*)
    UNARY_OP = auto()  # void(a*, result*)
    ADVANCE = auto()  # void(state*, offset*)
    INPUT_DEREF = auto()  # void(state*, result*)
    OUTPUT_DEREF = auto()  # void(state*, value*)


@runtime_checkable
class OpProtocol(Protocol):
    """
    Protocol defining the interface for operator adapters.

    All operator implementations (well-known, JIT-compiled, pre-compiled)
    must implement this protocol to be usable with cuda.compute algorithms.
    """

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        ...

    def compile(
        self,
        input_types: tuple[TypeDescriptor, ...],
        output_type: TypeDescriptor | None = None,
        context: OpContext = OpContext.BINARY_OP,
    ) -> Op:
        """
        Compile this operator to an Op for CCCL interop.

        Args:
            input_types: Tuple of TypeDescriptors for input arguments
            output_type: Optional TypeDescriptor for return value (inferred if None)
            context: The context in which the op is being compiled (affects wrapper signature)

        Returns:
            Compiled Op object for C++ interop
        """
        ...

    @property
    def func(self) -> Callable | None:
        """The underlying callable, if any."""
        ...


def _is_well_known_op(op: OpKind) -> bool:
    """Check if an OpKind is a well-known operation (not STATELESS or STATEFUL)."""
    return isinstance(op, OpKind) and op not in (OpKind.STATELESS, OpKind.STATEFUL)


class _WellKnownOp:
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
        return ("WellKnownOp", self._kind.name, self._kind.value)

    def compile(
        self,
        input_types: tuple[TypeDescriptor, ...],
        output_type: TypeDescriptor | None = None,
        context: OpContext = OpContext.BINARY_OP,
    ) -> Op:
        # Well-known ops don't need LTOIR - CCCL handles them internally
        return Op(
            operator_type=self._kind,
            name="",
            ltoir=b"",
            state_alignment=1,
            state=b"",
        )

    @property
    def func(self) -> Callable | None:
        return None

    @property
    def kind(self) -> OpKind:
        """The underlying OpKind."""
        return self._kind


class CompiledOp:
    """
    Pre-compiled operator from LTOIR bytecode.

    This allows users to bring their own compiler (BYOC) by providing
    pre-compiled LTOIR rather than relying on Numba for JIT compilation.

    The LTOIR must follow the CCCL ABI convention where all arguments and
    the return value are passed as void pointers. The exact signature depends
    on the OpContext in which the op is used.

    Example:
        from cuda.compute import CompiledOp
        from cuda.core import Program, ProgramOptions

        # Compile C++ to LTOIR using cuda.core
        source = '''
        extern "C" __device__ void my_add(void* a, void* b, void* result) {
            *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
        }
        '''
        opts = ProgramOptions(arch="sm_80", relocatable_device_code=True,
                              link_time_optimization=True)
        ltoir = Program(source, "c++", options=opts).compile("ltoir").code

        add_op = CompiledOp(ltoir, "my_add")
        reduce_into(d_in, d_out, add_op, num_items, h_init)
    """

    __slots__ = ["_ltoir", "_name", "_extra_ltoirs"]

    def __init__(
        self,
        ltoir: bytes,
        name: str,
        extra_ltoirs: Sequence[bytes] | None = None,
    ):
        """
        Create a pre-compiled operator from LTOIR bytecode.

        Args:
            ltoir: LTOIR bytecode compiled from C++ source
            name: The symbol name of the device function (must match extern "C" name)
            extra_ltoirs: Additional LTOIR modules to link (for composite ops that
                         call other compiled functions)
        """
        if not isinstance(ltoir, bytes):
            raise TypeError(f"ltoir must be bytes, got {type(ltoir).__name__}")
        if not ltoir:
            raise ValueError("ltoir cannot be empty")
        if not isinstance(name, str):
            raise TypeError(f"name must be str, got {type(name).__name__}")
        if not name:
            raise ValueError("name cannot be empty")

        self._ltoir = ltoir
        self._name = name
        self._extra_ltoirs = tuple(extra_ltoirs) if extra_ltoirs else ()

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        return ("CompiledOp", hash(self._ltoir), self._name)

    def compile(
        self,
        input_types: tuple[TypeDescriptor, ...] | None = None,
        output_type: TypeDescriptor | None = None,
        context: OpContext = OpContext.BINARY_OP,
    ) -> Op:
        """Return the pre-compiled Op (ignores types and context)."""
        return Op(
            operator_type=OpKind.STATELESS,
            name=self._name,
            ltoir=self._ltoir,
            state_alignment=1,
            state=b"",
            extra_ltoirs=list(self._extra_ltoirs) if self._extra_ltoirs else None,
        )

    @property
    def name(self) -> str:
        """The symbol name of the compiled function."""
        return self._name

    @property
    def ltoir(self) -> bytes:
        """The LTOIR bytecode."""
        return self._ltoir

    @property
    def extra_ltoirs(self) -> tuple[bytes, ...]:
        """Additional LTOIR modules to link."""
        return self._extra_ltoirs

    @property
    def func(self) -> Callable | None:
        """The underlying callable (None for compiled ops)."""
        return None


class _JitOp:
    """
    Internal wrapper for JIT-compiled callables.

    This wraps Python callables that will be compiled to LTOIR when
    the compile() method is called.
    """

    __slots__ = ["_func", "_cachable"]

    def __init__(self, func: Callable):
        self._func = func
        self._cachable = CachableFunction(func)

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this operator."""
        return self._cachable

    def compile(
        self,
        input_types: tuple[TypeDescriptor, ...],
        output_type: TypeDescriptor | None = None,
        context: OpContext = OpContext.BINARY_OP,
    ) -> Op:
        """Compile this operator using Numba JIT."""
        # Import Numba-specific code lazily
        from ._jit import compile_jit_op

        return compile_jit_op(self._func, input_types, output_type, context)

    @property
    def func(self) -> Callable:
        """Access the wrapped callable."""
        return self._func


# =============================================================================
# Public API
# =============================================================================

# Public alias for the protocol
OpAdapter = OpProtocol


def make_op_adapter(op) -> OpProtocol:
    """
    Create an Op adapter from a callable, well-known OpKind, or CompiledOp.

    Args:
        op: Callable, OpKind, CompiledOp, or existing op adapter

    Returns:
        An object implementing OpProtocol

    Raises:
        ImportError: If a callable is passed but Numba is not installed
    """
    # Already implements OpProtocol
    if isinstance(op, (CompiledOp, _WellKnownOp, _JitOp)):
        return op

    # Well-known operation
    if isinstance(op, OpKind):
        return _WellKnownOp(op)

    # Callable - requires JIT compilation
    if callable(op):
        # Check if JIT compilation is available
        try:
            import numba.cuda as _  # noqa: F401
        except ImportError:
            raise ImportError(
                "Using Python callables as operators requires the JIT compiler "
                "(numba-cuda), but it is not installed.\n\n"
                "Install with: pip install cuda-cccl[cu12]  # or cu13\n\n"
                "Alternatively, use CompiledOp with pre-compiled LTOIR, "
                "or use well-known operations like OpKind.PLUS."
            ) from None

        return _JitOp(op)

    raise TypeError(
        f"Cannot create op adapter from {type(op).__name__}. "
        "Expected Callable, OpKind, or CompiledOp."
    )


__all__ = [
    "CompiledOp",
    "OpAdapter",
    "OpKind",
    "OpProtocol",
    "make_op_adapter",
]
