# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base classes for iterators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .._bindings import IteratorState
from .._types import TypeDescriptor

if TYPE_CHECKING:
    pass


class IteratorBase:
    """
    Base class for iterators that use C++ codegen.

    Subclasses must implement:
    - _generate_advance_source() -> tuple[str, str]  # (symbol, source)
    - _generate_input_deref_source() -> tuple[str, str] | None
    - _generate_output_deref_source() -> tuple[str, str] | None

    The base class handles compilation and caching.
    """

    __slots__ = [
        "_state_bytes",
        "_state_alignment",
        "_value_type",
        "_advance_ltoir",
        "_input_deref_ltoir",
        "_output_deref_ltoir",
        "_extra_ltoirs",
    ]

    def __init__(
        self,
        state_bytes: bytes,
        state_alignment: int,
        value_type: TypeDescriptor,
    ):
        self._state_bytes = state_bytes
        self._state_alignment = state_alignment
        self._value_type = value_type
        self._advance_ltoir: tuple[str, bytes] | None = None
        self._input_deref_ltoir: tuple[str, bytes] | None = None
        self._output_deref_ltoir: tuple[str, bytes] | None = None
        self._extra_ltoirs: list[bytes] = []

    @property
    def state(self) -> IteratorState:
        """Return the iterator state for CCCL interop."""
        return IteratorState(self._state_bytes)

    @property
    def state_alignment(self) -> int:
        """Return the alignment of the iterator state."""
        return self._state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        """Return the TypeDescriptor for dereferenced values."""
        return self._value_type

    def _compile_if_needed(self) -> None:
        """Compile the iterator ops if not already compiled."""
        if self._advance_ltoir is not None:
            return

        from .._codegen import compile_cpp_to_ltoir

        # Compile advance
        adv_symbol, adv_source = self._generate_advance_source()
        adv_ltoir = compile_cpp_to_ltoir(adv_source, (adv_symbol,))
        self._advance_ltoir = (adv_symbol, adv_ltoir)

        # Compile input dereference if available
        input_deref = self._generate_input_deref_source()
        if input_deref is not None:
            in_symbol, in_source = input_deref
            in_ltoir = compile_cpp_to_ltoir(in_source, (in_symbol,))
            self._input_deref_ltoir = (in_symbol, in_ltoir)

        # Compile output dereference if available
        output_deref = self._generate_output_deref_source()
        if output_deref is not None:
            out_symbol, out_source = output_deref
            out_ltoir = compile_cpp_to_ltoir(out_source, (out_symbol,))
            self._output_deref_ltoir = (out_symbol, out_ltoir)

    def get_advance_ltoir(self) -> tuple[str, bytes]:
        """Get the LTOIR for the advance operation."""
        self._compile_if_needed()
        assert self._advance_ltoir is not None
        return self._advance_ltoir

    def get_input_dereference_ltoir(self) -> tuple[str, bytes] | None:
        """Get the LTOIR for input dereference operation."""
        self._compile_if_needed()
        return self._input_deref_ltoir

    def get_output_dereference_ltoir(self) -> tuple[str, bytes] | None:
        """Get the LTOIR for output dereference operation."""
        self._compile_if_needed()
        return self._output_deref_ltoir

    @property
    def is_input_iterator(self) -> bool:
        """Return True if this iterator supports input dereference."""
        return self._generate_input_deref_source() is not None

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        return self._generate_output_deref_source() is not None

    @property
    def extra_ltoirs(self) -> list[bytes]:
        """Return additional LTOIR modules needed for linking."""
        return self._extra_ltoirs

    # Abstract methods for subclasses
    def _generate_advance_source(self) -> tuple[str, str]:
        """Generate C++ source for advance operation. Returns (symbol, source)."""
        raise NotImplementedError

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        """Generate C++ source for input dereference. Returns (symbol, source) or None."""
        raise NotImplementedError

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        """Generate C++ source for output dereference. Returns (symbol, source) or None."""
        raise NotImplementedError


class CompiledIterator:
    """
    Pre-compiled iterator from LTOIR bytecode.

    This allows users to bring their own compiled iterators by providing
    pre-compiled LTOIR for advance and dereference operations.

    Example:
        from cuda.compute.iterators2 import CompiledIterator
        from cuda.compute._codegen import compile_cpp_to_ltoir

        # C++ source for a simple counting iterator
        advance_src = '''
        extern "C" __device__ void my_advance(void* state, void* offset) {
            *static_cast<int*>(state) += *static_cast<uint64_t*>(offset);
        }
        '''
        deref_src = '''
        extern "C" __device__ void my_deref(void* state, void* result) {
            *static_cast<int*>(result) = *static_cast<int*>(state);
        }
        '''

        adv_ltoir = compile_cpp_to_ltoir(advance_src, ("my_advance",))
        deref_ltoir = compile_cpp_to_ltoir(deref_src, ("my_deref",))

        iter = CompiledIterator(
            state_bytes=struct.pack("i", 0),
            state_alignment=4,
            value_type=types.int32,
            advance_ltoir=("my_advance", adv_ltoir),
            input_deref_ltoir=("my_deref", deref_ltoir),
        )
    """

    __slots__ = [
        "_state_bytes",
        "_state_alignment",
        "_value_type",
        "_advance_ltoir",
        "_input_deref_ltoir",
        "_output_deref_ltoir",
        "_extra_ltoirs",
    ]

    def __init__(
        self,
        state_bytes: bytes,
        state_alignment: int,
        value_type: TypeDescriptor,
        advance_ltoir: tuple[str, bytes],
        input_deref_ltoir: tuple[str, bytes] | None = None,
        output_deref_ltoir: tuple[str, bytes] | None = None,
        extra_ltoirs: Sequence[bytes] | None = None,
    ):
        """
        Create a pre-compiled iterator.

        Args:
            state_bytes: Raw bytes of the iterator state
            state_alignment: Alignment of the state in bytes
            value_type: TypeDescriptor for the value type
            advance_ltoir: Tuple of (symbol_name, ltoir_bytes) for advance
            input_deref_ltoir: Optional tuple for input dereference
            output_deref_ltoir: Optional tuple for output dereference
            extra_ltoirs: Additional LTOIR modules to link
        """
        if not isinstance(state_bytes, bytes):
            raise TypeError(f"state_bytes must be bytes, got {type(state_bytes)}")
        if not isinstance(advance_ltoir, tuple) or len(advance_ltoir) != 2:
            raise TypeError("advance_ltoir must be a (name, ltoir) tuple")

        self._state_bytes = state_bytes
        self._state_alignment = state_alignment
        self._value_type = value_type
        self._advance_ltoir = advance_ltoir
        self._input_deref_ltoir = input_deref_ltoir
        self._output_deref_ltoir = output_deref_ltoir
        self._extra_ltoirs = list(extra_ltoirs) if extra_ltoirs else []

    @property
    def state(self) -> IteratorState:
        """Return the iterator state for CCCL interop."""
        return IteratorState(self._state_bytes)

    @property
    def state_alignment(self) -> int:
        """Return the alignment of the iterator state."""
        return self._state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        """Return the TypeDescriptor for dereferenced values."""
        return self._value_type

    def get_advance_ltoir(self) -> tuple[str, bytes]:
        """Get the LTOIR for the advance operation."""
        return self._advance_ltoir

    def get_input_dereference_ltoir(self) -> tuple[str, bytes] | None:
        """Get the LTOIR for input dereference operation."""
        return self._input_deref_ltoir

    def get_output_dereference_ltoir(self) -> tuple[str, bytes] | None:
        """Get the LTOIR for output dereference operation."""
        return self._output_deref_ltoir

    @property
    def is_input_iterator(self) -> bool:
        """Return True if this iterator supports input dereference."""
        return self._input_deref_ltoir is not None

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        return self._output_deref_ltoir is not None

    @property
    def extra_ltoirs(self) -> list[bytes]:
        """Return additional LTOIR modules needed for linking."""
        return self._extra_ltoirs
