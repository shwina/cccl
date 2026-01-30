# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base classes for iterators.
"""

from __future__ import annotations

import hashlib
from typing import Hashable

from .._bindings import Iterator, IteratorKind, IteratorState, Op, OpKind
from .._caching import cache_with_registered_key_functions
from ..types import TypeDescriptor


class IteratorBase:
    """
    Base class for iterators that use C++ codegen.

    Subclasses must implement:
    - _generate_advance_source() -> tuple[str, str, list[bytes]]  # (symbol, source, extra_ltoirs)
    - _generate_input_deref_source() -> tuple[str, str, list[bytes]] | None
    - _generate_output_deref_source() -> tuple[str, str, list[bytes]] | None

    Optionally override:
    - children property to return tuple of child iterators for automatic dependency tracking

    The base class handles compilation, caching, and dependency collection.
    """

    __slots__ = [
        "_state_bytes",
        "_state_alignment",
        "_value_type",
        "_advance_ltoir",
        "_input_deref_ltoir",
        "_output_deref_ltoir",
        "_uid_cached",
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
        self._advance_ltoir: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_ltoir: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_ltoir: tuple[str, bytes, list[bytes]] | None = None
        self._uid_cached: str | None = None

    @property
    def state(self) -> IteratorState:
        """Return the iterator state for CCCL interop."""
        return IteratorState(self._state_bytes)

    @property
    def cvalue(self):
        """Return a ctypes representation of the iterator state for backward compatibility."""
        import ctypes
        import sys

        # For pointer-based iterators, state_bytes contains a pointer value
        # Convert it to ctypes.c_void_p
        if len(self._state_bytes) == 8:  # Pointer size on 64-bit systems
            ptr_value = int.from_bytes(self._state_bytes, sys.byteorder, signed=False)
            return ctypes.c_void_p(ptr_value)
        # For other iterators, just wrap the raw bytes
        return ctypes.c_char_p(self._state_bytes)

    @property
    def state_alignment(self) -> int:
        """Return the alignment of the iterator state."""
        return self._state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        """Return the TypeDescriptor for dereferenced values."""
        return self._value_type

    @property
    def children(self) -> tuple["IteratorBase", ...]:
        """Return child iterators for automatic dependency tracking. Override in subclasses."""
        return ()

    def _get_uid(self) -> str:
        """Return a deterministic unique identifier for this iterator type."""
        if self._uid_cached is None:
            self._uid_cached = _deterministic_suffix(self.kind)
        return self._uid_cached

    def _make_advance_symbol(self) -> str:
        """Generate symbol name for advance operation."""
        return f"{self.__class__.__name__}_advance_{self._get_uid()}"

    def _make_input_deref_symbol(self) -> str:
        """Generate symbol name for input dereference operation."""
        return f"{self.__class__.__name__}_input_deref_{self._get_uid()}"

    def _make_output_deref_symbol(self) -> str:
        """Generate symbol name for output dereference operation."""
        return f"{self.__class__.__name__}_output_deref_{self._get_uid()}"

    def _collect_child_ltoirs(self, op_type: str) -> list[bytes]:
        """Collect LTOIRs from all children for the specified operation type."""
        extras = []
        for child in self.children:
            if op_type == "advance":
                _, ltoir, child_extras = child.get_advance_ltoir()
                extras.extend([ltoir] + child_extras)
            elif op_type == "input_deref":
                result = child.get_input_dereference_ltoir()
                if result is not None:
                    _, ltoir, child_extras = result
                    extras.extend([ltoir] + child_extras)
            elif op_type == "output_deref":
                result = child.get_output_dereference_ltoir()
                if result is not None:
                    _, ltoir, child_extras = result
                    extras.extend([ltoir] + child_extras)
        return extras

    def _compile_if_needed(self) -> None:
        """Compile the iterator ops if not already compiled."""
        if self._advance_ltoir is not None:
            return

        from .._cpp_codegen import compile_cpp_to_ltoir

        # Ensure children are compiled first
        for child in self.children:
            child._compile_if_needed()

        # Compile advance
        adv_symbol, adv_source, adv_self_extras = self._generate_advance_source()
        adv_ltoir = compile_cpp_to_ltoir(adv_source, (adv_symbol,))
        adv_child_extras = self._collect_child_ltoirs("advance")
        self._advance_ltoir = (
            adv_symbol,
            adv_ltoir,
            adv_child_extras + adv_self_extras,
        )

        # Compile input dereference if available
        input_deref = self._generate_input_deref_source()
        if input_deref is not None:
            in_symbol, in_source, in_self_extras = input_deref
            in_ltoir = compile_cpp_to_ltoir(in_source, (in_symbol,))
            in_child_extras = self._collect_child_ltoirs("input_deref")
            self._input_deref_ltoir = (
                in_symbol,
                in_ltoir,
                in_child_extras + in_self_extras,
            )

        # Compile output dereference if available
        output_deref = self._generate_output_deref_source()
        if output_deref is not None:
            out_symbol, out_source, out_self_extras = output_deref
            out_ltoir = compile_cpp_to_ltoir(out_source, (out_symbol,))
            out_child_extras = self._collect_child_ltoirs("output_deref")
            self._output_deref_ltoir = (
                out_symbol,
                out_ltoir,
                out_child_extras + out_self_extras,
            )

    def get_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        """Get the LTOIR for the advance operation."""
        self._compile_if_needed()
        assert self._advance_ltoir is not None
        return self._advance_ltoir

    def get_input_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Get the LTOIR for input dereference operation."""
        self._compile_if_needed()
        if self._input_deref_ltoir is None:
            return None
        return self._input_deref_ltoir

    def get_output_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Get the LTOIR for output dereference operation."""
        self._compile_if_needed()
        if self._output_deref_ltoir is None:
            return None
        return self._output_deref_ltoir

    @property
    def is_input_iterator(self) -> bool:
        """Return True if this iterator supports input dereference."""
        return self._generate_input_deref_source() is not None

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        return self._generate_output_deref_source() is not None

    def to_cccl_iter(self, is_output: bool = False) -> Iterator:
        """
        Convert this iterator to a CCCL Iterator for algorithm interop.

        Args:
            is_output: If True, use output_dereference; otherwise use input_dereference

        Returns:
            CCCL Iterator object
        """
        # Get advance op
        adv_name, adv_ltoir, adv_extras = self.get_advance_ltoir()
        advance_op = Op(
            operator_type=OpKind.STATELESS,
            name=adv_name,
            ltoir=adv_ltoir,
            extra_ltoirs=adv_extras if adv_extras else None,
        )

        # Get dereference op based on direction
        if is_output:
            deref_result = self.get_output_dereference_ltoir()
            if deref_result is None:
                raise ValueError("This iterator does not support output operations")
        else:
            deref_result = self.get_input_dereference_ltoir()
            if deref_result is None:
                raise ValueError("This iterator does not support input operations")

        deref_name, deref_ltoir, deref_extras = deref_result
        deref_op = Op(
            operator_type=OpKind.STATELESS,
            name=deref_name,
            ltoir=deref_ltoir,
            extra_ltoirs=deref_extras if deref_extras else None,
        )

        # Create the CCCL Iterator
        return Iterator(
            self._state_alignment,
            IteratorKind.ITERATOR,
            advance_op,
            deref_op,
            self._value_type.info,
            state=self.state,
        )

    @property
    def kind(self) -> Hashable:
        """Return a hashable kind for caching purposes.

        Note: state_bytes is intentionally excluded - iterators with the same
        type structure but different runtime state should share cached reducers.
        """
        return (type(self).__name__, self._value_type)

    # Abstract methods for subclasses
    def _generate_advance_source(self) -> tuple[str, str, list[bytes]]:
        """Generate C++ source for advance operation. Returns (symbol, source, extra_ltoirs)."""
        raise NotImplementedError

    def _generate_input_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        """Generate C++ source for input dereference. Returns (symbol, source, extra_ltoirs) or None."""
        raise NotImplementedError

    def _generate_output_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        """Generate C++ source for output dereference. Returns (symbol, source, extra_ltoirs) or None."""
        raise NotImplementedError


def _deterministic_suffix(kind: Hashable) -> str:
    """
    Generate a deterministic suffix from an iterator's kind.

    This ensures that iterators with the same logical structure
    (same kind) generate identical symbol names, allowing compilation
    results to be shared across iterator instances.

    Args:
        kind: The iterator's kind (must be hashable)

    Returns:
        A 16-character hexadecimal string derived from the kind
    """
    kind_str = str(kind)
    return hashlib.sha256(kind_str.encode()).hexdigest()[:16]


cache_with_registered_key_functions.register(IteratorBase, lambda it: it.kind)
