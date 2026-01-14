# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Base classes for iterators.
"""

from __future__ import annotations

from typing import Hashable

from .._bindings import Iterator, IteratorKind, IteratorState, Op, OpKind
from .._types import TypeDescriptor
from ..op import CompiledOp


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

    def get_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        """Get the LTOIR for the advance operation."""
        self._compile_if_needed()
        assert self._advance_ltoir is not None
        name, ltoir = self._advance_ltoir
        return (name, ltoir, [])  # Simple iterators have no extra deps

    def get_input_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Get the LTOIR for input dereference operation."""
        self._compile_if_needed()
        if self._input_deref_ltoir is None:
            return None
        name, ltoir = self._input_deref_ltoir
        return (name, ltoir, [])

    def get_output_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Get the LTOIR for output dereference operation."""
        self._compile_if_needed()
        if self._output_deref_ltoir is None:
            return None
        name, ltoir = self._output_deref_ltoir
        return (name, ltoir, [])

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
            self._value_type.to_type_info(),
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
    """

    __slots__ = [
        "_state_bytes",
        "_state_alignment",
        "_value_type",
        "_advance_op",
        "_input_deref_op",
        "_output_deref_op",
    ]

    def __init__(
        self,
        state_bytes: bytes,
        state_alignment: int,
        value_type: TypeDescriptor,
        advance: CompiledOp,
        input_dereference: CompiledOp | None = None,
        output_dereference: CompiledOp | None = None,
    ):
        """
        Create a pre-compiled iterator.

        Args:
            state_bytes: Raw bytes of the iterator state
            state_alignment: Alignment of the state in bytes
            value_type: TypeDescriptor for the value type
            advance: CompiledOp for the advance operation
            input_dereference: Optional CompiledOp for input dereference
            output_dereference: Optional CompiledOp for output dereference
        """
        if not isinstance(state_bytes, bytes):
            raise TypeError(f"state_bytes must be bytes, got {type(state_bytes)}")
        if not isinstance(state_alignment, int) or state_alignment <= 0:
            raise ValueError("state_alignment must be a positive power of 2")
        if state_alignment & (state_alignment - 1) != 0:
            raise ValueError("state_alignment must be a positive power of 2")
        if not isinstance(value_type, TypeDescriptor):
            raise TypeError(
                f"value_type must be TypeDescriptor, got {type(value_type)}"
            )
        if not isinstance(advance, CompiledOp):
            raise TypeError(f"advance must be a CompiledOp, got {type(advance)}")
        if input_dereference is None and output_dereference is None:
            raise ValueError(
                "At least one of input_dereference or output_dereference must be provided"
            )
        if input_dereference is not None and not isinstance(
            input_dereference, CompiledOp
        ):
            raise TypeError(
                f"input_dereference must be a CompiledOp, got {type(input_dereference)}"
            )
        if output_dereference is not None and not isinstance(
            output_dereference, CompiledOp
        ):
            raise TypeError(
                f"output_dereference must be a CompiledOp, got {type(output_dereference)}"
            )

        self._state_bytes = state_bytes
        self._state_alignment = state_alignment
        self._value_type = value_type
        self._advance_op = advance
        self._input_deref_op = input_dereference
        self._output_deref_op = output_dereference

    @staticmethod
    def _op_to_ltoir_tuple(op: CompiledOp) -> tuple[str, bytes, list[bytes]]:
        """Normalize a CompiledOp to (name, ltoir, extra_ltoirs)."""
        extra_ltoirs = list(op.extra_ltoirs) if op.extra_ltoirs else []
        return (op.name, op.ltoir, extra_ltoirs)

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

    def get_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        """Get the LTOIR for the advance operation."""
        return self._op_to_ltoir_tuple(self._advance_op)

    def get_input_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Get the LTOIR for input dereference operation."""
        if self._input_deref_op is None:
            return None
        return self._op_to_ltoir_tuple(self._input_deref_op)

    def get_output_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """Get the LTOIR for output dereference operation."""
        if self._output_deref_op is None:
            return None
        return self._op_to_ltoir_tuple(self._output_deref_op)

    @property
    def is_input_iterator(self) -> bool:
        """Return True if this iterator supports input dereference."""
        return self._input_deref_op is not None

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        return self._output_deref_op is not None

    def to_cccl_iter(self, is_output: bool = False) -> Iterator:
        """
        Convert this iterator to a CCCL Iterator for algorithm interop.

        Args:
            is_output: If True, use output_dereference; otherwise use input_dereference

        Returns:
            CCCL Iterator object
        """
        # Get advance op
        adv_name, adv_ltoir, adv_extras = self._op_to_ltoir_tuple(self._advance_op)
        advance_op = Op(
            operator_type=OpKind.STATELESS,
            name=adv_name,
            ltoir=adv_ltoir,
            extra_ltoirs=adv_extras if adv_extras else None,
        )

        # Get dereference op based on direction
        if is_output:
            if self._output_deref_op is None:
                raise ValueError("This iterator does not support output operations")
            deref_name, deref_ltoir, deref_extras = self._op_to_ltoir_tuple(
                self._output_deref_op
            )
        else:
            if self._input_deref_op is None:
                raise ValueError("This iterator does not support input operations")
            deref_name, deref_ltoir, deref_extras = self._op_to_ltoir_tuple(
                self._input_deref_op
            )

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
            self._value_type.to_type_info(),
            state=self.state,
        )

    @property
    def kind(self) -> Hashable:
        """Return a hashable kind for caching purposes."""
        return (
            "CompiledIterator",
            self._advance_op.get_cache_key(),
            self._input_deref_op.get_cache_key() if self._input_deref_op else None,
            self._output_deref_op.get_cache_key() if self._output_deref_op else None,
            self._value_type,
        )
