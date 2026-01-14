# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Composite iterator implementations using C++ codegen.

These iterators wrap other iterators and compose their operations.
The generated C++ calls underlying ops by symbol name, with LTOIRs
linked via extra_ltoirs.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .._bindings import IteratorState
from .._codegen import compile_cpp_to_ltoir, cpp_type_from_descriptor
from .._types import TypeDescriptor
from ..op import CompiledOp, OpProtocol

if TYPE_CHECKING:
    from ._protocol import IteratorProtocol


def _unique_suffix() -> str:
    """Generate a unique suffix for symbol names."""
    return uuid.uuid4().hex[:8]


class TransformIterator:
    """
    Iterator that applies a transform operation to values from an underlying iterator.

    For input iteration: reads from underlying, applies transform, returns result.
    For output iteration: applies transform to input, writes to underlying.
    """

    __slots__ = [
        "_underlying",
        "_transform_op",
        "_value_type",
        "_uid",
        "_advance_ltoir",
        "_input_deref_ltoir",
        "_output_deref_ltoir",
        "_extra_ltoirs",
        "_is_input",
    ]

    def __init__(
        self,
        underlying: IteratorProtocol,
        transform_op: OpProtocol | CompiledOp,
        output_value_type: TypeDescriptor,
        is_input: bool = True,
    ):
        """
        Create a transform iterator.

        Args:
            underlying: The underlying iterator to transform
            transform_op: The unary transform operation
            output_value_type: TypeDescriptor for the transformed value type
            is_input: True for input iterator, False for output iterator
        """
        self._underlying = underlying
        self._transform_op = transform_op
        self._value_type = output_value_type
        self._is_input = is_input
        self._uid = _unique_suffix()

        self._advance_ltoir: tuple[str, bytes] | None = None
        self._input_deref_ltoir: tuple[str, bytes] | None = None
        self._output_deref_ltoir: tuple[str, bytes] | None = None
        self._extra_ltoirs: list[bytes] = []

    def _compile_if_needed(self) -> None:
        """Compile the iterator ops if not already compiled."""
        if self._advance_ltoir is not None:
            return

        # Get underlying iterator's ops
        adv_name, adv_ltoir = self._underlying.get_advance_ltoir()
        self._extra_ltoirs.append(adv_ltoir)
        self._extra_ltoirs.extend(self._underlying.extra_ltoirs)

        # Compile advance - just delegates to underlying
        self._advance_ltoir = self._compile_advance(adv_name)

        if self._is_input:
            # Input transform: deref underlying, then apply op
            in_deref = self._underlying.get_input_dereference_ltoir()
            if in_deref is None:
                raise ValueError("Underlying iterator must support input dereference")
            in_name, in_ltoir = in_deref
            self._extra_ltoirs.append(in_ltoir)

            # Get transform op LTOIR
            if isinstance(self._transform_op, CompiledOp):
                op_name = self._transform_op.name
                self._extra_ltoirs.append(self._transform_op.ltoir)
                self._extra_ltoirs.extend(self._transform_op.extra_ltoirs)
            else:
                raise NotImplementedError(
                    "TransformIterator currently requires CompiledOp for transform"
                )

            self._input_deref_ltoir = self._compile_input_deref(in_name, op_name)
        else:
            # Output transform: apply op, then write to underlying
            out_deref = self._underlying.get_output_dereference_ltoir()
            if out_deref is None:
                raise ValueError("Underlying iterator must support output dereference")
            out_name, out_ltoir = out_deref
            self._extra_ltoirs.append(out_ltoir)

            if isinstance(self._transform_op, CompiledOp):
                op_name = self._transform_op.name
                self._extra_ltoirs.append(self._transform_op.ltoir)
                self._extra_ltoirs.extend(self._transform_op.extra_ltoirs)
            else:
                raise NotImplementedError(
                    "TransformIterator currently requires CompiledOp for transform"
                )

            self._output_deref_ltoir = self._compile_output_deref(out_name, op_name)

    def _compile_advance(self, underlying_advance: str) -> tuple[str, bytes]:
        """Compile advance that delegates to underlying."""
        symbol = f"transform_advance_{self._uid}"
        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_advance}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    {underlying_advance}(state, offset);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_input_deref(
        self, underlying_deref: str, transform_op: str
    ) -> tuple[str, bytes]:
        """Compile input dereference: deref underlying, then transform."""
        symbol = f"transform_input_deref_{self._uid}"
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_deref}(void*, void*);
extern "C" __device__ void {transform_op}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* result) {{
    // Temporary for underlying value
    alignas({self._underlying.value_type.alignment}) {underlying_type} temp;
    // Deref underlying
    {underlying_deref}(state, &temp);
    // Apply transform
    {transform_op}(&temp, result);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_output_deref(
        self, underlying_deref: str, transform_op: str
    ) -> tuple[str, bytes]:
        """Compile output dereference: transform input, then write to underlying."""
        symbol = f"transform_output_deref_{self._uid}"
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_deref}(void*, void*);
extern "C" __device__ void {transform_op}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* value) {{
    // Temporary for transformed value
    alignas({self._underlying.value_type.alignment}) {underlying_type} temp;
    // Apply transform
    {transform_op}(value, &temp);
    // Write to underlying
    {underlying_deref}(state, &temp);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    @property
    def state(self) -> IteratorState:
        """Return the iterator state (same as underlying)."""
        return self._underlying.state

    @property
    def state_alignment(self) -> int:
        """Return the alignment of the iterator state."""
        return self._underlying.state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        """Return the TypeDescriptor for transformed values."""
        return self._value_type

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
        return self._is_input

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        return not self._is_input

    @property
    def extra_ltoirs(self) -> list[bytes]:
        """Return additional LTOIR modules needed for linking."""
        self._compile_if_needed()
        return self._extra_ltoirs


class ReverseIterator:
    """
    Iterator that reverses the direction of an underlying iterator.

    Advance with positive offset moves backward in the underlying iterator.
    """

    __slots__ = [
        "_underlying",
        "_uid",
        "_advance_ltoir",
        "_input_deref_ltoir",
        "_output_deref_ltoir",
        "_extra_ltoirs",
    ]

    def __init__(self, underlying: IteratorProtocol):
        """
        Create a reverse iterator.

        Args:
            underlying: The underlying iterator to reverse
        """
        self._underlying = underlying
        self._uid = _unique_suffix()

        self._advance_ltoir: tuple[str, bytes] | None = None
        self._input_deref_ltoir: tuple[str, bytes] | None = None
        self._output_deref_ltoir: tuple[str, bytes] | None = None
        self._extra_ltoirs: list[bytes] = []

    def _compile_if_needed(self) -> None:
        """Compile the iterator ops if not already compiled."""
        if self._advance_ltoir is not None:
            return

        # Get underlying iterator's ops
        adv_name, adv_ltoir = self._underlying.get_advance_ltoir()
        self._extra_ltoirs.append(adv_ltoir)
        self._extra_ltoirs.extend(self._underlying.extra_ltoirs)

        # Compile advance with negated offset
        self._advance_ltoir = self._compile_advance(adv_name)

        # Input dereference - same as underlying
        in_deref = self._underlying.get_input_dereference_ltoir()
        if in_deref is not None:
            in_name, in_ltoir = in_deref
            self._extra_ltoirs.append(in_ltoir)
            self._input_deref_ltoir = self._compile_input_deref(in_name)

        # Output dereference - same as underlying
        out_deref = self._underlying.get_output_dereference_ltoir()
        if out_deref is not None:
            out_name, out_ltoir = out_deref
            self._extra_ltoirs.append(out_ltoir)
            self._output_deref_ltoir = self._compile_output_deref(out_name)

    def _compile_advance(self, underlying_advance: str) -> tuple[str, bytes]:
        """Compile advance that negates the offset."""
        symbol = f"reverse_advance_{self._uid}"
        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_advance}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    // Negate the offset for reverse iteration
    int64_t neg_offset = -static_cast<int64_t>(*static_cast<uint64_t*>(offset));
    {underlying_advance}(state, &neg_offset);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_input_deref(self, underlying_deref: str) -> tuple[str, bytes]:
        """Compile input dereference (delegates to underlying)."""
        symbol = f"reverse_input_deref_{self._uid}"
        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_deref}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* result) {{
    {underlying_deref}(state, result);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_output_deref(self, underlying_deref: str) -> tuple[str, bytes]:
        """Compile output dereference (delegates to underlying)."""
        symbol = f"reverse_output_deref_{self._uid}"
        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_deref}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* value) {{
    {underlying_deref}(state, value);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    @property
    def state(self) -> IteratorState:
        """Return the iterator state (same as underlying)."""
        return self._underlying.state

    @property
    def state_alignment(self) -> int:
        """Return the alignment of the iterator state."""
        return self._underlying.state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        """Return the TypeDescriptor (same as underlying)."""
        return self._underlying.value_type

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
        return self._underlying.is_input_iterator

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        return self._underlying.is_output_iterator

    @property
    def extra_ltoirs(self) -> list[bytes]:
        """Return additional LTOIR modules needed for linking."""
        self._compile_if_needed()
        return self._extra_ltoirs
