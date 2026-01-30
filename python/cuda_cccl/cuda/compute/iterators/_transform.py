# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TransformIterator implementation."""

from __future__ import annotations

from textwrap import dedent

from .._bindings import IteratorState
from .._cpp_codegen import (
    cpp_type_from_descriptor,
    extract_extra_ltoirs,
)
from ..op import make_op_adapter
from ..types import TypeDescriptor
from ._base import IteratorBase
from ._codegen_utils import (
    ADVANCE_TEMPLATE,
    INPUT_DEREF_TEMPLATE,
    OUTPUT_DEREF_TEMPLATE,
    format_template,
)


class TransformIterator(IteratorBase):
    """
    Iterator that applies a transform operation to values from an underlying iterator.

    For input iteration: reads from underlying, applies transform, returns result.
    For output iteration: applies transform to input, writes to underlying.
    """

    __slots__ = [
        "_underlying",
        "_transform_op",
        "_value_type",
        "_is_input",
        "_compiled_op",
    ]

    def __init__(
        self,
        underlying: IteratorBase,
        transform_op,
        output_value_type: TypeDescriptor,
        is_input: bool = True,
    ):
        """
        Create a transform iterator.

        Args:
            underlying: The underlying iterator to transform
            transform_op: The unary transform operation (OpProtocol)
            output_value_type: TypeDescriptor for the transformed value type
            is_input: True for input iterator, False for output iterator
        """
        self._underlying = underlying
        self._transform_op = make_op_adapter(transform_op)
        self._value_type = output_value_type
        self._is_input = is_input
        self._compiled_op = None  # Lazy compiled Op

        super().__init__(
            state_bytes=bytes(self._underlying.state),
            state_alignment=self._underlying.state_alignment,
            value_type=output_value_type,
        )

    def _get_compiled_op(self):
        """Get the compiled Op, compiling lazily if needed."""
        if self._compiled_op is None:
            if self._is_input:
                input_type = self._underlying.value_type
                output_type = self._value_type
            else:
                input_type = self._value_type
                output_type = self._underlying.value_type

            self._compiled_op = self._transform_op.compile(
                (input_type,),
                output_type,
            )
        return self._compiled_op

    def _generate_advance_source(self) -> tuple[str, str, list[bytes]]:
        """Generate advance that delegates to underlying iterator."""
        underlying_advance, _, _ = self._underlying.get_advance_ltoir()
        symbol = self._make_advance_symbol()

        body = dedent(f"""
            {underlying_advance}(state, offset);
        """).strip()

        source = format_template(
            ADVANCE_TEMPLATE,
            symbol=symbol,
            body=body,
            extern_symbols=[underlying_advance],
        )
        return (symbol, source, [])

    def _generate_input_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        """Generate input dereference that reads from underlying then transforms."""
        if not self._is_input:
            return None

        underlying_result = self._underlying.get_input_dereference_ltoir()
        if underlying_result is None:
            raise ValueError("Underlying iterator must support input dereference")
        underlying_deref, _, _ = underlying_result

        compiled_op = self._get_compiled_op()
        transform_op = compiled_op.name
        op_ltoir = compiled_op.ltoir
        op_extras = extract_extra_ltoirs(compiled_op)

        symbol = self._make_input_deref_symbol()
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        body = dedent(f"""
            alignas({self._underlying.value_type.alignment}) {underlying_type} temp;
            {underlying_deref}(state, &temp);
            {transform_op}(&temp, result);
        """).strip()

        source = format_template(
            INPUT_DEREF_TEMPLATE,
            symbol=symbol,
            body=body,
            extern_symbols=[underlying_deref, transform_op],
        )
        return (symbol, source, [op_ltoir] + op_extras)

    def _generate_output_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        """Generate output dereference that transforms then writes to underlying."""
        if self._is_input:
            return None

        underlying_result = self._underlying.get_output_dereference_ltoir()
        if underlying_result is None:
            raise ValueError("Underlying iterator must support output dereference")
        underlying_deref, _, _ = underlying_result

        compiled_op = self._get_compiled_op()
        transform_op = compiled_op.name
        op_ltoir = compiled_op.ltoir
        op_extras = extract_extra_ltoirs(compiled_op)

        symbol = self._make_output_deref_symbol()
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        body = dedent(f"""
            alignas({self._underlying.value_type.alignment}) {underlying_type} temp;
            {transform_op}(value, &temp);
            {underlying_deref}(state, &temp);
        """).strip()

        source = format_template(
            OUTPUT_DEREF_TEMPLATE,
            symbol=symbol,
            body=body,
            extern_symbols=[underlying_deref, transform_op],
        )
        return (symbol, source, [op_ltoir] + op_extras)

    def advance(self, offset: int) -> "TransformIterator":
        """Return a new iterator advanced by offset elements."""
        if not hasattr(self._underlying, "__add__"):
            raise AttributeError("Underlying iterator does not support advance")
        return TransformIterator(
            self._underlying + offset,  # type: ignore[operator, arg-type]
            self._transform_op,
            self._value_type,
            is_input=self._is_input,
        )

    def __add__(self, offset: int) -> "TransformIterator":
        return self.advance(offset)

    def __radd__(self, offset: int) -> "TransformIterator":
        return self.advance(offset)

    @property
    def state(self) -> IteratorState:
        return self._underlying.state

    @property
    def state_alignment(self) -> int:
        return self._underlying.state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        return self._value_type

    @property
    def children(self):
        return (self._underlying,)

    @property
    def is_input_iterator(self) -> bool:
        return self._is_input

    @property
    def is_output_iterator(self) -> bool:
        return not self._is_input

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        # Convert _value_type to tuple if it's a list (for output iterators)
        value_type = (
            tuple(self._value_type)
            if isinstance(self._value_type, list)
            else self._value_type
        )
        return (
            "TransformIterator",
            self._is_input,
            self._transform_op,
            self._underlying.kind,
            value_type,
        )
