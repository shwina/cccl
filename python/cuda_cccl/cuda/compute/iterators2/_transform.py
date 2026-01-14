# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TransformIterator implementation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .._bindings import IteratorState
from .._codegen import compile_cpp_to_ltoir, cpp_type_from_descriptor
from .._types import TypeDescriptor
from ..op import CompiledOp

if TYPE_CHECKING:
    from ._protocol import IteratorProtocol


def _unique_suffix() -> str:
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
        "_advance_result",
        "_input_deref_result",
        "_output_deref_result",
        "_is_input",
    ]

    def __init__(
        self,
        underlying: IteratorProtocol,
        transform_op: CompiledOp,
        output_value_type: TypeDescriptor,
        is_input: bool = True,
    ):
        """
        Create a transform iterator.

        Args:
            underlying: The underlying iterator to transform
            transform_op: The unary transform operation (must be CompiledOp)
            output_value_type: TypeDescriptor for the transformed value type
            is_input: True for input iterator, False for output iterator
        """
        self._underlying = underlying
        self._transform_op = transform_op
        self._value_type = output_value_type
        self._is_input = is_input
        self._uid = _unique_suffix()

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None

    def _compile_if_needed(self) -> None:
        if self._advance_result is not None:
            return

        # Get underlying iterator's advance
        adv_name, adv_ltoir, adv_extras = self._underlying.get_advance_ltoir()

        # Compile our advance (delegates to underlying)
        symbol, ltoir = self._compile_advance(adv_name)
        # Our advance depends on underlying's advance
        self._advance_result = (symbol, ltoir, [adv_ltoir] + adv_extras)

        # Get transform op info
        op_name = self._transform_op.name
        op_ltoir = self._transform_op.ltoir
        op_extras = list(self._transform_op.extra_ltoirs)

        if self._is_input:
            in_result = self._underlying.get_input_dereference_ltoir()
            if in_result is None:
                raise ValueError("Underlying iterator must support input dereference")
            in_name, in_ltoir, in_extras = in_result

            symbol, ltoir = self._compile_input_deref(in_name, op_name)
            # Input deref depends on underlying deref + transform op
            self._input_deref_result = (
                symbol,
                ltoir,
                [in_ltoir] + in_extras + [op_ltoir] + op_extras,
            )
        else:
            out_result = self._underlying.get_output_dereference_ltoir()
            if out_result is None:
                raise ValueError("Underlying iterator must support output dereference")
            out_name, out_ltoir, out_extras = out_result

            symbol, ltoir = self._compile_output_deref(out_name, op_name)
            self._output_deref_result = (
                symbol,
                ltoir,
                [out_ltoir] + out_extras + [op_ltoir] + op_extras,
            )

    def _compile_advance(self, underlying_advance: str) -> tuple[str, bytes]:
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
        symbol = f"transform_input_deref_{self._uid}"
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_deref}(void*, void*);
extern "C" __device__ void {transform_op}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* result) {{
    alignas({self._underlying.value_type.alignment}) {underlying_type} temp;
    {underlying_deref}(state, &temp);
    {transform_op}(&temp, result);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_output_deref(
        self, underlying_deref: str, transform_op: str
    ) -> tuple[str, bytes]:
        symbol = f"transform_output_deref_{self._uid}"
        underlying_type = cpp_type_from_descriptor(self._underlying.value_type)

        source = f"""
#include <cstdint>

extern "C" __device__ void {underlying_deref}(void*, void*);
extern "C" __device__ void {transform_op}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* value) {{
    alignas({self._underlying.value_type.alignment}) {underlying_type} temp;
    {transform_op}(value, &temp);
    {underlying_deref}(state, &temp);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    @property
    def state(self) -> IteratorState:
        return self._underlying.state

    @property
    def state_alignment(self) -> int:
        return self._underlying.state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        return self._value_type

    def get_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        self._compile_if_needed()
        assert self._advance_result is not None
        return self._advance_result

    def get_input_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        self._compile_if_needed()
        return self._input_deref_result

    def get_output_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        self._compile_if_needed()
        return self._output_deref_result

    @property
    def is_input_iterator(self) -> bool:
        return self._is_input

    @property
    def is_output_iterator(self) -> bool:
        return not self._is_input
