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
from ..op import OpContext, OpProtocol

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
        "_compiled_op",
    ]

    def __init__(
        self,
        underlying: "IteratorProtocol",
        transform_op: OpProtocol,
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
        self._transform_op = transform_op
        self._value_type = output_value_type
        self._is_input = is_input
        self._uid = _unique_suffix()

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._compiled_op = None  # Lazy compiled Op

    def _get_compiled_op(self):
        """Get the compiled Op, compiling lazily if needed."""
        if self._compiled_op is None:
            from .._bindings import TypeEnum

            if self._is_input:
                # Input iterator: transform reads from underlying and outputs value_type
                # Input type is underlying's value_type
                input_type = self._underlying.value_type
                output_type = self._value_type
            else:
                # Output iterator: transform takes algorithm output and writes to underlying
                # Input type is value_type (what algorithm writes), output is underlying's type
                input_type = self._value_type
                output_type = self._underlying.value_type

            # For struct types (STORAGE), let Numba infer the output type
            # since we can't easily convert TypeDescriptor back to Numba type
            if output_type.type_enum == TypeEnum.STORAGE:
                output_type = None

            # Compile with UNARY_OP context
            self._compiled_op = self._transform_op.compile(
                input_types=(input_type,),
                output_type=output_type,
                context=OpContext.UNARY_OP,
            )
        return self._compiled_op

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
        compiled_op = self._get_compiled_op()
        op_name = compiled_op.name
        op_ltoir = compiled_op.ltoir
        op_extras = list(compiled_op.extra_ltoirs) if compiled_op.extra_ltoirs else []

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
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

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
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

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
#include <cuda/std/cstdint>
using namespace cuda::std;

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

    def advance(self, offset: int) -> "TransformIterator":
        """Return a new iterator advanced by offset elements."""
        if not hasattr(self._underlying, "advance"):
            raise AttributeError("Underlying iterator does not support advance")
        return TransformIterator(
            self._underlying.advance(offset),
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

    def to_cccl_iter(self, is_output: bool = False):
        """Convert to CCCL Iterator for algorithm interop."""
        from .._bindings import Iterator, IteratorKind, Op, OpKind

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

        return Iterator(
            self.state_alignment,
            IteratorKind.ITERATOR,
            advance_op,
            deref_op,
            self._value_type.to_type_info(),
            state=self.state,
        )

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return (
            "TransformIterator",
            self._is_input,
            self._transform_op.get_cache_key(),
            self._underlying.kind,
            self._value_type,
        )
