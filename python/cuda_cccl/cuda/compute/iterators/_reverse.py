# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ReverseIterator implementation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .._bindings import IteratorState
from .._codegen import compile_cpp_to_ltoir
from .._types import TypeDescriptor

if TYPE_CHECKING:
    pass


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


def _is_iterator(obj) -> bool:
    """Check if an object is an iterator."""
    return hasattr(obj, "to_cccl_iter") and callable(obj.to_cccl_iter)


def _ensure_iterator_for_reverse(obj):
    """Wrap array in PointerIterator at the END of the array for reverse iteration."""
    if _is_iterator(obj):
        return obj
    from ._pointer import PointerIteratorAtOffset

    # For arrays, create an iterator at the last element
    return PointerIteratorAtOffset(obj, offset_from_end=1)


class ReverseIterator:
    """
    Iterator that reverses the direction of an underlying iterator.

    Advance with positive offset moves backward in the underlying iterator.
    """

    __slots__ = [
        "_underlying",
        "_uid",
        "_advance_result",
        "_input_deref_result",
        "_output_deref_result",
    ]

    def __init__(self, underlying):
        """
        Create a reverse iterator.

        Args:
            underlying: The underlying iterator or array to reverse
        """
        self._underlying = _ensure_iterator_for_reverse(underlying)
        self._uid = _unique_suffix()

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None

    def _compile_if_needed(self) -> None:
        if self._advance_result is not None:
            return

        adv_name, adv_ltoir, adv_extras = self._underlying.get_advance_ltoir()
        symbol, ltoir = self._compile_advance(adv_name)
        self._advance_result = (symbol, ltoir, [adv_ltoir] + adv_extras)

        in_result = self._underlying.get_input_dereference_ltoir()
        if in_result is not None:
            in_name, in_ltoir, in_extras = in_result
            symbol, ltoir = self._compile_input_deref(in_name)
            self._input_deref_result = (symbol, ltoir, [in_ltoir] + in_extras)

        out_result = self._underlying.get_output_dereference_ltoir()
        if out_result is not None:
            out_name, out_ltoir, out_extras = out_result
            symbol, ltoir = self._compile_output_deref(out_name)
            self._output_deref_result = (symbol, ltoir, [out_ltoir] + out_extras)

    def _compile_advance(self, underlying_advance: str) -> tuple[str, bytes]:
        symbol = f"reverse_advance_{self._uid}"
        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {underlying_advance}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    int64_t neg_offset = -static_cast<int64_t>(*static_cast<uint64_t*>(offset));
    {underlying_advance}(state, &neg_offset);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_input_deref(self, underlying_deref: str) -> tuple[str, bytes]:
        symbol = f"reverse_input_deref_{self._uid}"
        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {underlying_deref}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* result) {{
    {underlying_deref}(state, result);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_output_deref(self, underlying_deref: str) -> tuple[str, bytes]:
        symbol = f"reverse_output_deref_{self._uid}"
        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {underlying_deref}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* value) {{
    {underlying_deref}(state, value);
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
        return self._underlying.value_type

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
        return self._underlying.is_input_iterator

    @property
    def is_output_iterator(self) -> bool:
        return self._underlying.is_output_iterator

    def to_cccl_iter(self, is_output: bool = False):
        """Convert to CCCL Iterator for algorithm interop."""
        from .._bindings import Iterator, IteratorKind, Op, OpKind

        adv_name, adv_ltoir, adv_extras = self.get_advance_ltoir()
        advance_op = Op(
            operator_type=OpKind.STATELESS,
            name=adv_name,
            ltoir=adv_ltoir,
            extra_ltoirs=adv_extras if adv_extras else None,
        )

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
            self.value_type.to_type_info(),
            state=self.state,
        )

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ReverseIterator", self._underlying.kind)
