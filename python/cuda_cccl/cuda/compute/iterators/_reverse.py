# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ReverseIterator implementation."""

from __future__ import annotations

from .._bindings import IteratorState
from .._cpp_codegen import compile_cpp_to_ltoir
from ..types import TypeDescriptor
from ._base import IteratorBase, _deterministic_suffix


def _ensure_iterator_for_reverse(obj):
    """Wrap array in PointerIterator at the END of the array for reverse iteration."""
    from ._pointer import PointerIterator, PointerIteratorAtOffset

    if isinstance(obj, PointerIterator):
        return PointerIteratorAtOffset(obj._array, offset_from_end=1)
    if hasattr(obj, "__cuda_array_interface__"):
        return PointerIteratorAtOffset(obj, offset_from_end=1)
    if isinstance(obj, IteratorBase):
        return obj

    raise TypeError("ReverseIterator requires a device array or iterator")


class ReverseIterator(IteratorBase):
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

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None

        super().__init__(
            state_bytes=bytes(self._underlying.state),
            state_alignment=self._underlying.state_alignment,
            value_type=self._underlying.value_type,
        )

        # Generate deterministic suffix after super().__init__() so self.kind is available
        self._uid = _deterministic_suffix(self.kind)

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

    @property
    def children(self):
        return (self._underlying,)

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

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ReverseIterator", self._underlying.kind)

    def __add__(self, offset: int):
        return ReverseIterator(self._underlying + offset)
