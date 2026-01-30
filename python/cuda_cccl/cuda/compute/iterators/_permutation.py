# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PermutationIterator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._cpp_codegen import compile_cpp_to_ltoir, cpp_type_from_descriptor
from ._base import IteratorBase, _deterministic_suffix

if TYPE_CHECKING:
    pass


def _ensure_iterator(obj):
    """Wrap array in PointerIterator if needed."""
    from ._pointer import PointerIterator

    if isinstance(obj, IteratorBase):
        return obj
    if hasattr(obj, "__cuda_array_interface__"):
        return PointerIterator(obj)
    raise TypeError("PermutationIterator requires iterators or device arrays")


class PermutationIterator(IteratorBase):
    """
    Iterator that accesses values through an index mapping.

    At position i, yields values[indices[i]]. The indices iterator is advanced
    normally, and the values iterator is accessed via random access using the
    index value.
    """

    __slots__ = [
        "_values",
        "_indices",
        "_uid",
        "_values_offset",
        "_indices_offset",
        "_advance_result",
        "_input_deref_result",
        "_output_deref_result",
    ]

    def __init__(
        self,
        values,
        indices,
    ):
        """
        Create a permutation iterator.

        Args:
            values: Iterator or array providing the values to be permuted
            indices: Iterator or array providing the indices for permutation
        """
        # Wrap arrays in PointerIterator
        self._values = _ensure_iterator(values)
        self._indices = _ensure_iterator(indices)

        # State contains both iterators' states concatenated
        values_state = bytes(memoryview(self._values.state))
        indices_state = bytes(memoryview(self._indices.state))

        values_size = len(values_state)
        indices_align = self._indices.state_alignment
        padding = (indices_align - (values_size % indices_align)) % indices_align

        state_bytes = values_state + (b"\x00" * padding) + indices_state
        state_alignment = max(
            self._values.state_alignment, self._indices.state_alignment
        )
        self._values_offset = 0
        self._indices_offset = values_size + padding

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=state_alignment,
            value_type=self._values.value_type,
        )

        # Generate deterministic suffix after super().__init__() so self.kind is available
        self._uid = _deterministic_suffix(self.kind)

    def _compile_if_needed(self) -> None:
        if self._advance_result is not None:
            return

        # Get indices iterator ops
        idx_adv_name, idx_adv_ltoir, idx_adv_extras = self._indices.get_advance_ltoir()

        idx_result = self._indices.get_input_dereference_ltoir()
        if idx_result is None:
            raise ValueError("Indices iterator must support input dereference")
        idx_deref_name, idx_deref_ltoir, idx_deref_extras = idx_result

        # Get values iterator ops
        val_adv_name, val_adv_ltoir, val_adv_extras = self._values.get_advance_ltoir()

        # Compile advance (only advances indices)
        symbol, ltoir = self._compile_advance(idx_adv_name)
        self._advance_result = (symbol, ltoir, [idx_adv_ltoir] + idx_adv_extras)

        # Input dereference
        val_in_result = self._values.get_input_dereference_ltoir()
        if val_in_result is not None:
            val_deref_name, val_deref_ltoir, val_deref_extras = val_in_result
            symbol, ltoir = self._compile_input_deref(
                idx_deref_name, val_adv_name, val_deref_name
            )
            self._input_deref_result = (
                symbol,
                ltoir,
                [idx_deref_ltoir]
                + idx_deref_extras
                + [val_adv_ltoir]
                + val_adv_extras
                + [val_deref_ltoir]
                + val_deref_extras,
            )

        # Output dereference
        val_out_result = self._values.get_output_dereference_ltoir()
        if val_out_result is not None:
            val_out_name, val_out_ltoir, val_out_extras = val_out_result
            symbol, ltoir = self._compile_output_deref(
                idx_deref_name, val_adv_name, val_out_name
            )
            self._output_deref_result = (
                symbol,
                ltoir,
                [idx_deref_ltoir]
                + idx_deref_extras
                + [val_adv_ltoir]
                + val_adv_extras
                + [val_out_ltoir]
                + val_out_extras,
            )

    def _compile_advance(self, indices_advance: str) -> tuple[str, bytes]:
        symbol = f"permutation_advance_{self._uid}"
        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {indices_advance}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    char* indices_state = static_cast<char*>(state) + {self._indices_offset};
    {indices_advance}(indices_state, offset);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_input_deref(
        self,
        indices_deref: str,
        values_advance: str,
        values_deref: str,
    ) -> tuple[str, bytes]:
        symbol = f"permutation_input_deref_{self._uid}"
        idx_type = cpp_type_from_descriptor(self._indices.value_type)
        values_state_size = len(bytes(memoryview(self._values.state)))

        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;
#include <cuda/std/cstring>

extern "C" __device__ void {indices_deref}(void*, void*);
extern "C" __device__ void {values_advance}(void*, void*);
extern "C" __device__ void {values_deref}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* result) {{
    char* values_state = static_cast<char*>(state) + {self._values_offset};
    char* indices_state = static_cast<char*>(state) + {self._indices_offset};

    alignas({self._indices.value_type.alignment}) {idx_type} idx;
    {indices_deref}(indices_state, &idx);

    alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
    memcpy(temp_values, values_state, {values_state_size});

    uint64_t offset = static_cast<uint64_t>(idx);
    {values_advance}(temp_values, &offset);
    {values_deref}(temp_values, result);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_output_deref(
        self,
        indices_deref: str,
        values_advance: str,
        values_deref: str,
    ) -> tuple[str, bytes]:
        symbol = f"permutation_output_deref_{self._uid}"
        idx_type = cpp_type_from_descriptor(self._indices.value_type)
        values_state_size = len(bytes(memoryview(self._values.state)))

        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;
#include <cuda/std/cstring>

extern "C" __device__ void {indices_deref}(void*, void*);
extern "C" __device__ void {values_advance}(void*, void*);
extern "C" __device__ void {values_deref}(void*, void*);

extern "C" __device__ void {symbol}(void* state, void* value) {{
    char* values_state = static_cast<char*>(state) + {self._values_offset};
    char* indices_state = static_cast<char*>(state) + {self._indices_offset};

    alignas({self._indices.value_type.alignment}) {idx_type} idx;
    {indices_deref}(indices_state, &idx);

    alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
    memcpy(temp_values, values_state, {values_state_size});

    uint64_t offset = static_cast<uint64_t>(idx);
    {values_advance}(temp_values, &offset);
    {values_deref}(temp_values, value);
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    @property
    def children(self):
        return (self._values, self._indices)

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
        return self._values.is_input_iterator

    @property
    def is_output_iterator(self) -> bool:
        return self._values.is_output_iterator

    def __add__(self, offset: int) -> "PermutationIterator":
        """Advance the indices iterator by offset, keeping values at base."""
        return PermutationIterator(
            self._values,  # values stays at base for random access
            self._indices + offset,  # only indices advances  # type: ignore[operator]
        )

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("PermutationIterator", self._values.kind, self._indices.kind)
