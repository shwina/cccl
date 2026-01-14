# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PermutationIterator implementation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .._bindings import IteratorState
from .._codegen import compile_cpp_to_ltoir, cpp_type_from_descriptor
from .._types import TypeDescriptor

if TYPE_CHECKING:
    from ._protocol import IteratorProtocol


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


class PermutationIterator:
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
        "_state_bytes",
        "_state_alignment",
        "_values_offset",
        "_indices_offset",
        "_advance_result",
        "_input_deref_result",
        "_output_deref_result",
    ]

    def __init__(
        self,
        values: IteratorProtocol,
        indices: IteratorProtocol,
    ):
        """
        Create a permutation iterator.

        Args:
            values: Iterator providing the values to be permuted
            indices: Iterator providing the indices for permutation
        """
        self._values = values
        self._indices = indices
        self._uid = _unique_suffix()

        # State contains both iterators' states concatenated
        values_state = bytes(memoryview(values.state))
        indices_state = bytes(memoryview(indices.state))

        values_size = len(values_state)
        indices_align = indices.state_alignment
        padding = (indices_align - (values_size % indices_align)) % indices_align

        self._state_bytes = values_state + (b"\x00" * padding) + indices_state
        self._state_alignment = max(values.state_alignment, indices.state_alignment)
        self._values_offset = 0
        self._indices_offset = values_size + padding

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None

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
#include <cstdint>

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
#include <cstdint>
#include <cstring>

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
#include <cstdint>
#include <cstring>

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
    def state(self) -> IteratorState:
        return IteratorState(self._state_bytes)

    @property
    def state_alignment(self) -> int:
        return self._state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        return self._values.value_type

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
