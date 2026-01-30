# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PermutationIterator implementation."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from .._cpp_codegen import cpp_type_from_descriptor
from ._base import IteratorBase
from ._codegen_utils import (
    ADVANCE_TEMPLATE,
    INPUT_DEREF_TEMPLATE,
    OUTPUT_DEREF_TEMPLATE,
    compose_iterator_states,
    format_template,
)

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
        "_values_offset",
        "_indices_offset",
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

        # Compose states from both iterators
        state_bytes, state_alignment, offsets = compose_iterator_states(
            [self._values, self._indices]
        )
        self._values_offset = offsets[0]
        self._indices_offset = offsets[1]

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=state_alignment,
            value_type=self._values.value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str, list[bytes]]:
        """Generate advance that only advances indices iterator."""
        indices_advance, _, _ = self._indices.get_advance_ltoir()
        symbol = self._make_advance_symbol()

        body = dedent(f"""
            char* indices_state = static_cast<char*>(state) + {self._indices_offset};
            {indices_advance}(indices_state, offset);
        """).strip()

        source = format_template(
            ADVANCE_TEMPLATE, symbol=symbol, body=body, extern_symbols=[indices_advance]
        )
        return (symbol, source, [])

    def _generate_input_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        """Generate input deref that reads index then accesses values."""
        idx_result = self._indices.get_input_dereference_ltoir()
        if idx_result is None:
            raise ValueError("Indices iterator must support input dereference")
        indices_deref, _, _ = idx_result

        val_in_result = self._values.get_input_dereference_ltoir()
        if val_in_result is None:
            return None
        values_deref, _, _ = val_in_result

        # Also need values advance for random access
        values_advance, val_adv_ltoir, val_adv_extras = self._values.get_advance_ltoir()

        symbol = self._make_input_deref_symbol()
        idx_type = cpp_type_from_descriptor(self._indices.value_type)
        values_state_size = len(bytes(memoryview(self._values.state)))

        body = dedent(f"""
            char* values_state = static_cast<char*>(state) + {self._values_offset};
            char* indices_state = static_cast<char*>(state) + {self._indices_offset};

            alignas({self._indices.value_type.alignment}) {idx_type} idx;
            {indices_deref}(indices_state, &idx);

            alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
            memcpy(temp_values, values_state, {values_state_size});

            uint64_t offset = static_cast<uint64_t>(idx);
            {values_advance}(temp_values, &offset);
            {values_deref}(temp_values, result);
        """).strip()

        source = format_template(
            INPUT_DEREF_TEMPLATE,
            symbol=symbol,
            body=body,
            extern_symbols=[indices_deref, values_advance, values_deref],
        )
        # Include values.advance since we reference it
        return (symbol, source, [val_adv_ltoir] + val_adv_extras)

    def _generate_output_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        """Generate output deref that reads index then writes to values."""
        idx_result = self._indices.get_input_dereference_ltoir()
        if idx_result is None:
            raise ValueError("Indices iterator must support input dereference")
        indices_deref, _, _ = idx_result

        val_out_result = self._values.get_output_dereference_ltoir()
        if val_out_result is None:
            return None
        values_deref, _, _ = val_out_result

        # Also need values advance for random access
        values_advance, val_adv_ltoir, val_adv_extras = self._values.get_advance_ltoir()

        symbol = self._make_output_deref_symbol()
        idx_type = cpp_type_from_descriptor(self._indices.value_type)
        values_state_size = len(bytes(memoryview(self._values.state)))

        body = dedent(f"""
            char* values_state = static_cast<char*>(state) + {self._values_offset};
            char* indices_state = static_cast<char*>(state) + {self._indices_offset};

            alignas({self._indices.value_type.alignment}) {idx_type} idx;
            {indices_deref}(indices_state, &idx);

            alignas({self._values.state_alignment}) char temp_values[{values_state_size}];
            memcpy(temp_values, values_state, {values_state_size});

            uint64_t offset = static_cast<uint64_t>(idx);
            {values_advance}(temp_values, &offset);
            {values_deref}(temp_values, value);
        """).strip()

        source = format_template(
            OUTPUT_DEREF_TEMPLATE,
            symbol=symbol,
            body=body,
            extern_symbols=[indices_deref, values_advance, values_deref],
        )
        # Include values.advance since we reference it
        return (symbol, source, [val_adv_ltoir] + val_adv_extras)

    @property
    def children(self):
        return (self._values, self._indices)

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
