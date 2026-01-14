# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DiscardIterator implementation."""

from __future__ import annotations

import uuid

from .._types import TypeDescriptor, uint8
from ._base import IteratorBase


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


class DiscardIterator(IteratorBase):
    """
    Iterator that discards all values written to it.

    Can be used as both input and output iterator. Input dereference returns
    a default value, output dereference discards the value.
    """

    def __init__(self, value_type: TypeDescriptor | None = None):
        """
        Create a discard iterator.

        Args:
            value_type: Optional TypeDescriptor for the value type.
                       Defaults to uint8.
        """
        if value_type is None:
            value_type = uint8

        self._uid = _unique_suffix()

        super().__init__(
            state_bytes=b"\x00",
            state_alignment=1,
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"discard_advance_{self._uid}"

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    (void)state;
    (void)offset;
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        symbol = f"discard_input_deref_{self._uid}"

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* result) {{
    (void)state;
    (void)result;
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        symbol = f"discard_output_deref_{self._uid}"

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* value) {{
    (void)state;
    (void)value;
}}
"""
        return (symbol, source)
