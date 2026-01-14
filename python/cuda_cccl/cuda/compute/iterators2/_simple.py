# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Simple iterator implementations using C++ codegen.
"""

from __future__ import annotations

import uuid

import numpy as np

from .._codegen import cpp_type_from_descriptor
from .._types import TypeDescriptor, from_numpy_dtype
from ._base import IteratorBase


def _unique_suffix() -> str:
    """Generate a unique suffix for symbol names."""
    return uuid.uuid4().hex[:8]


class CountingIterator(IteratorBase):
    """
    Iterator representing a sequence of incrementing values.

    The iterator starts at `start` and increments by 1 for each advance.
    """

    def __init__(self, start: np.number):
        """
        Create a counting iterator starting at `start`.

        Args:
            start: The initial value (must be a numpy scalar)
        """
        if not isinstance(start, np.generic):
            start = np.array(start).flatten()[0]

        self._start_value = start
        value_type = from_numpy_dtype(start.dtype)
        state_bytes = start.tobytes()

        # Generate unique symbol names
        self._uid = _unique_suffix()

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        """Generate C++ source for advance operation."""
        symbol = f"counting_advance_{self._uid}"
        cpp_type = cpp_type_from_descriptor(self._value_type)

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    auto* s = static_cast<{cpp_type}*>(state);
    auto dist = *static_cast<uint64_t*>(offset);
    *s += static_cast<{cpp_type}>(dist);
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        """Generate C++ source for input dereference."""
        symbol = f"counting_deref_{self._uid}"
        cpp_type = cpp_type_from_descriptor(self._value_type)

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* result) {{
    *static_cast<{cpp_type}*>(result) = *static_cast<{cpp_type}*>(state);
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        """CountingIterator is input-only."""
        return None


class ConstantIterator(IteratorBase):
    """
    Iterator representing a sequence of constant values.

    Every dereference returns the same constant value.
    """

    def __init__(self, value: np.number):
        """
        Create a constant iterator with the given value.

        Args:
            value: The constant value (must be a numpy scalar)
        """
        if not isinstance(value, np.generic):
            value = np.array(value).flatten()[0]

        self._constant_value = value
        value_type = from_numpy_dtype(value.dtype)
        state_bytes = value.tobytes()

        self._uid = _unique_suffix()

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        """Generate C++ source for advance (no-op for constant iterator)."""
        symbol = f"constant_advance_{self._uid}"

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    // No-op: constant iterator state doesn't change
    (void)state;
    (void)offset;
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        """Generate C++ source for input dereference."""
        symbol = f"constant_deref_{self._uid}"
        cpp_type = cpp_type_from_descriptor(self._value_type)

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* result) {{
    *static_cast<{cpp_type}*>(result) = *static_cast<{cpp_type}*>(state);
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        """ConstantIterator is input-only."""
        return None


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
        from .._types import uint8

        if value_type is None:
            value_type = uint8

        self._uid = _unique_suffix()

        # State is just a placeholder byte
        super().__init__(
            state_bytes=b"\x00",
            state_alignment=1,
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        """Generate C++ source for advance (no-op for discard iterator)."""
        symbol = f"discard_advance_{self._uid}"

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    // No-op: discard iterator doesn't track position
    (void)state;
    (void)offset;
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        """Generate C++ source for input dereference (returns default value)."""
        symbol = f"discard_input_deref_{self._uid}"

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* result) {{
    // No-op: discard iterator returns garbage/default
    (void)state;
    (void)result;
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        """Generate C++ source for output dereference (discards value)."""
        symbol = f"discard_output_deref_{self._uid}"

        source = f"""
#include <cstdint>

extern "C" __device__ void {symbol}(void* state, void* value) {{
    // No-op: discard iterator discards all writes
    (void)state;
    (void)value;
}}
"""
        return (symbol, source)
