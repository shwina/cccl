# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CountingIterator implementation."""

from __future__ import annotations

import uuid

import numpy as np

from .._codegen import cpp_type_from_descriptor
from .._types import from_numpy_dtype
from ._base import IteratorBase


def _unique_suffix() -> str:
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

        self._uid = _unique_suffix()

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
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
        return None
