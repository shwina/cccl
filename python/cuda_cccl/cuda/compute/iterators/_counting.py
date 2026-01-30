# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CountingIterator implementation."""

from __future__ import annotations

import numpy as np

from .._cpp_codegen import cpp_type_from_descriptor
from ..types import from_numpy_dtype
from ._base import IteratorBase, _deterministic_suffix


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

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

        # Generate deterministic suffix after super().__init__() so self.kind is available
        self._uid = _deterministic_suffix(self.kind)

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"counting_advance_{self._uid}"
        cpp_type = cpp_type_from_descriptor(self._value_type)

        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

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
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    *static_cast<{cpp_type}*>(result) = *static_cast<{cpp_type}*>(state);
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        return None

    def __add__(self, offset: int) -> "CountingIterator":
        """Return a new CountingIterator advanced by offset elements."""
        new_start = self._start_value + offset
        return CountingIterator(new_start)
