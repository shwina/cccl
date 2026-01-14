# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CacheModifiedInputIterator implementation."""

from __future__ import annotations

import uuid
from typing import Literal

from .._codegen import cpp_type_from_descriptor
from .._types import from_numpy_dtype
from .._utils.protocols import get_data_pointer, get_dtype
from ._base import IteratorBase


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


# Map modifier names to PTX cache operators and C++ intrinsics
_CACHE_MODIFIERS = {
    "stream": ("cs", "__ldcs"),  # Cache streaming (evict first)
    "global": ("cg", "__ldcg"),  # Cache at L2 only
    "volatile": ("cv", "__ldcv"),  # Don't cache, always fetch
}


class CacheModifiedInputIterator(IteratorBase):
    """
    Iterator that wraps a device pointer with cache-modified loads.

    This iterator uses PTX cache modifiers to control how data is loaded:
    - "stream": Uses streaming loads (ld.global.cs) - hints that data will not be reused
    - "global": Uses global cache loads (ld.global.cg) - caches only at L2
    - "volatile": Uses volatile loads (ld.global.cv) - always fetches from memory
    """

    def __init__(
        self,
        array,
        modifier: Literal["stream", "global", "volatile"] = "stream",
    ):
        """
        Create a cache-modified input iterator.

        Args:
            array: Device array to wrap (must support __cuda_array_interface__)
            modifier: Cache modifier - "stream", "global", or "volatile"
        """
        if modifier not in _CACHE_MODIFIERS:
            raise ValueError(
                f"Unknown modifier: {modifier}. Must be one of {list(_CACHE_MODIFIERS.keys())}"
            )

        self._modifier = modifier
        self._array = array  # Keep reference to prevent GC
        ptr = get_data_pointer(array)
        dtype = get_dtype(array)

        self._ptr = ptr
        value_type = from_numpy_dtype(dtype)

        # State is just the pointer (8 bytes on 64-bit)
        import struct

        state_bytes = struct.pack("Q", ptr)

        self._uid = _unique_suffix()

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # Pointer alignment
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"cache_mod_advance_{self._uid}"
        cpp_type = cpp_type_from_descriptor(self._value_type)

        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    auto* s = static_cast<{cpp_type}**>(state);
    auto dist = *static_cast<uint64_t*>(offset);
    *s += dist;
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        symbol = f"cache_mod_deref_{self._uid}"
        cpp_type = cpp_type_from_descriptor(self._value_type)
        _, intrinsic = _CACHE_MODIFIERS[self._modifier]

        # For types that don't support direct __ldcs, we need to use inline PTX
        # __ldcs works for 4-byte and 8-byte types
        if self._value_type.size in (4, 8):
            source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    auto* ptr = *static_cast<{cpp_type}**>(state);
    *static_cast<{cpp_type}*>(result) = {intrinsic}(ptr);
}}
"""
        else:
            # For smaller types, fall back to regular load
            # (cache modifiers for small types aren't as beneficial)
            source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    auto* ptr = *static_cast<{cpp_type}**>(state);
    *static_cast<{cpp_type}*>(result) = *ptr;
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        # Cache-modified iterator is input-only
        return None

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return (
            "CacheModifiedInputIterator",
            self._modifier,
            self._value_type,
        )
