# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CacheModifiedInputIterator implementation."""

from __future__ import annotations

from typing import Literal

from .._cpp_codegen import compile_cpp_to_ltoir, cpp_type_from_descriptor
from .._utils.protocols import get_data_pointer, get_dtype
from ..types import from_numpy_dtype
from ._base import IteratorBase, _deterministic_suffix

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

    __slots__ = [
        "_modifier",
        "_array",
        "_ptr",
        "_uid",
        "_advance_result",
        "_input_deref_result",
    ]

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

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # Pointer alignment
            value_type=value_type,
        )

        # Generate deterministic suffix after super().__init__() so self.kind is available
        self._uid = _deterministic_suffix(self.kind)

    def _compile_if_needed(self) -> None:
        if self._advance_result is not None:
            return

        # Compile advance
        symbol, ltoir = self._compile_advance()
        self._advance_result = (symbol, ltoir, [])

        # Compile input dereference
        symbol, ltoir = self._compile_input_deref()
        self._input_deref_result = (symbol, ltoir, [])

    def _compile_advance(self) -> tuple[str, bytes]:
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
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_input_deref(self) -> tuple[str, bytes]:
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
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def get_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        self._compile_if_needed()
        assert self._advance_result is not None
        return self._advance_result

    def get_input_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        self._compile_if_needed()
        return self._input_deref_result

    def get_output_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        # Cache-modified iterator is input-only
        return None

    @property
    def is_input_iterator(self) -> bool:
        return True

    @property
    def is_output_iterator(self) -> bool:
        return False

    def __add__(self, offset: int) -> "CacheModifiedInputIterator":
        """Advance the iterator by offset elements."""
        import struct

        from .._utils.protocols import get_dtype

        dtype = get_dtype(self._array)
        offset_ptr = self._ptr + offset * dtype.itemsize

        # Create a new instance with the offset pointer
        clone = CacheModifiedInputIterator.__new__(CacheModifiedInputIterator)
        clone._modifier = self._modifier
        clone._array = self._array
        clone._ptr = offset_ptr
        clone._state_bytes = struct.pack("Q", offset_ptr)
        clone._state_alignment = 8
        clone._value_type = self._value_type
        clone._advance_result = None
        clone._input_deref_result = None
        clone._uid = _deterministic_suffix(clone.kind)
        return clone

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return (
            "CacheModifiedInputIterator",
            self._modifier,
            self._value_type,
        )
