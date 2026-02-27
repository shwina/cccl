# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CacheModifiedInputIterator implementation."""

from __future__ import annotations

import struct
from typing import Literal

from .._bindings import Op, make_cache_modified_input_iterator_ops
from .._utils.protocols import get_data_pointer, get_dtype
from ..types import from_numpy_dtype
from ._base import IteratorBase

# Map modifier names to cccl_cache_modifier_t enum integer values
# CCCL_LOAD_CG = 2 (Cache at L2 only)
# CCCL_LOAD_CS = 3 (Cache streaming, evict first)
# CCCL_LOAD_CV = 4 (Don't cache, always fetch)
_MODIFIER_TO_ENUM = {
    "stream": 3,    # CCCL_LOAD_CS
    "global": 2,    # CCCL_LOAD_CG
    "volatile": 4,  # CCCL_LOAD_CV
}


class CacheModifiedInputIterator(IteratorBase):
    """
    Iterator that wraps a device pointer with cache-modified loads.

    This iterator uses PTX cache modifiers to control how data is loaded:
    - "stream": Uses streaming loads (ld.global.cs) - hints that data will not be reused
    - "global": Uses global cache loads (ld.global.cg) - caches only at L2
    - "volatile": Uses volatile loads (ld.global.cv) - always fetches from memory

    Supports element types of size 1, 2, 4, 8, or 16 bytes.
    """

    __slots__ = [
        "_modifier",
        "_array",
        "_ptr",
        "_c_ops",
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
        if modifier not in _MODIFIER_TO_ENUM:
            raise ValueError(
                f"Unknown modifier: {modifier}. Must be one of {list(_MODIFIER_TO_ENUM.keys())}"
            )

        self._modifier = modifier
        self._array = array  # Keep reference to prevent GC
        ptr = get_data_pointer(array)
        dtype = get_dtype(array)

        self._ptr = ptr
        value_type = from_numpy_dtype(dtype)

        # Cache-modified loads only supported for power-of-two sizes up to 16 bytes
        # These correspond to PTX instructions: ld.global.{modifier}.b{8,16,32,64,128}
        if value_type.size not in (1, 2, 4, 8, 16):
            raise ValueError(
                f"CacheModifiedInputIterator only supports types of size 1, 2, 4, 8, or 16 bytes. "
                f"Got type with size {value_type.size} bytes. "
                f"This matches PTX cache-modified load instruction limitations."
            )

        # State is just the pointer (8 bytes on 64-bit)
        state_bytes = struct.pack("Q", ptr)

        self._c_ops = None  # Cached C ops (advance, deref)

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # Pointer alignment
            value_type=value_type,
        )

    def _get_c_ops(self):
        if self._c_ops is None:
            self._c_ops = make_cache_modified_input_iterator_ops(
                self._value_type.info,
                self._state_bytes,
                _MODIFIER_TO_ENUM[self._modifier],
            )
        return self._c_ops

    def _make_advance_op(self) -> Op:
        return self._get_c_ops()[0]

    def _make_input_deref_op(self) -> Op | None:
        return self._get_c_ops()[1]

    def _make_output_deref_op(self) -> Op | None:
        # Cache-modified iterator is input-only
        return None

    def __add__(self, offset: int) -> "CacheModifiedInputIterator":
        """Advance the iterator by offset elements."""
        out = CacheModifiedInputIterator(self._array, self._modifier)
        offset_ptr = self._ptr + offset * get_dtype(out._array).itemsize
        out._ptr = offset_ptr
        out._state_bytes = struct.pack("Q", offset_ptr)
        out._uid_cached = None
        return out

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return (
            "CacheModifiedInputIterator",
            self._modifier,
            self._value_type,
        )
