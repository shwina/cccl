# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DiscardIterator implementation."""

from __future__ import annotations

from .._utils.protocols import get_dtype
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..types import TypeDescriptor, from_numpy_dtype
from ._base import IteratorBase, _deterministic_suffix


class DiscardIterator(IteratorBase):
    """
    Iterator that discards all reads and writes.
    """

    def __init__(self, reference_iterator=None):
        """
        Create a discard iterator.

        Args:
            reference_iterator: Optional iterator or device array used to infer
                value_type/state_type. Defaults to a temporary byte buffer.
        """
        if reference_iterator is None:
            reference_iterator = TempStorageBuffer(1)

        self._reference_iterator = reference_iterator

        if hasattr(reference_iterator, "__cuda_array_interface__"):
            value_type = from_numpy_dtype(get_dtype(reference_iterator))
            state_bytes = bytes(value_type.dtype.itemsize)
        elif isinstance(reference_iterator, IteratorBase):
            value_type = reference_iterator.value_type
            if isinstance(value_type, TypeDescriptor):
                state_bytes = bytes(value_type.dtype.itemsize)
            else:
                state_bytes = bytes(value_type.info.size)
        else:
            raise TypeError("reference_iterator must be a device array or iterator")

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

        # Generate deterministic suffix after super().__init__() so self.kind is available
        self._uid = _deterministic_suffix(self.kind)

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"discard_advance_{self._uid}"

        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    (void)state;
    (void)offset;
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        symbol = f"discard_input_deref_{self._uid}"

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    (void)state;
    (void)result;
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        symbol = f"discard_output_deref_{self._uid}"

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* value) {{
    (void)state;
    (void)value;
}}
"""
        return (symbol, source)

    def __add__(self, offset: int) -> "DiscardIterator":
        """Return a new DiscardIterator (stateless, so position doesn't matter)."""
        # DiscardIterator is stateless - advance is a no-op
        # If reference_iterator supports advance, advance it; otherwise use as-is
        ref = self._reference_iterator
        if isinstance(ref, IteratorBase) and hasattr(ref, "__add__"):
            ref = ref + offset  # type: ignore[operator]
        return DiscardIterator(ref)
