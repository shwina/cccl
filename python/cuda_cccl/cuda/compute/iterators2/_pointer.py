# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PointerIterator implementation - simple bidirectional iterator for device arrays."""

from __future__ import annotations

import ctypes
import uuid
from typing import TYPE_CHECKING

from .._codegen import cpp_type_from_descriptor
from .._types import from_numpy_dtype
from .._utils.protocols import get_data_pointer, get_dtype
from ._base import IteratorBase

if TYPE_CHECKING:
    pass


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


class PointerIterator(IteratorBase):
    """
    Simple iterator wrapping a device array pointer.

    Supports both input (reading) and output (writing) operations.
    """

    def __init__(self, array):
        """
        Create a pointer iterator from a device array.

        Args:
            array: Device array with __cuda_array_interface__
        """
        # Get pointer and dtype from array
        ptr = get_data_pointer(array)
        dtype = get_dtype(array)
        value_type = from_numpy_dtype(dtype)

        # State is just the pointer (8 bytes on 64-bit)
        state_bytes = ctypes.c_void_p(ptr)
        state_bytes_buffer = (ctypes.c_char * 8)()
        ctypes.memmove(state_bytes_buffer, ctypes.byref(state_bytes), 8)
        state_bytes = bytes(state_bytes_buffer)

        self._uid = _unique_suffix()
        self._cpp_type = cpp_type_from_descriptor(value_type)
        self._array = array  # Keep reference to prevent GC

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # pointer alignment
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"pointer_advance_{self._uid}"
        cpp_type = self._cpp_type

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    auto* ptr_state = static_cast<{cpp_type}**>(state);
    auto dist = *static_cast<uint64_t*>(offset);
    *ptr_state += dist;
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        symbol = f"pointer_input_deref_{self._uid}"
        cpp_type = self._cpp_type

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    auto* ptr_state = static_cast<{cpp_type}**>(state);
    *static_cast<{cpp_type}*>(result) = **ptr_state;
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        symbol = f"pointer_output_deref_{self._uid}"
        cpp_type = self._cpp_type

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* value) {{
    auto* ptr_state = static_cast<{cpp_type}**>(state);
    **ptr_state = *static_cast<{cpp_type}*>(value);
}}
"""
        return (symbol, source)
