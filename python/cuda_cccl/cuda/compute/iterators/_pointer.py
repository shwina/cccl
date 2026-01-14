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


def _try_get_cpp_type(value_type):
    """Try to get C++ type name, return None for struct types."""
    try:
        return cpp_type_from_descriptor(value_type)
    except ValueError:
        return None


class PointerIterator(IteratorBase):
    """
    Simple iterator wrapping a device array pointer.

    Supports both input (reading) and output (writing) operations.
    Handles both scalar types (using typed C++ code) and struct types
    (using byte-level memcpy).
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
        self._cpp_type = _try_get_cpp_type(value_type)  # None for struct types
        self._element_size = value_type.size
        self._array = array  # Keep reference to prevent GC

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # pointer alignment
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"pointer_advance_{self._uid}"

        if self._cpp_type:
            # Scalar type - use typed pointer arithmetic
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
    auto dist = *static_cast<int64_t*>(offset);
    *ptr_state += dist;
}}
"""
        else:
            # Struct type - use byte-level pointer arithmetic
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    auto* ptr_state = static_cast<char**>(state);
    auto dist = *static_cast<int64_t*>(offset);
    *ptr_state += dist * {self._element_size};
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        symbol = f"pointer_input_deref_{self._uid}"

        if self._cpp_type:
            # Scalar type - use typed dereference
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
    *static_cast<{self._cpp_type}*>(result) = **ptr_state;
}}
"""
        else:
            # Struct type - use memcpy
            source = f"""
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    auto* ptr_state = static_cast<char**>(state);
    memcpy(result, *ptr_state, {self._element_size});
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        symbol = f"pointer_output_deref_{self._uid}"

        if self._cpp_type:
            # Scalar type - use typed dereference
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* value) {{
    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
    **ptr_state = *static_cast<{self._cpp_type}*>(value);
}}
"""
        else:
            # Struct type - use memcpy
            source = f"""
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* value) {{
    auto* ptr_state = static_cast<char**>(state);
    memcpy(*ptr_state, value, {self._element_size});
}}
"""
        return (symbol, source)


class PointerIteratorAtOffset(IteratorBase):
    """
    Pointer iterator that starts at an offset from the end of an array.

    Used by ReverseIterator to start at the last element.
    """

    def __init__(self, array, offset_from_end: int = 0):
        """
        Create a pointer iterator at an offset from the end.

        Args:
            array: Device array with __cuda_array_interface__
            offset_from_end: How many elements from the end (1 = last element)
        """
        # Get pointer and dtype from array
        ptr = get_data_pointer(array)
        dtype = get_dtype(array)
        value_type = from_numpy_dtype(dtype)

        # Calculate the offset pointer
        # For offset_from_end=1, we want ptr + (len-1) * itemsize
        array_len = array.shape[0] if hasattr(array, "shape") else len(array)
        offset_ptr = ptr + (array_len - offset_from_end) * dtype.itemsize

        # State is just the pointer (8 bytes on 64-bit)
        state_bytes = ctypes.c_void_p(offset_ptr)
        state_bytes_buffer = (ctypes.c_char * 8)()
        ctypes.memmove(state_bytes_buffer, ctypes.byref(state_bytes), 8)
        state_bytes = bytes(state_bytes_buffer)

        self._uid = _unique_suffix()
        self._cpp_type = _try_get_cpp_type(value_type)  # None for struct types
        self._element_size = value_type.size
        self._array = array  # Keep reference to prevent GC

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # pointer alignment
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"pointer_advance_{self._uid}"

        if self._cpp_type:
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
    auto dist = *static_cast<int64_t*>(offset);
    *ptr_state += dist;
}}
"""
        else:
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    auto* ptr_state = static_cast<char**>(state);
    auto dist = *static_cast<int64_t*>(offset);
    *ptr_state += dist * {self._element_size};
}}
"""
        return (symbol, source)

    def _generate_input_deref_source(self) -> tuple[str, str] | None:
        symbol = f"pointer_input_deref_{self._uid}"

        if self._cpp_type:
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
    *static_cast<{self._cpp_type}*>(result) = **ptr_state;
}}
"""
        else:
            source = f"""
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* result) {{
    auto* ptr_state = static_cast<char**>(state);
    memcpy(result, *ptr_state, {self._element_size});
}}
"""
        return (symbol, source)

    def _generate_output_deref_source(self) -> tuple[str, str] | None:
        symbol = f"pointer_output_deref_{self._uid}"

        if self._cpp_type:
            source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* value) {{
    auto* ptr_state = static_cast<{self._cpp_type}**>(state);
    **ptr_state = *static_cast<{self._cpp_type}*>(value);
}}
"""
        else:
            source = f"""
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
using namespace cuda::std;

extern "C" __device__ void {symbol}(void* state, void* value) {{
    auto* ptr_state = static_cast<char**>(state);
    memcpy(*ptr_state, value, {self._element_size});
}}
"""
        return (symbol, source)
