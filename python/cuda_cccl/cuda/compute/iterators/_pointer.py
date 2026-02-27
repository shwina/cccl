# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PointerIterator implementation - simple bidirectional iterator for device arrays."""

from __future__ import annotations

import ctypes
import sys

from .._bindings import Op, make_pointer_iterator_ops
from .._utils.protocols import get_data_pointer, get_dtype
from ..types import from_numpy_dtype
from ._base import IteratorBase


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

        # State is just the pointer
        state_bytes = ctypes.c_void_p(ptr)
        state_bytes_buffer = (ctypes.c_char * 8)()
        ctypes.memmove(state_bytes_buffer, ctypes.byref(state_bytes), 8)
        state_bytes = bytes(state_bytes_buffer)

        self._array = array  # Keep reference to prevent GC

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,  # pointer alignment
            value_type=value_type,
        )

    @property
    def array(self):
        return self._array

    def _get_c_input_ops(self):
        if not hasattr(self, "_c_input_ops"):
            self._c_input_ops = make_pointer_iterator_ops(
                self._value_type.info, self._state_bytes, True, False
            )
        return self._c_input_ops

    def _get_c_output_ops(self):
        if not hasattr(self, "_c_output_ops"):
            self._c_output_ops = make_pointer_iterator_ops(
                self._value_type.info, self._state_bytes, False, True
            )
        return self._c_output_ops

    def _make_advance_op(self) -> Op:
        return self._get_c_input_ops()[0]

    def _make_input_deref_op(self) -> Op | None:
        return self._get_c_input_ops()[1]

    def _make_output_deref_op(self) -> Op | None:
        return self._get_c_output_ops()[1]

    def __add__(self, offset: int):
        dtype = get_dtype(self._array)
        offset_ptr = self._current_pointer() + offset * dtype.itemsize
        return self._clone_with_pointer(offset_ptr)

    def _current_pointer(self) -> int:
        return int.from_bytes(self._state_bytes, sys.byteorder, signed=False)

    def _clone_with_pointer(self, pointer_value: int):
        """Clone this iterator with a different pointer value."""
        clone = PointerIterator(self._array)
        state_bytes_buffer = (ctypes.c_char * 8)()
        ptr_obj = ctypes.c_void_p(pointer_value)
        ctypes.memmove(state_bytes_buffer, ctypes.byref(ptr_obj), 8)
        clone._state_bytes = bytes(state_bytes_buffer)
        clone._uid_cached = None
        return clone
