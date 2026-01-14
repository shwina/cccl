# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""DiscardIterator implementation."""

from __future__ import annotations

import uuid

from .._types import TypeDescriptor, from_numpy_dtype, uint8
from .._utils.protocols import get_dtype
from ._base import IteratorBase


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


def _get_value_type(arg) -> TypeDescriptor:
    """Extract TypeDescriptor from argument."""
    if arg is None:
        return uint8
    if isinstance(arg, TypeDescriptor):
        return arg
    # Assume it's an array-like, get dtype
    try:
        dtype = get_dtype(arg)
        return from_numpy_dtype(dtype)
    except Exception:
        return uint8


class DiscardIterator(IteratorBase):
    """
    Iterator that discards all values written to it.

    Can be used as both input and output iterator. Input dereference returns
    a default value, output dereference discards the value.
    """

    def __init__(self, value_type=None):
        """
        Create a discard iterator.

        Args:
            value_type: Optional TypeDescriptor, array, or None.
                       If array, the dtype is used. Defaults to uint8.
        """
        value_type = _get_value_type(value_type)

        self._uid = _unique_suffix()

        super().__init__(
            state_bytes=b"\x00",
            state_alignment=1,
            value_type=value_type,
        )

    def _generate_advance_source(self) -> tuple[str, str]:
        symbol = f"discard_advance_{self._uid}"

        source = f"""
#include <cuda/std/cstdint>
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
