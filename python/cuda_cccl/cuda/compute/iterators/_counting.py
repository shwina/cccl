# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CountingIterator implementation."""

from __future__ import annotations

import numpy as np

from .._bindings import Op, make_counting_iterator_ops
from ..types import from_numpy_dtype
from ._base import IteratorBase


class CountingIterator(IteratorBase):
    """
    Iterator representing a sequence of incrementing values.

    Similar to `thrust::counting_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1counting__iterator.html>`_.

    The iterator starts at `start` and increments by 1 for each advance.

    Example:
        The code snippet below demonstrates the usage of a ``CountingIterator``
        representing the sequence ``[10, 11, 12]``:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/counting_iterator_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        start: The initial value of the sequence
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

    def _get_c_ops(self):
        if not hasattr(self, "_c_ops"):
            self._c_ops = make_counting_iterator_ops(
                self._value_type.info, self._state_bytes
            )
        return self._c_ops

    def _make_advance_op(self) -> Op:
        return self._get_c_ops()[0]

    def _make_input_deref_op(self) -> Op | None:
        return self._get_c_ops()[1]

    def _make_output_deref_op(self) -> Op | None:
        return None

    def __add__(self, offset: int) -> "CountingIterator":
        """Return a new CountingIterator advanced by offset elements."""
        new_start = self._start_value + offset
        return CountingIterator(new_start)
