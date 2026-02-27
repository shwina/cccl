# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ConstantIterator implementation."""

from __future__ import annotations

import numpy as np

from .._bindings import Op, make_constant_iterator_ops
from ..types import from_numpy_dtype
from ._base import IteratorBase


class ConstantIterator(IteratorBase):
    """
    Iterator representing a sequence of constant values.

    Similar to `thrust::constant_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1constant__iterator.html>`_.

    Every dereference returns the same constant value.

    Example:
        The code snippet below demonstrates the usage of a ``ConstantIterator``
        representing a sequence of constant values:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/constant_iterator_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        value: The value of every item in the sequence
    """

    def __init__(self, value: np.number):
        """
        Create a constant iterator with the given value.

        Args:
            value: The constant value (must be a numpy scalar)
        """
        if not isinstance(value, np.generic):
            value = np.array(value).flatten()[0]

        self._constant_value = value
        value_type = from_numpy_dtype(value.dtype)
        state_bytes = value.tobytes()

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=value_type.alignment,
            value_type=value_type,
        )

    def _get_c_ops(self):
        if not hasattr(self, "_c_ops"):
            self._c_ops = make_constant_iterator_ops(
                self._value_type.info, self._state_bytes
            )
        return self._c_ops

    def _make_advance_op(self) -> Op:
        return self._get_c_ops()[0]

    def _make_input_deref_op(self) -> Op | None:
        return self._get_c_ops()[1]

    def _make_output_deref_op(self) -> Op | None:
        return None

    def __add__(self, offset: int) -> "ConstantIterator":
        """Return a new ConstantIterator (value doesn't change with position)."""
        return ConstantIterator(self._constant_value)
