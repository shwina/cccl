# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ShuffleIterator implementation."""

from __future__ import annotations

import struct

import numpy as np

from .._bindings import Op, make_shuffle_iterator_ops
from ..types import from_numpy_dtype
from ._base import IteratorBase


class ShuffleIterator(IteratorBase):
    """
    Iterator that produces a deterministic random permutation of indices.

    At position ``i``, yields ``bijection(i)`` where the bijection is a random
    permutation of ``[0, num_items)`` parameterized by ``seed``.

    Example:
        The code snippet below demonstrates the usage of a ``ShuffleIterator``
        to randomly permute indices:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/shuffle_iterator_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        num_items: Number of elements in the domain to permute. Must be > 0.
        seed: Seed for the random permutation. Different seeds produce
            different (deterministic) permutations. Defaults to 0.
    """

    __slots__ = ["_num_items", "_seed", "_current_index", "_c_ops"]

    def __init__(self, num_items: int, seed: int = 0, *, _current_index: int = 0):
        if num_items <= 0:
            raise ValueError("num_items must be > 0")

        self._num_items = int(num_items)
        self._seed = int(seed)
        self._current_index = int(_current_index)
        self._c_ops = None

        # State layout matches C++ ShuffleState:
        #   int64_t  current_index  (offset 0,  size 8)
        #   uint64_t num_items      (offset 8,  size 8)
        #   uint64_t seed           (offset 16, size 8)
        state_bytes = struct.pack(
            "<qQQ", self._current_index, self._num_items, self._seed
        )

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,
            value_type=from_numpy_dtype(np.dtype("int64")),
        )

    def _get_c_ops(self) -> tuple:
        if self._c_ops is None:
            self._c_ops = make_shuffle_iterator_ops(self._state_bytes)
        return self._c_ops

    def _make_advance_op(self) -> Op:
        return self._get_c_ops()[0]

    def _make_input_deref_op(self) -> Op | None:
        return self._get_c_ops()[1]

    def _make_output_deref_op(self) -> Op | None:
        return None

    def __add__(self, offset: int) -> "ShuffleIterator":
        return ShuffleIterator(
            self._num_items,
            self._seed,
            _current_index=self._current_index + offset,
        )
