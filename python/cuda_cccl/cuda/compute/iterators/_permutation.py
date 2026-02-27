# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PermutationIterator implementation."""

from __future__ import annotations

from .._bindings import Iterator, IteratorKind, Op, make_permutation_iterator_ops
from ._base import IteratorBase, compose_iterator_states
from ._common import ensure_iterator


def _make_iter_struct(it: IteratorBase, advance_op: Op, deref_op: Op) -> Iterator:
    """Build a Cython Iterator struct from individual Op objects."""
    return Iterator(
        it.state_alignment,
        IteratorKind.ITERATOR,
        advance_op,
        deref_op,
        it.value_type.info,
        state=it.state,
    )


class PermutationIterator(IteratorBase):
    """
    Iterator that accesses values through an index mapping.

    At position i, yields values[indices[i]].

    Similar to `thrust::permutation_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1permutation__iterator.html>`_.

    Example:
        The code snippet below demonstrates accessing values through an index mapping.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/permutation_iterator_basic.py
            :language: python
            :start-after: # example-begin
    """

    __slots__ = [
        "_values",
        "_indices",
        "_values_offset",
        "_indices_offset",
    ]

    def __init__(
        self,
        values,
        indices,
    ):
        """
        Create a permutation iterator.

        Args:
            values: Iterator or array providing the values to be permuted
            indices: Iterator or array providing the indices for permutation
        """
        # Wrap arrays in PointerIterator
        self._values = ensure_iterator(values)
        self._indices = ensure_iterator(indices)

        # Compose states from both iterators
        state_bytes, state_alignment, offsets = compose_iterator_states(
            [self._values, self._indices]
        )
        self._values_offset = offsets[0]
        self._indices_offset = offsets[1]

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=state_alignment,
            value_type=self._values.value_type,
        )

    def _call_c_permutation(
        self,
        values_iter: Iterator,
        indices_iter: Iterator,
    ) -> tuple:
        """Call make_permutation_iterator_ops with the given child Iterators."""
        state_size = len(self._state_bytes)
        return make_permutation_iterator_ops(
            values_iter,
            indices_iter,
            self._values_offset,
            self._indices_offset,
            self._state_bytes,
            state_size,
            self._state_alignment,
        )

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that only advances indices iterator."""
        indices_advance = self._indices.get_advance_op()
        # Use any available deref for advance generation (C only uses advance name)
        indices_deref = (
            self._indices.get_input_deref_op()
            or self._indices.get_output_deref_op()
        )
        if indices_deref is None:
            raise ValueError("Indices iterator must support at least one dereference")

        values_advance = self._values.get_advance_op()
        values_deref = (
            self._values.get_input_deref_op()
            or self._values.get_output_deref_op()
        )
        if values_deref is None:
            raise ValueError("Values iterator must support at least one dereference")

        values_iter = _make_iter_struct(self._values, values_advance, values_deref)
        indices_iter = _make_iter_struct(self._indices, indices_advance, indices_deref)
        advance, _ = self._call_c_permutation(values_iter, indices_iter)
        return advance

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input deref that reads index then accesses values."""
        indices_deref = self._indices.get_input_deref_op()
        if indices_deref is None:
            raise ValueError("Indices iterator must support input dereference")

        values_deref = self._values.get_input_deref_op()
        if values_deref is None:
            return None

        values_advance = self._values.get_advance_op()
        indices_advance = self._indices.get_advance_op()

        values_iter = _make_iter_struct(self._values, values_advance, values_deref)
        indices_iter = _make_iter_struct(self._indices, indices_advance, indices_deref)
        _, deref = self._call_c_permutation(values_iter, indices_iter)
        return deref

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output deref that reads index then writes to values."""
        indices_deref = self._indices.get_input_deref_op()
        if indices_deref is None:
            raise ValueError("Indices iterator must support input dereference")

        values_deref = self._values.get_output_deref_op()
        if values_deref is None:
            return None

        values_advance = self._values.get_advance_op()
        indices_advance = self._indices.get_advance_op()

        values_iter = _make_iter_struct(self._values, values_advance, values_deref)
        indices_iter = _make_iter_struct(self._indices, indices_advance, indices_deref)
        _, deref = self._call_c_permutation(values_iter, indices_iter)
        return deref

    @property
    def children(self):
        return (self._values, self._indices)

    def __add__(self, offset: int) -> "PermutationIterator":
        """Advance the indices iterator by offset, keeping values at base."""
        return PermutationIterator(
            self._values,  # values stays at base for random access
            self._indices + offset,  # only indices advances  # type: ignore[operator]
        )

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("PermutationIterator", self._values.kind, self._indices.kind)
