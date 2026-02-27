# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ZipIterator implementation."""

from __future__ import annotations

from .._bindings import Iterator, IteratorKind, Op, make_zip_iterator_ops
from ..types import struct
from ._base import IteratorBase, compose_iterator_states
from ._common import ensure_iterator


def _make_child_iter(it: IteratorBase, advance_op: Op, deref_op: Op) -> Iterator:
    """Build a Cython Iterator struct from individual Op objects."""
    return Iterator(
        it.state_alignment,
        IteratorKind.ITERATOR,
        advance_op,
        deref_op,
        it.value_type.info,
        state=it.state,
    )


class ZipIterator(IteratorBase):
    """
    Iterator that zips multiple iterators together.

    At each position, yields a tuple of values from all underlying iterators.

    Similar to `thrust::zip_iterator <https://nvidia.github.io/cccl/thrust/api/classthrust_1_1zip__iterator.html>`_.

    Example:
        The code snippet below demonstrates how to zip together an array and a
        :class:`CountingIterator <cuda.compute.iterators.CountingIterator>` to
        find the index of the maximum value of the array.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/zip_iterator_counting.py
            :language: python
            :start-after: # example-begin
    """

    __slots__ = [
        "_iterators",
        "_field_names",
        "_value_offsets",
        "_state_offsets",
    ]

    def __init__(self, *args):
        """
        Create a zip iterator.

        Args:
            *args: Iterators or arrays to zip together. Can be:
                   - Multiple iterators/arrays: ZipIterator(it1, it2, it3)
                   - A single sequence of iterators: ZipIterator([it1, it2, it3])
        """
        # Handle both ZipIterator(it1, it2) and ZipIterator([it1, it2])
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            iterators = args[0]
        else:
            iterators = args

        if len(iterators) < 1:
            raise ValueError("ZipIterator requires at least one iterator")

        # Wrap arrays in PointerIterator
        iterators = [ensure_iterator(it) for it in iterators]

        self._iterators = list(iterators)

        # Compose states from all iterators
        self._state_bytes, self._state_alignment, self._state_offsets = (
            compose_iterator_states(self._iterators)
        )

        # Build combined value type (struct layout)
        self._field_names = [f"field_{i}" for i in range(len(self._iterators))]
        fields = {
            name: it.value_type for name, it in zip(self._field_names, self._iterators)
        }
        self._value_type = struct(fields, name=f"Zip{len(iterators)}")
        self._value_offsets = [
            self._value_type.dtype.fields[name][1] for name in self._field_names
        ]

        super().__init__(
            state_bytes=self._state_bytes,
            state_alignment=self._state_alignment,
            value_type=self._value_type,
        )

    def _call_c_zip(self, child_iters: list) -> tuple:
        """Call make_zip_iterator_ops with the given child Iterators."""
        state_size = len(self._state_bytes)
        return make_zip_iterator_ops(
            child_iters,
            self._state_offsets,
            self._value_offsets,
            self._state_bytes,
            state_size,
            self._state_alignment,
            self._value_type.info,
        )

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that calls all child iterator advances."""
        child_iters = []
        for it in self._iterators:
            advance_op = it.get_advance_op()
            # Use input deref if available; fall back to output for advance generation
            deref_op = it.get_input_deref_op() or it.get_output_deref_op()
            if deref_op is None:
                raise ValueError(
                    "Each child iterator must support at least one dereference operation"
                )
            child_iters.append(_make_child_iter(it, advance_op, deref_op))
        advance, _ = self._call_c_zip(child_iters)
        return advance

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input deref that calls all child iterator input derefs."""
        child_iters = []
        for it in self._iterators:
            advance_op = it.get_advance_op()
            deref_op = it.get_input_deref_op()
            if deref_op is None:
                return None
            child_iters.append(_make_child_iter(it, advance_op, deref_op))
        _, deref = self._call_c_zip(child_iters)
        return deref

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output deref that calls all child iterator output derefs."""
        child_iters = []
        for it in self._iterators:
            advance_op = it.get_advance_op()
            deref_op = it.get_output_deref_op()
            if deref_op is None:
                return None
            child_iters.append(_make_child_iter(it, advance_op, deref_op))
        _, deref = self._call_c_zip(child_iters)
        return deref

    @property
    def children(self):
        return tuple(self._iterators)

    def __add__(self, offset: int) -> "ZipIterator":
        """Advance all child iterators by offset."""
        advanced_iterators = [it + offset for it in self._iterators]  # type: ignore[operator]
        return ZipIterator(*advanced_iterators)

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ZipIterator", tuple(it.kind for it in self._iterators))
