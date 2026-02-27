# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ReverseIterator implementation."""

from __future__ import annotations

from .._bindings import Iterator, IteratorKind, Op, make_reverse_iterator_ops
from .._utils.protocols import get_size, is_device_array
from ._base import IteratorBase
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


class ReverseIterator(IteratorBase):
    """
    Iterator that reverses the direction of an underlying iterator.

    Advance with positive offset moves backward in the underlying iterator.
    """

    __slots__ = [
        "_underlying",
    ]

    def __init__(self, underlying):
        """
        Create a reverse iterator.

        Args:
            underlying: The underlying iterator or array to reverse
        """

        if is_device_array(underlying):
            # TODO: this is probably incorrect behaviour. In C++, initializing
            # with a pointer to the end of the array is left to be done explicitly
            # by the user.
            self._underlying = ensure_iterator(underlying) + (get_size(underlying) - 1)
        else:
            self._underlying = ensure_iterator(underlying)

        super().__init__(
            state_bytes=bytes(self._underlying.state),
            state_alignment=self._underlying.state_alignment,
            value_type=self._underlying.value_type,
        )

    def _make_advance_op(self) -> Op:
        """Provide Op for advance that negates offset direction."""
        advance_op = self._underlying.get_advance_op()
        # Use input deref if available; fall back to output deref for advance generation
        deref_op = (
            self._underlying.get_input_deref_op()
            or self._underlying.get_output_deref_op()
        )
        if deref_op is None:
            raise ValueError(
                "Underlying iterator must support at least one dereference operation"
            )
        underlying_iter = _make_iter_struct(self._underlying, advance_op, deref_op)
        advance, _ = make_reverse_iterator_ops(underlying_iter)
        return advance

    def _make_input_deref_op(self) -> Op | None:
        """Provide Op for input dereference that delegates to underlying."""
        advance_op = self._underlying.get_advance_op()
        deref_op = self._underlying.get_input_deref_op()
        if deref_op is None:
            return None
        underlying_iter = _make_iter_struct(self._underlying, advance_op, deref_op)
        _, deref = make_reverse_iterator_ops(underlying_iter)
        return deref

    def _make_output_deref_op(self) -> Op | None:
        """Provide Op for output dereference that delegates to underlying."""
        advance_op = self._underlying.get_advance_op()
        deref_op = self._underlying.get_output_deref_op()
        if deref_op is None:
            return None
        underlying_iter = _make_iter_struct(self._underlying, advance_op, deref_op)
        _, deref = make_reverse_iterator_ops(underlying_iter)
        return deref

    @property
    def children(self):
        return (self._underlying,)

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ReverseIterator", self._underlying.kind)

    def __add__(self, offset: int):
        return ReverseIterator(self._underlying + offset)
