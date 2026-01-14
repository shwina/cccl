# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Iterator protocol for cuda.compute.

Defines the interface that all iterators must implement to work with
cuda.compute algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .._bindings import IteratorState
    from .._types import TypeDescriptor


@runtime_checkable
class IteratorProtocol(Protocol):
    """
    Protocol defining the interface for iterators in cuda.compute.

    All iterators must implement this protocol to be usable with algorithms
    like reduce_into, scan, etc.
    """

    @property
    def state(self) -> IteratorState:
        """Return the iterator state for CCCL interop."""
        ...

    @property
    def state_alignment(self) -> int:
        """Return the alignment of the iterator state."""
        ...

    @property
    def value_type(self) -> TypeDescriptor:
        """Return the TypeDescriptor for dereferenced values."""
        ...

    def get_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        """
        Get the LTOIR for the advance operation.

        Returns:
            Tuple of (symbol_name, ltoir_bytes, extra_ltoirs)
        """
        ...

    def get_input_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """
        Get the LTOIR for input dereference operation.

        Returns:
            Tuple of (symbol_name, ltoir_bytes, extra_ltoirs), or None if not an input iterator
        """
        ...

    def get_output_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        """
        Get the LTOIR for output dereference operation.

        Returns:
            Tuple of (symbol_name, ltoir_bytes, extra_ltoirs), or None if not an output iterator
        """
        ...

    @property
    def is_input_iterator(self) -> bool:
        """Return True if this iterator supports input dereference."""
        ...

    @property
    def is_output_iterator(self) -> bool:
        """Return True if this iterator supports output dereference."""
        ...
