# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
RawIterator: Pre-compiled LTOIR iterator for the Raw LTOIR layer.

This module provides RawIterator, which represents a pre-compiled LTOIR iterator
that can be used directly with cuda.compute algorithms, bypassing Numba compilation.

Example:
    >>> from cuda.compute.raw import RawIterator
    >>> from cuda.compute.raw.types import int32
    >>>
    >>> # Create from pre-compiled LTOIR
    >>> my_iter = RawIterator(
    ...     state=my_state_bytes,
    ...     state_alignment=8,
    ...     value_type=int32,
    ...     advance=("my_advance", advance_ltoir),
    ...     input_dereference=("my_deref", deref_ltoir),
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Hashable

from .._bindings import Iterator, IteratorKind, IteratorState, Op, OpKind, TypeInfo

if TYPE_CHECKING:
    from typing_extensions import Buffer

__all__ = ["RawIterator"]


class RawIterator:
    """A pre-compiled LTOIR iterator, ready for use with cuda.compute algorithms.

    RawIterator wraps pre-compiled LTOIR bytecode for iterator operations
    (advance and dereference), allowing users to bring their own compiler
    output to cuda.compute.

    The LTOIR must define functions with the following ABIs:

    Advance function:
        extern "C" void advance_name(void* state_ptr, uint64_t offset);

    Input dereference function:
        extern "C" void input_deref_name(void* state_ptr, void* result_ptr);

    Output dereference function:
        extern "C" void output_deref_name(void* state_ptr, void* value_ptr);

    Attributes:
        state: The iterator state bytes.
        state_alignment: Alignment requirement for the state.
        value_type: TypeInfo for the dereferenced value type.
        is_input_iterator: Whether this iterator supports input dereference.
        is_output_iterator: Whether this iterator supports output dereference.
    """

    __slots__ = [
        "_state",
        "_state_alignment",
        "_value_type",
        "_advance",
        "_input_dereference",
        "_output_dereference",
        "_host_advance",
    ]

    def __init__(
        self,
        state: bytes | memoryview | "Buffer",
        state_alignment: int,
        value_type: TypeInfo,
        advance: tuple[str, bytes],
        input_dereference: tuple[str, bytes] | None = None,
        output_dereference: tuple[str, bytes] | None = None,
        host_advance: Callable | None = None,
    ):
        """Create a RawIterator from pre-compiled LTOIR.

        Args:
            state: The iterator state as bytes or memoryview.
            state_alignment: Alignment requirement for the state in bytes.
            value_type: TypeInfo for the dereferenced value type.
            advance: Tuple of (abi_name, ltoir_bytes) for the advance function.
            input_dereference: Optional tuple of (abi_name, ltoir_bytes) for
                input dereference. Required for input iterators.
            output_dereference: Optional tuple of (abi_name, ltoir_bytes) for
                output dereference. Required for output iterators.
            host_advance: Optional host-side advance function for algorithms
                that need to advance iterators on the host.

        Raises:
            ValueError: If neither input_dereference nor output_dereference is provided,
                or if advance tuple is invalid.
        """
        if input_dereference is None and output_dereference is None:
            raise ValueError(
                "At least one of input_dereference or output_dereference must be provided"
            )

        self._validate_ltoir_tuple(advance, "advance")
        if input_dereference is not None:
            self._validate_ltoir_tuple(input_dereference, "input_dereference")
        if output_dereference is not None:
            self._validate_ltoir_tuple(output_dereference, "output_dereference")

        self._state = state
        self._state_alignment = state_alignment
        self._value_type = value_type
        self._advance = advance
        self._input_dereference = input_dereference
        self._output_dereference = output_dereference
        self._host_advance = host_advance

    @staticmethod
    def _validate_ltoir_tuple(t: tuple, name: str) -> None:
        """Validate an LTOIR tuple (name, bytes)."""
        if not isinstance(t, tuple) or len(t) != 2:
            raise ValueError(f"{name} must be a tuple of (abi_name, ltoir_bytes)")
        abi_name, ltoir = t
        if not isinstance(abi_name, str) or len(abi_name) == 0:
            raise ValueError(f"{name}[0] must be a non-empty string (ABI name)")
        if not isinstance(ltoir, bytes) or len(ltoir) == 0:
            raise ValueError(f"{name}[1] must be non-empty bytes (LTOIR)")

    @property
    def state(self) -> bytes | memoryview | "Buffer":
        """The iterator state."""
        return self._state

    @property
    def state_alignment(self) -> int:
        """Alignment requirement for the state in bytes."""
        return self._state_alignment

    @property
    def value_type(self) -> TypeInfo:
        """TypeInfo for the dereferenced value type."""
        return self._value_type

    @property
    def is_input_iterator(self) -> bool:
        """Whether this iterator supports input dereference."""
        return self._input_dereference is not None

    @property
    def is_output_iterator(self) -> bool:
        """Whether this iterator supports output dereference."""
        return self._output_dereference is not None

    @property
    def host_advance(self) -> Callable | None:
        """Host-side advance function, if provided."""
        return self._host_advance

    def get_advance_ltoir(self) -> tuple[str, bytes]:
        """Return the advance LTOIR tuple (abi_name, ltoir_bytes)."""
        return self._advance

    def get_input_dereference_ltoir(self) -> tuple[str, bytes]:
        """Return the input dereference LTOIR tuple (abi_name, ltoir_bytes).

        Raises:
            AttributeError: If this is not an input iterator.
        """
        if self._input_dereference is None:
            raise AttributeError("This iterator is not an input iterator")
        return self._input_dereference

    def get_output_dereference_ltoir(self) -> tuple[str, bytes]:
        """Return the output dereference LTOIR tuple (abi_name, ltoir_bytes).

        Raises:
            AttributeError: If this is not an output iterator.
        """
        if self._output_dereference is None:
            raise AttributeError("This iterator is not an output iterator")
        return self._output_dereference

    def to_cccl_iterator(self, io_kind: str) -> Iterator:
        """Convert to low-level Iterator for C library interop.

        Args:
            io_kind: Either "input" or "output" to select which dereference to use.

        Returns:
            Iterator object suitable for passing to the C bindings.

        Raises:
            ValueError: If io_kind is invalid.
            AttributeError: If the requested dereference is not available.
        """
        advance_op = Op(
            operator_type=OpKind.STATELESS,
            name=self._advance[0],
            ltoir=self._advance[1],
        )

        if io_kind == "input":
            if self._input_dereference is None:
                raise AttributeError("This iterator is not an input iterator")
            deref = self._input_dereference
        elif io_kind == "output":
            if self._output_dereference is None:
                raise AttributeError("This iterator is not an output iterator")
            deref = self._output_dereference
        else:
            raise ValueError(f"io_kind must be 'input' or 'output', got {io_kind!r}")

        deref_op = Op(
            operator_type=OpKind.STATELESS,
            name=deref[0],
            ltoir=deref[1],
        )

        # Wrap state bytes in IteratorState
        state_obj = IteratorState(self._state)

        return Iterator(
            alignment=self._state_alignment,
            iterator_type=IteratorKind.ITERATOR,
            advance_fn=advance_op,
            dereference_fn=deref_op,
            value_type=self._value_type,
            state=state_obj,
            host_advance_fn=self._host_advance,
        )

    def get_cache_key(self) -> Hashable:
        """Return a hashable cache key for this iterator.

        The cache key is based on the function names and LTOIR content hashes.
        """
        input_key = (
            (self._input_dereference[0], hash(self._input_dereference[1]))
            if self._input_dereference
            else None
        )
        output_key = (
            (self._output_dereference[0], hash(self._output_dereference[1]))
            if self._output_dereference
            else None
        )
        return (
            self._advance[0],
            hash(self._advance[1]),
            input_key,
            output_key,
            self._value_type.size,
            self._value_type.alignment,
        )

    def __repr__(self) -> str:
        parts = [f"RawIterator(value_type={self._value_type}"]
        if self.is_input_iterator:
            parts.append("input=True")
        if self.is_output_iterator:
            parts.append("output=True")
        parts.append(")")
        return ", ".join(parts[:-1]) + parts[-1]
