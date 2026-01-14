# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ZipIterator implementation."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Sequence

from .._bindings import IteratorState
from .._codegen import compile_cpp_to_ltoir
from .._types import TypeDescriptor, custom_type

if TYPE_CHECKING:
    from ._protocol import IteratorProtocol


def _unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


class ZipIterator:
    """
    Iterator that zips multiple iterators together.

    At each position, yields a tuple/struct of values from all underlying iterators.
    """

    __slots__ = [
        "_iterators",
        "_uid",
        "_state_bytes",
        "_state_alignment",
        "_value_type",
        "_state_offsets",
        "_advance_result",
        "_input_deref_result",
        "_output_deref_result",
    ]

    def __init__(self, iterators: Sequence[IteratorProtocol]):
        """
        Create a zip iterator.

        Args:
            iterators: Sequence of iterators to zip together
        """
        if len(iterators) < 1:
            raise ValueError("ZipIterator requires at least one iterator")

        self._iterators = list(iterators)
        self._uid = _unique_suffix()

        # Build combined state
        state_parts = []
        offsets = []
        current_offset = 0
        max_alignment = 1

        for it in self._iterators:
            it_state = bytes(memoryview(it.state))
            it_align = it.state_alignment
            max_alignment = max(max_alignment, it_align)

            padding = (it_align - (current_offset % it_align)) % it_align
            if padding > 0:
                state_parts.append(b"\x00" * padding)
                current_offset += padding

            offsets.append(current_offset)
            state_parts.append(it_state)
            current_offset += len(it_state)

        self._state_bytes = b"".join(state_parts)
        self._state_alignment = max_alignment
        self._state_offsets = offsets

        # Build combined value type
        total_size = 0
        max_val_align = 1
        for it in self._iterators:
            vt = it.value_type
            padding = (vt.alignment - (total_size % vt.alignment)) % vt.alignment
            total_size += padding + vt.size
            max_val_align = max(max_val_align, vt.alignment)

        total_size = ((total_size + max_val_align - 1) // max_val_align) * max_val_align
        self._value_type = custom_type(
            total_size, max_val_align, f"zip{len(iterators)}"
        )

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None

    def _compile_if_needed(self) -> None:
        if self._advance_result is not None:
            return

        # Collect all iterator ops and their extras
        advance_info = []  # (name, ltoir, extras)
        input_deref_info = []  # (name, ltoir, extras) or None
        output_deref_info = []  # (name, ltoir, extras) or None

        for it in self._iterators:
            adv_name, adv_ltoir, adv_extras = it.get_advance_ltoir()
            advance_info.append((adv_name, adv_ltoir, adv_extras))

            in_result = it.get_input_dereference_ltoir()
            input_deref_info.append(in_result)

            out_result = it.get_output_dereference_ltoir()
            output_deref_info.append(out_result)

        # Compile advance
        advance_names = [info[0] for info in advance_info]
        symbol, ltoir = self._compile_advance(advance_names)
        advance_extras = []
        for info in advance_info:
            advance_extras.append(info[1])
            advance_extras.extend(info[2])
        self._advance_result = (symbol, ltoir, advance_extras)

        # Compile input dereference if all support it
        if all(info is not None for info in input_deref_info):
            # Filter to get only non-None values (mypy needs this)
            valid_input_info = [info for info in input_deref_info if info is not None]
            deref_names = [info[0] for info in valid_input_info]
            symbol, ltoir = self._compile_input_deref(deref_names)
            deref_extras: list[bytes] = []
            for info in valid_input_info:
                deref_extras.append(info[1])
                deref_extras.extend(info[2])
            self._input_deref_result = (symbol, ltoir, deref_extras)

        # Compile output dereference if all support it
        if all(info is not None for info in output_deref_info):
            valid_output_info = [info for info in output_deref_info if info is not None]
            deref_names = [info[0] for info in valid_output_info]
            symbol, ltoir = self._compile_output_deref(deref_names)
            deref_extras_out: list[bytes] = []
            for info in valid_output_info:
                deref_extras_out.append(info[1])
                deref_extras_out.extend(info[2])
            self._output_deref_result = (symbol, ltoir, deref_extras_out)

    def _compile_advance(self, advance_names: list[str]) -> tuple[str, bytes]:
        symbol = f"zip_advance_{self._uid}"

        externs = "\n".join(
            f'extern "C" __device__ void {name}(void*, void*);'
            for name in advance_names
        )

        calls = "\n    ".join(
            f"{name}(static_cast<char*>(state) + {offset}, offset);"
            for name, offset in zip(advance_names, self._state_offsets)
        )

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

{externs}

extern "C" __device__ void {symbol}(void* state, void* offset) {{
    {calls}
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_input_deref(self, deref_names: list[str]) -> tuple[str, bytes]:
        symbol = f"zip_input_deref_{self._uid}"

        externs = "\n".join(
            f'extern "C" __device__ void {name}(void*, void*);' for name in deref_names
        )

        # Calculate value offsets
        value_offsets = []
        current = 0
        for it in self._iterators:
            vt = it.value_type
            padding = (vt.alignment - (current % vt.alignment)) % vt.alignment
            current += padding
            value_offsets.append(current)
            current += vt.size

        calls = "\n    ".join(
            f"{name}(static_cast<char*>(state) + {state_off}, "
            f"static_cast<char*>(result) + {val_off});"
            for name, state_off, val_off in zip(
                deref_names, self._state_offsets, value_offsets
            )
        )

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

{externs}

extern "C" __device__ void {symbol}(void* state, void* result) {{
    {calls}
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    def _compile_output_deref(self, deref_names: list[str]) -> tuple[str, bytes]:
        symbol = f"zip_output_deref_{self._uid}"

        externs = "\n".join(
            f'extern "C" __device__ void {name}(void*, void*);' for name in deref_names
        )

        value_offsets = []
        current = 0
        for it in self._iterators:
            vt = it.value_type
            padding = (vt.alignment - (current % vt.alignment)) % vt.alignment
            current += padding
            value_offsets.append(current)
            current += vt.size

        calls = "\n    ".join(
            f"{name}(static_cast<char*>(state) + {state_off}, "
            f"static_cast<char*>(value) + {val_off});"
            for name, state_off, val_off in zip(
                deref_names, self._state_offsets, value_offsets
            )
        )

        source = f"""
#include <cuda/std/cstdint>
using namespace cuda::std;

{externs}

extern "C" __device__ void {symbol}(void* state, void* value) {{
    {calls}
}}
"""
        ltoir = compile_cpp_to_ltoir(source, (symbol,))
        return (symbol, ltoir)

    @property
    def state(self) -> IteratorState:
        return IteratorState(self._state_bytes)

    @property
    def state_alignment(self) -> int:
        return self._state_alignment

    @property
    def value_type(self) -> TypeDescriptor:
        return self._value_type

    def get_advance_ltoir(self) -> tuple[str, bytes, list[bytes]]:
        self._compile_if_needed()
        assert self._advance_result is not None
        return self._advance_result

    def get_input_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        self._compile_if_needed()
        return self._input_deref_result

    def get_output_dereference_ltoir(self) -> tuple[str, bytes, list[bytes]] | None:
        self._compile_if_needed()
        return self._output_deref_result

    @property
    def is_input_iterator(self) -> bool:
        return all(it.is_input_iterator for it in self._iterators)

    @property
    def is_output_iterator(self) -> bool:
        return all(it.is_output_iterator for it in self._iterators)

    def to_cccl_iter(self, is_output: bool = False):
        """Convert to CCCL Iterator for algorithm interop."""
        from .._bindings import Iterator, IteratorKind, Op, OpKind

        adv_name, adv_ltoir, adv_extras = self.get_advance_ltoir()
        advance_op = Op(
            operator_type=OpKind.STATELESS,
            name=adv_name,
            ltoir=adv_ltoir,
            extra_ltoirs=adv_extras if adv_extras else None,
        )

        if is_output:
            deref_result = self.get_output_dereference_ltoir()
            if deref_result is None:
                raise ValueError("This iterator does not support output operations")
        else:
            deref_result = self.get_input_dereference_ltoir()
            if deref_result is None:
                raise ValueError("This iterator does not support input operations")

        deref_name, deref_ltoir, deref_extras = deref_result
        deref_op = Op(
            operator_type=OpKind.STATELESS,
            name=deref_name,
            ltoir=deref_ltoir,
            extra_ltoirs=deref_extras if deref_extras else None,
        )

        return Iterator(
            self.state_alignment,
            IteratorKind.ITERATOR,
            advance_op,
            deref_op,
            self._value_type.to_type_info(),
            state=self.state,
        )

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ZipIterator", tuple(it.kind for it in self._iterators))
