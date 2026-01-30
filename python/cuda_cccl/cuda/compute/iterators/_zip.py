# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ZipIterator implementation."""

from __future__ import annotations

from .._bindings import IteratorState
from .._cpp_codegen import compile_cpp_to_ltoir
from ..types import TypeDescriptor, struct
from ._base import IteratorBase, _deterministic_suffix


def _ensure_iterator(obj):
    """Wrap array in PointerIterator if needed."""
    from ._pointer import PointerIterator

    if isinstance(obj, IteratorBase):
        return obj
    if hasattr(obj, "__cuda_array_interface__"):
        return PointerIterator(obj)
    raise TypeError("ZipIterator requires iterators or device arrays")


class ZipIterator(IteratorBase):
    """
    Iterator that zips multiple iterators together.

    At each position, yields a tuple/struct of values from all underlying iterators.
    """

    __slots__ = [
        "_iterators",
        "_uid",
        # TypeDescriptors for each component (for Numba tuple)
        "_component_types",
        "_field_names",
        "_value_offsets",
        "_state_offsets",
        "_advance_result",
        "_input_deref_result",
        "_output_deref_result",
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
        iterators = [_ensure_iterator(it) for it in iterators]

        self._iterators = list(iterators)

        # Build combined state
        state_parts = []
        offsets = []
        current_offset = 0
        max_alignment = 1

        for it in self._iterators:
            it_state = bytes(it.state)
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

        # Build combined value type (struct layout)
        self._field_names = [f"field_{i}" for i in range(len(self._iterators))]
        fields = {
            name: it.value_type for name, it in zip(self._field_names, self._iterators)
        }
        self._value_type = struct(fields, name=f"Zip{len(iterators)}")
        self._value_offsets = [
            self._value_type.dtype.fields[name][1] for name in self._field_names
        ]
        # Store component types for Numba Tuple conversion
        self._component_types = tuple(it.value_type for it in self._iterators)

        self._advance_result: tuple[str, bytes, list[bytes]] | None = None
        self._input_deref_result: tuple[str, bytes, list[bytes]] | None = None
        self._output_deref_result: tuple[str, bytes, list[bytes]] | None = None

        super().__init__(
            state_bytes=self._state_bytes,
            state_alignment=self._state_alignment,
            value_type=self._value_type,
        )

        # Generate deterministic suffix after super().__init__() so self.kind is available
        self._uid = _deterministic_suffix(self.kind)

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
#include <cuda_fp16.h>
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

        calls = "\n    ".join(
            f"{name}(static_cast<char*>(state) + {state_off}, "
            f"static_cast<char*>(result) + {val_off});"
            for name, state_off, val_off in zip(
                deref_names, self._state_offsets, self._value_offsets
            )
        )

        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
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

        calls = "\n    ".join(
            f"{name}(static_cast<char*>(state) + {state_off}, "
            f"static_cast<char*>(value) + {val_off});"
            for name, state_off, val_off in zip(
                deref_names, self._state_offsets, self._value_offsets
            )
        )

        source = f"""
#include <cuda/std/cstdint>
#include <cuda_fp16.h>
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

    @property
    def children(self):
        return tuple(self._iterators)

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

    def __add__(self, offset: int) -> "ZipIterator":
        """Advance all child iterators by offset."""
        advanced_iterators = [it + offset for it in self._iterators]  # type: ignore[operator]
        return ZipIterator(*advanced_iterators)

    @property
    def kind(self):
        """Return a hashable kind for caching purposes."""
        return ("ZipIterator", tuple(it.kind for it in self._iterators))
