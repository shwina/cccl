# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ConstantIterator implementation."""

from __future__ import annotations

from textwrap import dedent

import numpy as np

from .._cpp_codegen import cpp_type_from_descriptor
from ..types import from_numpy_dtype
from ._base import IteratorBase
from ._codegen_utils import ADVANCE_TEMPLATE, INPUT_DEREF_TEMPLATE, format_template


class ConstantIterator(IteratorBase):
    """
    Iterator representing a sequence of constant values.

    Every dereference returns the same constant value.
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

    def _generate_advance_source(self) -> tuple[str, str, list[bytes]]:
        symbol = self._make_advance_symbol()

        body = """(void)state;
(void)offset;"""

        source = format_template(ADVANCE_TEMPLATE, symbol=symbol, body=body)
        return (symbol, source, [])

    def _generate_input_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        symbol = self._make_input_deref_symbol()
        cpp_type = cpp_type_from_descriptor(self._value_type)

        body = dedent(f"""
            *static_cast<{cpp_type}*>(result) = *static_cast<{cpp_type}*>(state);
        """).strip()

        source = format_template(INPUT_DEREF_TEMPLATE, symbol=symbol, body=body)
        return (symbol, source, [])

    def _generate_output_deref_source(self) -> tuple[str, str, list[bytes]] | None:
        return None

    def __add__(self, offset: int) -> "ConstantIterator":
        """Return a new ConstantIterator (value doesn't change with position)."""
        return ConstantIterator(self._constant_value)
