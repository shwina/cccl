# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
New iterator implementations using C++ codegen.

This package provides iterator implementations that compile to LTOIR via
C++ code generation, replacing the Numba-based implementations.
"""

from ._base import CompiledIterator, IteratorBase
from ._composite import ReverseIterator, TransformIterator
from ._protocol import IteratorProtocol
from ._simple import ConstantIterator, CountingIterator, DiscardIterator

__all__ = [
    "CompiledIterator",
    "ConstantIterator",
    "CountingIterator",
    "DiscardIterator",
    "IteratorBase",
    "IteratorProtocol",
    "ReverseIterator",
    "TransformIterator",
]
