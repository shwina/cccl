# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
New iterator implementations using C++ codegen.

This package provides iterator implementations that compile to LTOIR via
C++ code generation.
"""

from ._base import CompiledIterator, IteratorBase
from ._constant import ConstantIterator
from ._counting import CountingIterator
from ._discard import DiscardIterator
from ._permutation import PermutationIterator
from ._protocol import IteratorProtocol
from ._reverse import ReverseIterator
from ._transform import TransformIterator
from ._zip import ZipIterator

__all__ = [
    "CompiledIterator",
    "ConstantIterator",
    "CountingIterator",
    "DiscardIterator",
    "IteratorBase",
    "IteratorProtocol",
    "PermutationIterator",
    "ReverseIterator",
    "TransformIterator",
    "ZipIterator",
]
