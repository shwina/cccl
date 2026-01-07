# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Numba compiler layer for cuda.compute.

This module provides Numba-based compilation of Python callables and iterators
to the Raw LTOIR layer. It can be swapped out for other compilers (MLIR, Triton, etc.)
that produce compatible LTOIR.

The main entry points are:
    - compile_op(): Compile a Python callable to RawOp
    - compile_iterator_advance(): Compile iterator advance method to LTOIR
    - compile_iterator_input_dereference(): Compile input dereference to LTOIR
    - compile_iterator_output_dereference(): Compile output dereference to LTOIR
"""

from ._compiler import (
    compile_iterator_advance,
    compile_iterator_input_dereference,
    compile_iterator_output_dereference,
    compile_op,
    numba_type_to_type_info,
)
from ._odr_helpers import (
    create_advance_void_ptr_wrapper,
    create_input_dereference_void_ptr_wrapper,
    create_op_void_ptr_wrapper,
    create_output_dereference_void_ptr_wrapper,
)
from ._utils import (
    get_inferred_return_type,
    signature_from_annotations,
    to_numba_type,
)

__all__ = [
    # Compiler functions (main entry points)
    "compile_op",
    "compile_iterator_advance",
    "compile_iterator_input_dereference",
    "compile_iterator_output_dereference",
    "numba_type_to_type_info",
    # ODR helpers (lower-level)
    "create_op_void_ptr_wrapper",
    "create_advance_void_ptr_wrapper",
    "create_input_dereference_void_ptr_wrapper",
    "create_output_dereference_void_ptr_wrapper",
    # Utils
    "get_inferred_return_type",
    "signature_from_annotations",
    "to_numba_type",
]
