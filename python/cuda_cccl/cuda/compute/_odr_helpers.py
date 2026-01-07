# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Backward compatibility re-exports from _numba._odr_helpers.

This module has been moved to _numba/_odr_helpers.py as part of the
layered architecture refactor. Import from _numba instead.
"""

# Re-export everything from the new location for backward compatibility
from ._numba._odr_helpers import (
    create_advance_void_ptr_wrapper,
    create_input_dereference_void_ptr_wrapper,
    create_op_void_ptr_wrapper,
    create_output_dereference_void_ptr_wrapper,
)

__all__ = [
    "create_op_void_ptr_wrapper",
    "create_advance_void_ptr_wrapper",
    "create_input_dereference_void_ptr_wrapper",
    "create_output_dereference_void_ptr_wrapper",
]
