# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Backward compatibility re-exports from _numba._utils.

This module has been moved to _numba/_utils.py as part of the
layered architecture refactor. Import from _numba instead.
"""

# Re-export everything from the new location for backward compatibility
from ._numba._utils import (
    get_inferred_return_type,
    signature_from_annotations,
    to_numba_type,
)

__all__ = [
    "get_inferred_return_type",
    "signature_from_annotations",
    "to_numba_type",
]
