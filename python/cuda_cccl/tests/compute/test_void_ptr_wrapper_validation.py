# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for _create_void_ptr_wrapper identifier validation.

These tests verify that invalid function names are rejected before
they can reach the exec() calls in _odr_helpers.py.
"""

import pytest
from numba import types

from cuda.compute._odr_helpers import _ArgMode, _ArgSpec, _create_void_ptr_wrapper


def _make_arg_specs():
    """One float32 input, one float32 output."""
    return [
        _ArgSpec(types.float32, _ArgMode.LOAD),
        _ArgSpec(types.float32, _ArgMode.STORE),
    ]


def _make_inner_sig():
    return types.float32(types.float32)


def _passthrough(x):
    return x


# ---------------------------------------------------------------------------
# Invalid function names
# ---------------------------------------------------------------------------


def test_newline_in_name_is_rejected():
    """Newlines are the primary exec() injection vector."""
    with pytest.raises(ValueError, match="must be a valid identifier"):
        _create_void_ptr_wrapper(
            _passthrough, "foo\nbar", _make_arg_specs(), _make_inner_sig()
        )


def test_space_in_name_is_rejected():
    with pytest.raises(ValueError, match="must be a valid identifier"):
        _create_void_ptr_wrapper(
            _passthrough, "my func", _make_arg_specs(), _make_inner_sig()
        )


def test_hyphen_in_name_is_rejected():
    with pytest.raises(ValueError, match="must be a valid identifier"):
        _create_void_ptr_wrapper(
            _passthrough, "my-func", _make_arg_specs(), _make_inner_sig()
        )


def test_semicolon_injection_in_name_is_rejected():
    with pytest.raises(ValueError, match="must be a valid identifier"):
        _create_void_ptr_wrapper(
            _passthrough, "f; import os", _make_arg_specs(), _make_inner_sig()
        )


def test_empty_name_is_rejected():
    with pytest.raises(ValueError, match="must be a valid identifier"):
        _create_void_ptr_wrapper(_passthrough, "", _make_arg_specs(), _make_inner_sig())


def test_numeric_name_is_rejected():
    with pytest.raises(ValueError, match="must be a valid identifier"):
        _create_void_ptr_wrapper(
            _passthrough, "123func", _make_arg_specs(), _make_inner_sig()
        )


# ---------------------------------------------------------------------------
# Valid function names
# ---------------------------------------------------------------------------


def test_simple_name_is_accepted():
    """Ordinary identifiers must not raise."""
    _create_void_ptr_wrapper(
        _passthrough, "my_op", _make_arg_specs(), _make_inner_sig()
    )


def test_underscore_prefix_name_is_accepted():
    _create_void_ptr_wrapper(_passthrough, "_op", _make_arg_specs(), _make_inner_sig())


def test_unicode_identifier_name_is_accepted():
    """Python allows unicode identifiers; they should be valid function names."""
    _create_void_ptr_wrapper(_passthrough, "αβ", _make_arg_specs(), _make_inner_sig())
