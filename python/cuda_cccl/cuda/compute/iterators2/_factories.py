# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Factory functions for iterators.

These provide the user-facing API, accepting either Python callables or
CompiledOp and converting appropriately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

import numpy as np

from .._types import TypeDescriptor, from_numpy_dtype
from .._utils.protocols import get_dtype
from ..op import CompiledOp, make_op_adapter

if TYPE_CHECKING:
    from ._protocol import IteratorProtocol


def _is_iterator(obj) -> bool:
    """Check if an object is an iterator (has to_cccl_iter method)."""
    return hasattr(obj, "to_cccl_iter") and callable(obj.to_cccl_iter)


def _get_value_type(underlying) -> TypeDescriptor:
    """Get value type from an iterator or array."""
    if _is_iterator(underlying):
        return underlying.value_type
    # It's an array - get dtype
    dtype = get_dtype(underlying)
    return from_numpy_dtype(dtype)


def _ensure_input_iterator(underlying) -> "IteratorProtocol":
    """Ensure underlying is an input iterator, wrapping arrays if needed."""
    if _is_iterator(underlying):
        return underlying
    # Wrap array in CacheModifiedInputIterator (default behavior for input)
    from ._cache_modified import CacheModifiedInputIterator

    return CacheModifiedInputIterator(underlying, modifier="stream")


def _ensure_output_iterator(underlying) -> "IteratorProtocol":
    """Ensure underlying is an output iterator, wrapping arrays if needed."""
    if _is_iterator(underlying):
        return underlying
    # Wrap array in PointerIterator (supports both input and output)
    from ._pointer import PointerIterator

    return PointerIterator(underlying)


def _infer_output_type(op_adapter, input_type: TypeDescriptor) -> TypeDescriptor:
    """Infer output type from an op adapter for a given input type."""
    # For CompiledOp, we can't infer - assume same type
    if isinstance(op_adapter, CompiledOp):
        return input_type

    # For JIT ops, use Numba type inference
    if hasattr(op_adapter, "func") and op_adapter.func is not None:
        from .._jit import _to_internal_type, get_type_descriptor

        internal_input = _to_internal_type(input_type)
        from ..numba_utils import get_inferred_return_type

        internal_output = get_inferred_return_type(op_adapter.func, (internal_input,))
        return get_type_descriptor(internal_output)

    # Fallback: assume same type
    return input_type


def TransformIterator(
    underlying: Union["IteratorProtocol", "np.ndarray"],
    op: Callable | CompiledOp,
):
    """
    Create an input transform iterator.

    Applies the unary transform operation to values read from the underlying
    iterator or array.

    Args:
        underlying: The underlying iterator or device array
        op: Unary transform operation (Python callable or CompiledOp)

    Returns:
        TransformIterator configured for input
    """
    from ._transform import TransformIterator as _TransformIterator

    # Get input type from underlying (works for both iterators and arrays)
    input_type = _get_value_type(underlying)

    # Ensure underlying is an input iterator
    underlying_iter = _ensure_input_iterator(underlying)

    # Create op adapter (handles callable -> _JitOp, CompiledOp passthrough)
    op_adapter = make_op_adapter(op)

    # Infer output type
    output_type = _infer_output_type(op_adapter, input_type)

    return _TransformIterator(
        underlying=underlying_iter,
        transform_op=op_adapter,
        output_value_type=output_type,
        is_input=True,
    )


def TransformOutputIterator(
    underlying: Union["IteratorProtocol", "np.ndarray"],
    op: Callable | CompiledOp,
):
    """
    Create an output transform iterator.

    Applies the unary transform operation to values before writing to the
    underlying iterator or array.

    Args:
        underlying: The underlying iterator or device array to write to
        op: Unary transform operation (Python callable or CompiledOp)

    Returns:
        TransformIterator configured for output
    """
    from ._transform import TransformIterator as _TransformIterator

    # Get underlying type (works for both iterators and arrays)
    underlying_type = _get_value_type(underlying)

    # Ensure underlying is an output iterator
    underlying_iter = _ensure_output_iterator(underlying)

    # Create op adapter
    op_adapter = make_op_adapter(op)

    # For output iterator, the value_type is what the algorithm writes,
    # which then gets transformed and written to the underlying iterator.
    # We use the underlying type as the visible value_type since the
    # transform should produce the same type.
    return _TransformIterator(
        underlying=underlying_iter,
        transform_op=op_adapter,
        output_value_type=underlying_type,
        is_input=False,
    )
