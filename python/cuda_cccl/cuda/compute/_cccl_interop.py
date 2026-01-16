# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

import functools
import os
import subprocess
import tempfile
import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Protocol, TypeGuard

import numba
import numpy as np
from numba import cuda, types
from numba.core.extending import as_numba_type

# TODO: adding a type-ignore here because `cuda` being a
# namespace package confuses mypy when `cuda.<something_else>`
# is installed, but not `cuda.cccl`. For namespace packages,
# it appears we need to actually install the sub-package
# in order for mypy to find its py.typed file. However, CI
# does type checking of `cuda.cccl` without actually installing
# it.
#
# We need to find a better solution for this.
from cuda.cccl import get_include_paths  # type: ignore

from ._bindings import (
    CommonData,
    Iterator,
    IteratorKind,
    IteratorState,
    Op,
    Pointer,
    TypeEnum,
    TypeInfo,
    Value,
    make_pointer_object,
)
from ._utils.protocols import get_data_pointer, get_dtype, is_contiguous
from .op import OpKind
from .typing import DeviceArrayLike, GpuStruct

_TYPE_TO_ENUM = {
    types.int8: TypeEnum.INT8,
    types.int16: TypeEnum.INT16,
    types.int32: TypeEnum.INT32,
    types.int64: TypeEnum.INT64,
    types.uint8: TypeEnum.UINT8,
    types.uint16: TypeEnum.UINT16,
    types.uint32: TypeEnum.UINT32,
    types.uint64: TypeEnum.UINT64,
    types.float16: TypeEnum.FLOAT16,
    types.float32: TypeEnum.FLOAT32,
    types.float64: TypeEnum.FLOAT64,
}


if TYPE_CHECKING:
    from numba.core.typing import Signature

    class IteratorProtocol(Protocol):
        kind: object

        def to_cccl_iter(self, *, is_output: bool) -> Any: ...


def _type_to_enum(numba_type: types.Type) -> TypeEnum:
    if numba_type in _TYPE_TO_ENUM:
        return _TYPE_TO_ENUM[numba_type]
    return TypeEnum.STORAGE


@functools.lru_cache(maxsize=1)
def _legacy_iterators_available() -> bool:
    import importlib.util

    # If the new iterator implementation exists, prefer it.
    if importlib.util.find_spec("cuda.compute.iterators._base") is not None:
        return False
    # Legacy iterators live in _iterators.py
    return importlib.util.find_spec("cuda.compute.iterators._iterators") is not None


# TODO: replace with functools.cache once our docs build environment
# is upgraded to at least Python 3.9
@functools.lru_cache(maxsize=None)
def _numba_type_to_info(numba_type: types.Type) -> TypeInfo:
    context = cuda.descriptor.cuda_target.target_context
    value_type = context.get_value_type(numba_type)
    if isinstance(numba_type, types.Record):
        # then `value_type` is a pointer and we need the
        # alignment of the pointee.
        value_type = value_type.pointee
    size = value_type.get_abi_size(context.target_data)
    alignment = value_type.get_abi_alignment(context.target_data)
    return TypeInfo(size, alignment, _type_to_enum(numba_type))


@functools.lru_cache(maxsize=None)
def _numpy_type_to_info(numpy_type: np.dtype) -> TypeInfo:
    numba_type = numba.from_dtype(numpy_type)
    return _numba_type_to_info(numba_type)


def _device_array_to_cccl_iter(array: DeviceArrayLike) -> Iterator:
    if not is_contiguous(array):
        raise ValueError("Non-contiguous arrays are not supported.")
    dtype = get_dtype(array)

    # Handle structured dtypes by creating a proper gpu_struct
    if dtype.type == np.void:
        from .struct import gpu_struct

        numba_type = as_numba_type(gpu_struct(dtype))
        info = _numba_type_to_info(numba_type)
    else:
        info = _numpy_type_to_info(dtype)

    state_info = _numpy_type_to_info(np.intp)
    return Iterator(
        state_info.alignment,
        IteratorKind.POINTER,
        Op(),
        Op(),
        info,
        # Note: this is slightly slower, but supports all ndarray-like objects
        # as long as they support CAI
        # TODO: switch to use gpumemoryview once it's ready
        state=get_data_pointer(array),
    )


def _none_to_cccl_iter() -> Iterator:
    # Any type could be used here, we just need to pass NULL.
    info = _numpy_type_to_info(np.uint8)
    return Iterator(info.alignment, IteratorKind.POINTER, Op(), Op(), info, state=None)


def type_enum_as_name(enum_value: int) -> str:
    return (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "STORAGE",
    )[enum_value]


def is_iterator(obj: object) -> TypeGuard["IteratorProtocol"]:
    """Check if an object is an iterator (protocol or IteratorBase)."""
    if hasattr(obj, "to_cccl_iter") and callable(obj.to_cccl_iter):
        return True
    try:
        from .iterators._iterators import IteratorBase
    except Exception:
        return False
    return isinstance(obj, IteratorBase)


def _iterator_to_cccl_iter(it, is_output: bool) -> Iterator:
    context = cuda.descriptor.cuda_target.target_context
    state_ptr_type = it.state_ptr_type
    state_type = it.state_type
    size = context.get_value_type(state_type).get_abi_size(context.target_data)
    iterator_state = memoryview(it.state)
    if not iterator_state.nbytes == size:
        raise ValueError(
            f"Iterator state size, {iterator_state.nbytes} bytes, for iterator type {type(it)} "
            f"does not match size of numba type, {size} bytes"
        )
    alignment = context.get_value_type(state_ptr_type).get_abi_alignment(
        context.target_data
    )

    advance_abi_name, advance_ltoir = it.get_advance_ltoir()
    if is_output:
        deref_abi_name, deref_ltoir = it.get_output_dereference_ltoir()
    else:
        deref_abi_name, deref_ltoir = it.get_input_dereference_ltoir()

    advance_op = Op(
        operator_type=OpKind.STATELESS,
        name=advance_abi_name,
        ltoir=advance_ltoir,
    )
    deref_op = Op(
        operator_type=OpKind.STATELESS,
        name=deref_abi_name,
        ltoir=deref_ltoir,
    )
    return Iterator(
        alignment,
        IteratorKind.ITERATOR,
        advance_op,
        deref_op,
        _numba_type_to_info(it.value_type),
        state=it.state,
    )


def get_iterator_kind(it):
    """Get the cache key kind from an iterator."""
    return it.kind


def _to_cccl_iter(it, is_output: bool) -> Iterator:
    """Convert an iterator or array to a CCCL Iterator."""
    if it is None:
        return _none_to_cccl_iter()
    # Iterator protocol / IteratorBase
    if is_iterator(it):
        if hasattr(it, "to_cccl_iter") and callable(it.to_cccl_iter):
            return it.to_cccl_iter(is_output=is_output)
        return _iterator_to_cccl_iter(it, is_output=is_output)
    # Device array
    return _device_array_to_cccl_iter(it)


def to_cccl_input_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, is_output=False)


def to_cccl_output_iter(array_or_iterator) -> Iterator:
    return _to_cccl_iter(array_or_iterator, is_output=True)


def to_cccl_value_state(array_or_struct: np.ndarray | GpuStruct) -> memoryview:
    if isinstance(array_or_struct, np.ndarray):
        assert array_or_struct.flags.contiguous
        data = array_or_struct.data.cast("B")
        return data
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value_state(array_or_struct._data)


def to_cccl_value(array_or_struct: np.ndarray | GpuStruct) -> Value:
    if isinstance(array_or_struct, np.ndarray):
        info = _numpy_type_to_info(array_or_struct.dtype)
        return Value(info, array_or_struct.data.cast("B"))
    else:
        # it's a GpuStruct, use the array underlying it
        return to_cccl_value(array_or_struct._data)


def to_stateless_cccl_op(op, sig: "Signature") -> Op:
    from ._odr_helpers import create_op_void_ptr_wrapper

    wrapped_op, wrapper_sig = create_op_void_ptr_wrapper(op, sig)

    ltoir, _ = cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")
    return Op(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
    )


class ZipValueType:
    """
    Special type for ZipIterator values that includes component types.

    This enables Numba to create a Tuple type for user-defined operations
    that access individual elements via indexing (pair[0], pair[1]).
    """

    def __init__(self, value_type, component_types):
        self.value_type = value_type
        self.component_types = component_types


def get_value_type(d_in):
    """
    Get the value type for an input array, iterator, or struct.

    Returns:
        - For ZipIterator: ZipValueType (enables Numba Tuple conversion)
        - For other iterators (with to_cccl_iter): their value_type (TypeDescriptor)
        - For _Struct instances: the struct class itself
        - For arrays with structured dtype: an anonymous gpu_struct class
        - For arrays with scalar dtype: a TypeDescriptor
    """
    from ._types import from_numpy_dtype
    from .struct import _Struct, gpu_struct

    # Iterator protocol - returns TypeDescriptor or ZipValueType
    if is_iterator(d_in):
        if hasattr(d_in, "value_type"):
            return d_in.value_type
        # ZipIterator returns ZipValueType for Numba Tuple conversion
        if hasattr(d_in, "_component_types") and d_in._component_types is not None:
            return ZipValueType(d_in.value_type, d_in._component_types)
        return d_in.value_type

    # Struct instances - return the class
    if isinstance(d_in, _Struct):
        return d_in.__class__

    # Arrays - check dtype
    dtype = get_dtype(d_in)

    # Structured dtype handling differs between legacy and new iterator layouts
    if dtype.fields is not None:
        if _legacy_iterators_available():
            return as_numba_type(gpu_struct(dtype))
        return gpu_struct(dtype)

    # Scalar dtype
    if _legacy_iterators_available():
        return numba.from_dtype(dtype)
    return from_numpy_dtype(dtype)


def set_cccl_iterator_state(cccl_it: Iterator, input_it):
    if cccl_it.is_kind_pointer():
        ptr = get_data_pointer(input_it)
        ptr_obj = make_pointer_object(ptr, input_it)
        cccl_it.state = ptr_obj
    else:
        state_ = input_it.state
        if isinstance(state_, (IteratorState, Pointer)):
            cccl_it.state = state_
        else:
            cccl_it.state = make_pointer_object(state_, input_it)


@functools.lru_cache()
def get_includes() -> List[str]:
    def as_option(p):
        if p is None:
            return ""
        return f"-I{p}"

    paths = get_include_paths().as_tuple()
    opts = [as_option(path) for path in paths]
    return opts


def _check_compile_result(cubin: bytes):
    # check compiled code for LDL/STL instructions
    temp_cubin_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        temp_cubin_file.write(cubin)
        out = subprocess.run(
            ["nvdisasm", "-gi", temp_cubin_file.name], capture_output=True
        )
        if out.returncode != 0:
            raise RuntimeError("nvdisasm failed")
        sass = out.stdout.decode("utf-8")
    except FileNotFoundError:
        sass = "nvdiasm not found, skipping SASS validation"
        warnings.warn(sass)

    assert "LDL" not in sass, "LDL instruction found in SASS"
    assert "STL" not in sass, "STL instruction found in SASS"
    return temp_cubin_file.name


# this global variable controls whether the compile result is checked
# for LDL/STL instructions. Should be set to `True` for testing only.
_check_sass: bool = False


def call_build(build_impl_fn: Callable, *args, **kwargs):
    """Calls given build_impl_fn callable while providing compute capability and paths

    Returns result of the call.
    """
    global _check_sass

    cc_major, cc_minor = cuda.get_current_device().compute_capability
    cub_path, thrust_path, libcudacxx_path, cuda_include_path = get_includes()
    common_data = CommonData(
        cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, cuda_include_path
    )
    result = build_impl_fn(
        *args,
        common_data,
        **kwargs,
    )

    if _check_sass:
        cubin = result._get_cubin()
        temp_cubin_file_name = _check_compile_result(cubin)
        os.unlink(temp_cubin_file_name)

    return result


def _make_host_cfunc(state_ptr_ty, fn):
    sig = numba.void(state_ptr_ty, numba.int64)
    c_advance_fn = numba.cfunc(sig)(fn)

    return c_advance_fn.ctypes


def cccl_iterator_set_host_advance(cccl_it: Iterator, array_or_iterator):
    if cccl_it.is_kind_iterator():
        it = array_or_iterator
        fn_impl = it.host_advance
        if fn_impl is not None:
            cccl_it.host_advance_fn = _make_host_cfunc(it.state_ptr_type, fn_impl)
        else:
            raise ValueError(
                f"Iterator of type {type(it)} does not provide definition of host_advance function"
            )
