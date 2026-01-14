# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
JIT compilation support for cuda.compute.

This module handles compiling Python callables to LTOIR. It is imported
lazily only when Python callables are used as operators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from numba import cuda, types
from numba.core.extending import as_numba_type

from ._bindings import Op, OpKind, TypeEnum
from .op import OpContext

if TYPE_CHECKING:
    from ._types import TypeDescriptor


def _type_descriptor_to_numba(td: TypeDescriptor):
    """
    Convert a TypeDescriptor to an internal type representation.

    Args:
        td: TypeDescriptor to convert

    Returns:
        Corresponding internal type
    """
    enum_to_numba = {
        TypeEnum.INT8: types.int8,
        TypeEnum.INT16: types.int16,
        TypeEnum.INT32: types.int32,
        TypeEnum.INT64: types.int64,
        TypeEnum.UINT8: types.uint8,
        TypeEnum.UINT16: types.uint16,
        TypeEnum.UINT32: types.uint32,
        TypeEnum.UINT64: types.uint64,
        TypeEnum.FLOAT16: types.float16,
        TypeEnum.FLOAT32: types.float32,
        TypeEnum.FLOAT64: types.float64,
        TypeEnum.BOOLEAN: types.boolean,
    }

    if td.type_enum in enum_to_numba:
        return enum_to_numba[td.type_enum]

    # Handle complex types (they come through as STORAGE with specific names)
    if td.type_enum == TypeEnum.STORAGE:
        if td.name == "complex64":
            return types.complex64
        if td.name == "complex128":
            return types.complex128

    # For STORAGE types (structs), conversion requires gpu_struct registration
    raise ValueError(
        f"Cannot convert TypeDescriptor({td.name}) to internal type. "
        "For struct types, use gpu_struct()."
    )


def _to_internal_type(t):
    """
    Convert various type representations to internal type format.

    Handles:
    - Internal types (returned as-is)
    - TypeDescriptor instances
    - ZipValueType (converted to Numba Tuple)
    - Struct classes with _get_numba_type method
    """
    if isinstance(t, types.Type):
        return t

    from ._cccl_interop import ZipValueType
    from ._types import TypeDescriptor

    # ZipValueType - convert to Numba Tuple for indexing support
    if isinstance(t, ZipValueType):
        component_numba_types = tuple(_to_internal_type(ct) for ct in t.component_types)
        return types.Tuple(component_numba_types)

    if isinstance(t, TypeDescriptor):
        return _type_descriptor_to_numba(t)

    # Struct class with lazy registration
    if isinstance(t, type) and hasattr(t, "_get_numba_type"):
        return t._get_numba_type()

    return as_numba_type(t)


def _is_struct_type_descriptor(t) -> bool:
    """Check if a type is a struct-like TypeDescriptor that can't be converted."""
    from ._cccl_interop import ZipValueType
    from ._types import TypeDescriptor

    # ZipValueType is handled specially (converted to Tuple), not a struct problem
    if isinstance(t, ZipValueType):
        return False

    # Struct classes are handled via _get_numba_type
    if isinstance(t, type) and hasattr(t, "_get_numba_type"):
        return False

    if not isinstance(t, TypeDescriptor):
        return False

    # Check if it's a STORAGE type that's not complex (complex can be converted)
    if t.type_enum == TypeEnum.STORAGE:
        return t.name not in ("complex64", "complex128")

    return False


def compile_jit_op(
    func: Callable,
    input_types: tuple,
    output_type=None,
    context: OpContext = OpContext.BINARY_OP,
) -> Op:
    """
    Compile a Python callable to a CCCL Op.

    Args:
        func: Python callable to compile
        input_types: Tuple of types for input arguments
        output_type: Optional type for return value
        context: The context determining the wrapper signature

    Returns:
        Compiled Op object for C++ interop
    """
    import inspect

    from ._odr_helpers import create_op_void_ptr_wrapper
    from .numba_utils import get_inferred_return_type, signature_from_annotations

    # Check if any input types are struct TypeDescriptors that can't be converted
    # In that case, try to get types from function annotations
    has_unconvertible_struct = any(_is_struct_type_descriptor(t) for t in input_types)

    if has_unconvertible_struct:
        # Try to get types from function annotations
        sig_obj = inspect.signature(func)
        params = list(sig_obj.parameters.values())

        # Check if all struct parameters have annotations
        all_annotated = True
        for i, param in enumerate(params):
            if i < len(input_types) and _is_struct_type_descriptor(input_types[i]):
                if param.annotation == inspect.Parameter.empty:
                    all_annotated = False
                    break

        if all_annotated:
            internal_input_types = []
            for i, param in enumerate(params):
                if param.annotation != inspect.Parameter.empty:
                    internal_input_types.append(as_numba_type(param.annotation))
                elif i < len(input_types):
                    internal_input_types.append(_to_internal_type(input_types[i]))
            internal_input_types = tuple(internal_input_types)
        else:
            # For ZipIterator tuples and other dynamic struct types,
            # we need to create a compatible Numba struct type from the TypeDescriptor

            internal_input_types = []
            for t in input_types:
                if _is_struct_type_descriptor(t):
                    # Create a Numba UniTuple or Record type based on the struct
                    # For zip2 etc, use a generic approach based on size
                    internal_input_types.append(_create_numba_struct_type(t))
                else:
                    internal_input_types.append(_to_internal_type(t))
            internal_input_types = tuple(internal_input_types)
    else:
        # Convert input types to internal representation normally
        internal_input_types = tuple(_to_internal_type(t) for t in input_types)

    # Determine output type
    if output_type is not None:
        internal_output_type = _to_internal_type(output_type)
    else:
        # Try annotations first
        try:
            sig = signature_from_annotations(func)
            internal_output_type = sig.return_type
        except ValueError:
            # Infer from function
            internal_output_type = get_inferred_return_type(func, internal_input_types)

    # Build the signature
    sig = internal_output_type(*internal_input_types)

    # Create wrapper based on context
    if context in (OpContext.BINARY_OP, OpContext.UNARY_OP):
        wrapped_op, wrapper_sig = create_op_void_ptr_wrapper(func, sig)
    elif context == OpContext.ADVANCE:
        from ._odr_helpers import create_advance_void_ptr_wrapper

        state_ptr_type = internal_input_types[0]
        wrapped_op, wrapper_sig = create_advance_void_ptr_wrapper(func, state_ptr_type)
    elif context == OpContext.INPUT_DEREF:
        from ._odr_helpers import create_input_dereference_void_ptr_wrapper

        state_ptr_type = internal_input_types[0]
        value_type = internal_output_type
        wrapped_op, wrapper_sig = create_input_dereference_void_ptr_wrapper(
            func, state_ptr_type, value_type
        )
    elif context == OpContext.OUTPUT_DEREF:
        from ._odr_helpers import create_output_dereference_void_ptr_wrapper

        state_ptr_type = internal_input_types[0]
        value_type = (
            internal_input_types[1]
            if len(internal_input_types) > 1
            else internal_output_type
        )
        wrapped_op, wrapper_sig = create_output_dereference_void_ptr_wrapper(
            func, state_ptr_type, value_type
        )
    else:
        raise ValueError(f"Unknown OpContext: {context}")

    # Compile to LTOIR
    ltoir, _ = cuda.compile(wrapped_op, sig=wrapper_sig, output="ltoir")

    return Op(
        operator_type=OpKind.STATELESS,
        name=wrapped_op.__name__,
        ltoir=ltoir,
        state_alignment=1,
        state=None,
    )


def _internal_type_to_type_info(internal_type: types.Type):
    """
    Convert an internal type to CCCL TypeInfo.

    Args:
        internal_type: Internal type to convert

    Returns:
        TypeInfo object for CCCL interop
    """
    from ._bindings import TypeInfo

    context = cuda.descriptor.cuda_target.target_context
    value_type = context.get_value_type(internal_type)

    # Handle record types (passed by pointer)
    if isinstance(internal_type, types.Record):
        value_type = value_type.pointee

    size = value_type.get_abi_size(context.target_data)
    alignment = value_type.get_abi_alignment(context.target_data)

    # Map to TypeEnum
    type_enum_map = {
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
        types.boolean: TypeEnum.BOOLEAN,
        types.complex64: TypeEnum.STORAGE,
        types.complex128: TypeEnum.STORAGE,
    }
    type_enum = type_enum_map.get(internal_type, TypeEnum.STORAGE)

    return TypeInfo(size, alignment, type_enum)


def get_type_descriptor(internal_type):
    """
    Convert an internal type to a TypeDescriptor.

    This is used when we have an internal type from an iterator and need
    to convert it to a TypeDescriptor for the public API.
    """
    from ._types import TypeDescriptor

    info = _internal_type_to_type_info(internal_type)
    return TypeDescriptor(
        info.size,
        info.alignment,
        TypeEnum(info.typenum),
        str(internal_type),
    )
