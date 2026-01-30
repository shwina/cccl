# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
C++ code generation and compilation infrastructure for cuda.compute.

This module provides utilities to generate C++ source code and compile it
to LTOIR (Link-Time Optimization IR) for use with CCCL algorithms.
"""

from __future__ import annotations

import functools

from cuda.cccl import get_include_paths  # type: ignore[import-not-found]


def _get_arch_string() -> str:
    """Get the compute capability string for the current device."""
    from cuda.core import Device

    device = Device()
    cc_major, cc_minor = device.compute_capability
    return f"sm_{cc_major}{cc_minor}"


@functools.lru_cache(maxsize=1)
def _get_include_paths() -> list[str]:
    """Get include paths for CCCL headers."""
    paths = get_include_paths().as_tuple()
    return [p for p in paths if p is not None]


@functools.lru_cache(maxsize=256)
def compile_cpp_to_ltoir(
    source: str,
    symbols: tuple[str, ...],
    arch: str | None = None,
) -> bytes:
    """
    Compile C++ source code to LTOIR.

    Args:
        source: C++ source code string
        symbols: Tuple of symbol names to extract from the compiled code
        arch: Target architecture (e.g., "sm_80"). If None, uses current device.

    Returns:
        LTOIR bytes that can be used with CompiledOp

    Example:
        source = '''
        extern "C" __device__ void my_add(void* a, void* b, void* result) {
            *static_cast<int*>(result) = *static_cast<int*>(a) + *static_cast<int*>(b);
        }
        '''
        ltoir = compile_cpp_to_ltoir(source, ("my_add",))
    """
    from cuda.core import Program, ProgramOptions

    if arch is None:
        arch = _get_arch_string()

    # Get include paths
    include_paths = _get_include_paths()

    # Configure compilation options for LTO
    opts = ProgramOptions(
        arch=arch,
        relocatable_device_code=True,
        link_time_optimization=True,
        std="c++17",
        include_path=include_paths,
    )

    # Compile to LTOIR
    program = Program(source, "c++", options=opts)
    result = program.compile("ltoir")

    return result.code


def extract_extra_ltoirs(obj) -> list[bytes]:
    """
    Safely extract extra_ltoirs from an Op/Iterator object.

    Args:
        obj: An object that may have an extra_ltoirs attribute

    Returns:
        List of LTOIR byte strings, or empty list if not present
    """
    extras = getattr(obj, "extra_ltoirs", None)
    return list(extras) if extras else []


# =============================================================================
# C++ type name utilities
# =============================================================================


def cpp_type_name(size: int, is_signed: bool = True, is_float: bool = False) -> str:
    """
    Get the C++ type name for a given size and signedness.

    Args:
        size: Size in bytes (1, 2, 4, or 8)
        is_signed: Whether the type is signed (for integers)
        is_float: Whether the type is floating point

    Returns:
        C++ type name string
    """
    if is_float:
        if size == 2:
            return "__half"
        elif size == 4:
            return "float"
        elif size == 8:
            return "double"
        else:
            raise ValueError(f"Unsupported float size: {size}")
    else:
        prefix = "" if is_signed else "u"
        if size == 1:
            return f"{prefix}int8_t"
        elif size == 2:
            return f"{prefix}int16_t"
        elif size == 4:
            return f"{prefix}int32_t"
        elif size == 8:
            return f"{prefix}int64_t"
        else:
            raise ValueError(f"Unsupported integer size: {size}")


def cpp_type_from_descriptor(type_desc) -> str:
    """
    Get the C++ type name from a TypeDescriptor.

    Args:
        type_desc: A TypeDescriptor instance

    Returns:
        C++ type name string
    """
    from ._bindings import TypeEnum

    # Map TypeEnum to C++ types
    type_map = {
        TypeEnum.INT8: "int8_t",
        TypeEnum.INT16: "int16_t",
        TypeEnum.INT32: "int32_t",
        TypeEnum.INT64: "int64_t",
        TypeEnum.UINT8: "uint8_t",
        TypeEnum.UINT16: "uint16_t",
        TypeEnum.UINT32: "uint32_t",
        TypeEnum.UINT64: "uint64_t",
        TypeEnum.FLOAT16: "__half",
        TypeEnum.FLOAT32: "float",
        TypeEnum.FLOAT64: "double",
        TypeEnum.BOOLEAN: "bool",
    }

    if type_desc.info.typenum in type_map:
        return type_map[type_desc.info.typenum]

    # For STORAGE types (structs), we can't return a proper type name.
    # The caller should handle struct types specially using void* with reinterpret_cast.
    raise ValueError(
        f"Cannot generate C++ type name for {type_desc}. "
        "Struct types require void* with explicit size/alignment handling."
    )
