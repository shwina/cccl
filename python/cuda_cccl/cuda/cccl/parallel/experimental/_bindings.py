# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Preload `nvrtc` and `nvJitLink` before importing the Cython extension.
# These shared libraries are indirect dependencies, pulled in via the direct
# dependency `cccl.c.parallel`. To ensure reliable symbol resolution at
# runtime, we explicitly load them first using `cuda.path_finder`.
#
# Without this step, importing the Cython extension directly may fail or behave
# inconsistently depending on environment setup and dynamic linker behavior.
# This indirection ensures the right loading order, regardless of how
# `_bindings` is first imported across the codebase.
#
# See also:
# https://github.com/NVIDIA/cuda-python/tree/main/cuda_pathfinder/cuda/pathfinder

<<<<<<< HEAD
# type: ignore[import-not-found]
from cuda.pathfinder import load_nvidia_dynamic_lib

for libname in ("nvrtc", "nvJitLink"):
    load_nvidia_dynamic_lib(libname)
=======
import sys
from pathlib import Path

from cuda.bindings.path_finder import (  # type: ignore[import-not-found]
    _load_nvidia_dynamic_library,
)
from cuda.cccl._version_utils import detect_cuda_version, get_recommended_extra

>>>>>>> 370d94bb3a (Initial cursor solution for multi-cuda support)

def _setup_cuda_version_path(cuda_version: int):
    """Set up the Python path to include version-specific binaries."""
    # Get the path to this module
    current_module_path = Path(__file__).parent

    # Look for version-specific binary directory
    version_dir = current_module_path / f"cu{cuda_version}"
    if version_dir.exists():
        # Add version-specific directory to Python path temporarily
        # This allows imports to find the correct versioned binaries
        version_str = str(version_dir)
        if version_str not in sys.path:
            sys.path.insert(0, version_str)
        print(f"Added CUDA {cuda_version} binary path: {version_dir}")
        return True

    return False


def _load_cuda_libraries():
    """Load the appropriate CUDA libraries based on detected CUDA version."""
    cuda_version = detect_cuda_version()

    # Default to CUDA 12 if version cannot be detected
    if cuda_version is None:
        cuda_version = 12
        print("Warning: Could not detect CUDA version, defaulting to CUDA 12")

    if cuda_version not in [12, 13]:
        raise RuntimeError(
            f"Unsupported CUDA version: {cuda_version}. Only CUDA 12 and 13 are supported."
        )

    print(f"Detected CUDA version: {cuda_version}")

    # Try to set up version-specific binary path
    has_version_specific_binaries = _setup_cuda_version_path(cuda_version)

    # Load appropriate libraries for the detected CUDA version
    for libname in ("nvrtc", "nvJitLink"):
        try:
            _load_nvidia_dynamic_library(libname)
        except Exception as e:
            # If direct loading fails, this might be because the user installed
            # the wrong CUDA version extras. Provide helpful error message.
            extra_name = get_recommended_extra(cuda_version)

            error_msg = (
                f"Failed to load {libname} library for CUDA {cuda_version}. "
                f"Make sure you have the correct CUDA toolkit installed, or install "
                f"cuda-cccl[{extra_name}] to get the required CUDA {cuda_version} components."
            )

            if has_version_specific_binaries:
                error_msg += (
                    "\n\nNote: Version-specific binaries were found in the wheel, but library loading failed. "
                    "This might indicate a compatibility issue with your CUDA installation."
                )
            else:
                error_msg += (
                    f"\n\nNote: No version-specific binaries found in wheel. "
                    f"This wheel may not include binaries for CUDA {cuda_version}."
                )

            raise RuntimeError(error_msg) from e


# Load libraries at import time
_load_cuda_libraries()

# Import the actual bindings implementation
# The path setup above ensures the correct version-specific binaries are loaded
try:
    from ._bindings_impl import *  # noqa: E402 F403
except ImportError as e:
    # Provide helpful error message if binding import fails
    cuda_version = detect_cuda_version() or 12
    extra_name = get_recommended_extra(cuda_version)

    raise ImportError(
        f"Failed to import CUDA CCCL bindings for CUDA {cuda_version}. "
        f"Please ensure you have installed cuda-cccl[{extra_name}] for CUDA {cuda_version} support."
    ) from e
