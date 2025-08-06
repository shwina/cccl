# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
CUDA version detection utilities shared across the cccl package.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def detect_cuda_version() -> Optional[int]:
    """
    Detect the CUDA major version from the installed toolkit.
    Returns the major version (12 or 13) or None if not found.
    """
    # Check environment variable first
    cuda_path_str = os.environ.get("CUDA_PATH")
    if cuda_path_str:
        cuda_path = Path(cuda_path_str)
        if cuda_path.exists():
            return get_cuda_version_from_path(cuda_path)

    # Check nvcc in PATH
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        cuda_path = Path(nvcc_path).parent.parent
        return get_cuda_version_from_path(cuda_path)

    # Check default installation paths
    for default_path in [Path("/usr/local/cuda"), Path("/opt/cuda")]:
        if default_path.exists():
            version = get_cuda_version_from_path(default_path)
            if version:
                return version

    return None


def get_cuda_version_from_path(cuda_path: Path) -> Optional[int]:
    """Extract CUDA major version from a CUDA installation path."""
    # Try to read version from version.json (CUDA 11.1+)
    version_json = cuda_path / "version.json"
    if version_json.exists():
        try:
            import json

            with open(version_json) as f:
                data = json.load(f)
                version_str = data.get("cuda", {}).get("version", "")
                if version_str:
                    major_version = int(version_str.split(".")[0])
                    return major_version
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # Try to read version from version.txt (older CUDA versions)
    version_txt = cuda_path / "version.txt"
    if version_txt.exists():
        try:
            with open(version_txt) as f:
                line = f.readline().strip()
                # Format: "CUDA Version 12.6"
                if "CUDA Version" in line:
                    version_str = line.split()[-1]
                    major_version = int(version_str.split(".")[0])
                    return major_version
        except (ValueError, IndexError):
            pass

    # Try to get version from nvcc
    nvcc_path = cuda_path / "bin" / "nvcc"
    if nvcc_path.exists():
        try:
            result = subprocess.run(
                [str(nvcc_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line.lower():
                        # Extract version from line like "Cuda compilation tools, release 12.6, V12.6.68"
                        import re

                        match = re.search(r"release (\d+)\.(\d+)", line)
                        if match:
                            major_version = int(match.group(1))
                            return major_version
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
            pass

    return None


def get_cuda_path() -> Optional[Path]:
    """Get the CUDA installation path."""
    cuda_path_str = os.environ.get("CUDA_PATH")
    if cuda_path_str:
        cuda_path = Path(cuda_path_str)
        if cuda_path.exists():
            return cuda_path

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        return Path(nvcc_path).parent.parent

    default_path = Path("/usr/local/cuda")
    if default_path.exists():
        return default_path

    return None


def validate_cuda_version(cuda_version: Optional[int]) -> None:
    """Validate that the CUDA version is supported."""
    if cuda_version is not None and cuda_version not in [12, 13]:
        raise RuntimeError(
            f"Unsupported CUDA version: {cuda_version}. "
            f"Only CUDA 12 and 13 are supported. "
            f"Please install the appropriate cuda-cccl[cu12] or cuda-cccl[cu13] extra."
        )


def get_recommended_extra(cuda_version: Optional[int]) -> str:
    """Get the recommended pip extra for the detected CUDA version."""
    if cuda_version == 13:
        return "cu13"
    else:
        return "cu12"  # Default to cu12 for unknown versions
