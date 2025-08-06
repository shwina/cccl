# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from cuda.cccl._version_utils import (
    detect_cuda_version,
    get_cuda_path,
    validate_cuda_version,
)


@dataclass
class IncludePaths:
    cuda: Optional[Path]
    libcudacxx: Path
    cub: Path
    thrust: Path
    cuda_version: Optional[int] = None

    def as_tuple(self):
        # Note: higher-level ... lower-level order:
        return (self.libcudacxx, self.cub, self.thrust, self.cuda)


@lru_cache()
def get_include_paths(probe_file: str = "cub/version.cuh") -> IncludePaths:
    # TODO: once docs env supports Python >= 3.9, we
    # can move this to a module-level import.
    from importlib.resources import as_file, files

    cuda_incl = None
    cuda_version = None
    cuda_path = get_cuda_path()
    if cuda_path is not None:
        cuda_incl = cuda_path / "include"
        cuda_version = detect_cuda_version()

    # Provide helpful error message if CUDA version is not supported
    validate_cuda_version(cuda_version)

    with as_file(files("cuda.cccl.headers.include")) as f:
        cccl_incl = Path(f)

    probe_file_path = Path(probe_file)
    if not (cccl_incl / probe_file_path).exists():
        for sp in sys.path:
            cccl_incl = Path(sp).resolve() / "cuda" / "cccl" / "headers" / "include"
            if (cccl_incl / probe_file_path).exists():
                break
        else:
            raise RuntimeError("Unable to locate CCCL include directory.")

    return IncludePaths(
        cuda=cuda_incl,
        libcudacxx=cccl_incl,
        cub=cccl_incl,
        thrust=cccl_incl,
        cuda_version=cuda_version,
    )
