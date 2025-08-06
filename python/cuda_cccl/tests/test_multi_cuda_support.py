# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for multi-CUDA version support in cuda-cccl.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cuda.cccl._version_utils import (
    detect_cuda_version,
    get_cuda_path,
    get_cuda_version_from_path,
    get_recommended_extra,
    validate_cuda_version,
)


class TestCudaVersionDetection:
    """Test CUDA version detection functionality."""

    def test_detect_cuda_version_with_nvcc(self):
        """Test CUDA version detection when nvcc is available."""
        # This test only runs if nvcc is actually available
        if shutil.which("nvcc"):
            version = detect_cuda_version()
            assert version is None or version in [12, 13]

    def test_get_recommended_extra(self):
        """Test recommended extra selection."""
        assert get_recommended_extra(12) == "cu12"
        assert get_recommended_extra(13) == "cu13"
        assert get_recommended_extra(None) == "cu12"  # Default

    def test_validate_cuda_version_supported(self):
        """Test validation of supported CUDA versions."""
        # Should not raise for supported versions
        validate_cuda_version(12)
        validate_cuda_version(13)
        validate_cuda_version(None)  # Should not raise

    def test_validate_cuda_version_unsupported(self):
        """Test validation rejects unsupported CUDA versions."""
        with pytest.raises(RuntimeError, match="Unsupported CUDA version: 11"):
            validate_cuda_version(11)

        with pytest.raises(RuntimeError, match="Unsupported CUDA version: 14"):
            validate_cuda_version(14)

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_cuda_version_no_cuda(self):
        """Test behavior when no CUDA is detected."""
        with patch("shutil.which", return_value=None):
            with patch("pathlib.Path.exists", return_value=False):
                version = detect_cuda_version()
                assert version is None

    def test_get_cuda_version_from_path_with_version_json(self):
        """Test version detection from version.json file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_path = Path(temp_dir)
            version_json = cuda_path / "version.json"

            # Create mock version.json for CUDA 12
            version_json.write_text('{"cuda": {"version": "12.6.0"}}')

            version = get_cuda_version_from_path(cuda_path)
            assert version == 12

    def test_get_cuda_version_from_path_with_version_txt(self):
        """Test version detection from version.txt file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_path = Path(temp_dir)
            version_txt = cuda_path / "version.txt"

            # Create mock version.txt for CUDA 13
            version_txt.write_text("CUDA Version 13.0\n")

            version = get_cuda_version_from_path(cuda_path)
            assert version == 13

    def test_get_cuda_version_from_path_no_version_files(self):
        """Test version detection when no version files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cuda_path = Path(temp_dir)
            version = get_cuda_version_from_path(cuda_path)
            assert version is None


class TestIncludePathsIntegration:
    """Test integration with include paths."""

    def test_include_paths_contains_version(self):
        """Test that include paths contain CUDA version information."""
        from cuda.cccl.headers.include_paths import get_include_paths

        paths = get_include_paths()

        # Should have version info (None if no CUDA detected)
        assert paths.cuda_version is None or paths.cuda_version in [12, 13]

        # Should have required paths
        assert paths.libcudacxx is not None
        assert paths.cub is not None
        assert paths.thrust is not None


class TestOptionalDependencies:
    """Test optional dependency configuration."""

    def test_pyproject_toml_has_correct_extras(self):
        """Test that pyproject.toml has the correct optional dependencies."""
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        extras = config["project"]["optional-dependencies"]

        # Check cu12 extra
        assert "cu12" in extras
        cu12_deps = extras["cu12"]
        assert any("cuda-bindings>=12.9.1,<13.0.0" in dep for dep in cu12_deps)
        assert any("nvidia-cuda-nvrtc-cu12" in dep for dep in cu12_deps)
        assert any("nvidia-nvjitlink-cu12" in dep for dep in cu12_deps)

        # Check cu13 extra
        assert "cu13" in extras
        cu13_deps = extras["cu13"]
        assert any("cuda-bindings>=13.0.0,<14.0.0" in dep for dep in cu13_deps)
        assert any("nvidia-cuda-nvrtc-cu13" in dep for dep in cu13_deps)
        assert any("nvidia-nvjitlink-cu13" in dep for dep in cu13_deps)

    def test_base_dependencies_exclude_cuda_specific(self):
        """Test that base dependencies don't include CUDA-specific packages."""
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        base_deps = config["project"]["dependencies"]

        # Should not contain CUDA-specific dependencies
        cuda_specific_patterns = [
            "cuda-bindings",
            "nvidia-cuda-nvrtc",
            "nvidia-nvjitlink",
        ]

        for dep in base_deps:
            for pattern in cuda_specific_patterns:
                assert pattern not in dep, (
                    f"Base dependency {dep} contains CUDA-specific pattern {pattern}"
                )


@pytest.mark.skipif(not shutil.which("nvcc"), reason="CUDA toolkit not available")
class TestWithCudaToolkit:
    """Tests that require CUDA toolkit to be installed."""

    def test_actual_cuda_detection(self):
        """Test with actual CUDA installation."""
        version = detect_cuda_version()
        assert version in [12, 13], f"Detected unsupported CUDA version: {version}"

    def test_cuda_path_detection(self):
        """Test CUDA path detection with actual installation."""
        cuda_path = get_cuda_path()
        assert cuda_path is not None
        assert cuda_path.exists()
        assert (cuda_path / "bin" / "nvcc").exists()

    def test_version_consistency(self):
        """Test that detected version is consistent across methods."""
        cuda_path = get_cuda_path()
        if cuda_path:
            path_version = get_cuda_version_from_path(cuda_path)
            detected_version = detect_cuda_version()
            assert path_version == detected_version
