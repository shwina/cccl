#!/usr/bin/env python3
"""
Build script for creating a single cuda-cccl wheel with support for both CUDA 12 and 13.

This script builds separate binary extensions for CUDA 12 and 13, then packages them
into a single wheel that can automatically detect and load the appropriate binary
at runtime.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from cuda.cccl._version_utils import detect_cuda_version


def run_command(
    cmd: List[str], cwd: Path = None, env: dict = None
) -> subprocess.CompletedProcess:
    """Run a command with error handling."""
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"  Working directory: {cwd}")

    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

    return result


def find_cuda_installations() -> dict:
    """Find available CUDA installations."""
    cuda_installations = {}

    # Check common CUDA installation paths
    potential_paths = [
        Path("/usr/local/cuda-12"),
        Path("/usr/local/cuda-13"),
        Path("/opt/cuda-12"),
        Path("/opt/cuda-13"),
    ]

    # Also check versioned installations in default CUDA path
    cuda_base = Path("/usr/local/cuda")
    if cuda_base.exists() and cuda_base.is_symlink():
        # If /usr/local/cuda is a symlink, look for versioned directories
        parent = cuda_base.parent
        for cuda_dir in parent.glob("cuda-*"):
            if cuda_dir.is_dir():
                potential_paths.append(cuda_dir)

    for path in potential_paths:
        if path.exists() and (path / "bin" / "nvcc").exists():
            # Try to detect version
            try:
                result = subprocess.run(
                    [str(path / "bin" / "nvcc"), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    import re

                    for line in result.stdout.split("\n"):
                        if "release" in line.lower():
                            match = re.search(r"release (\d+)\.(\d+)", line)
                            if match:
                                major_version = int(match.group(1))
                                if major_version in [12, 13]:
                                    cuda_installations[major_version] = path
                                break
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                continue

    return cuda_installations


def build_for_cuda_version(cuda_version: int, cuda_path: Path, build_dir: Path) -> Path:
    """Build the package for a specific CUDA version."""
    print(f"\n=== Building for CUDA {cuda_version} ===")

    # Set up environment for this CUDA version
    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_path)
    env["CUDA_PATH"] = str(cuda_path)
    env["PATH"] = f"{cuda_path / 'bin'}:{env.get('PATH', '')}"

    # Create version-specific build directory
    version_build_dir = build_dir / f"cuda{cuda_version}"
    version_build_dir.mkdir(parents=True, exist_ok=True)

    # Copy source to build directory
    src_dir = Path(__file__).parent
    for item in src_dir.iterdir():
        if item.name in [
            ".git",
            "build",
            "dist",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
        ]:
            continue

        dest = version_build_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # Build the wheel
    run_command(
        [sys.executable, "-m", "build", "--wheel", "--outdir", "dist"],
        cwd=version_build_dir,
        env=env,
    )

    # Find the generated wheel
    dist_dir = version_build_dir / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel found for CUDA {cuda_version}")

    return wheels[0]


def merge_wheels(wheels: List[Path], output_dir: Path) -> Path:
    """Merge multiple wheels into a single wheel with version-specific binaries."""
    print("\n=== Merging wheels ===")

    # Extract all wheels to temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted_wheels = []

        for i, wheel in enumerate(wheels):
            extract_dir = temp_path / f"wheel_{i}"
            extract_dir.mkdir()

            # Extract wheel
            run_command(
                ["python", "-m", "wheel", "unpack", str(wheel), str(extract_dir)]
            )

            # Find the extracted directory
            wheel_dirs = list(extract_dir.iterdir())
            if len(wheel_dirs) != 1:
                raise RuntimeError(
                    f"Expected 1 directory in extracted wheel, found {len(wheel_dirs)}"
                )

            extracted_wheels.append(wheel_dirs[0])

        # Use the first wheel as the base and merge binaries from others
        base_wheel = extracted_wheels[0]

        # Create version-specific directories for binaries
        for i, wheel_dir in enumerate(extracted_wheels):
            cuda_version = 12 if i == 0 else 13  # Adjust based on your wheel order

            # Find and move binary files to version-specific directories
            for root, dirs, files in os.walk(wheel_dir):
                for file in files:
                    if file.endswith((".so", ".dll", ".pyd")):
                        src_path = Path(root) / file
                        rel_path = src_path.relative_to(wheel_dir)

                        # Create version-specific path
                        parts = list(rel_path.parts)
                        if (
                            len(parts) >= 2
                            and parts[0] == "cuda"
                            and parts[1] == "cccl"
                        ):
                            # Insert cuda version directory
                            new_parts = parts[:2] + [f"cu{cuda_version}"] + parts[2:]
                            dest_path = base_wheel / Path(*new_parts)
                        else:
                            dest_path = base_wheel / rel_path

                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        if i == 0:
                            # For base wheel, just ensure directory exists
                            continue
                        else:
                            # Copy from other wheels
                            shutil.copy2(src_path, dest_path)

        # Repack the merged wheel
        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                "python",
                "-m",
                "wheel",
                "pack",
                str(base_wheel),
                "--dest-dir",
                str(output_dir),
            ]
        )

        # Find the output wheel
        output_wheels = list(output_dir.glob("*.whl"))
        if not output_wheels:
            raise RuntimeError("Failed to create merged wheel")

        return output_wheels[0]


def main():
    """Main build script."""
    print("CUDA CCCL Multi-Version Build Script")
    print("====================================")

    # Find CUDA installations
    cuda_installations = find_cuda_installations()
    print(f"Found CUDA installations: {cuda_installations}")

    if not cuda_installations:
        print("No CUDA installations found. Building with current environment.")
        # Fall back to regular build
        current_version = detect_cuda_version()
        if current_version:
            print(f"Detected CUDA {current_version} in current environment")
            run_command([sys.executable, "-m", "build", "--wheel"])
        else:
            print("No CUDA detected. Building without CUDA support.")
            run_command([sys.executable, "-m", "build", "--wheel"])
        return

    required_versions = [12, 13]
    missing_versions = [v for v in required_versions if v not in cuda_installations]

    if missing_versions:
        print(f"WARNING: Missing CUDA versions: {missing_versions}")
        print("Building with available versions only.")

    # Create build directory
    build_dir = Path("build_multi")
    build_dir.mkdir(exist_ok=True)

    # Build for each available CUDA version
    wheels = []
    for version in required_versions:
        if version in cuda_installations:
            wheel = build_for_cuda_version(
                version, cuda_installations[version], build_dir
            )
            wheels.append(wheel)

    if len(wheels) > 1:
        # Merge wheels if we have multiple versions
        output_dir = Path("dist")
        merged_wheel = merge_wheels(wheels, output_dir)
        print(f"\nMerged wheel created: {merged_wheel}")
    else:
        # Single version build
        if wheels:
            output_dir = Path("dist")
            output_dir.mkdir(exist_ok=True)
            final_wheel = output_dir / wheels[0].name
            shutil.copy2(wheels[0], final_wheel)
            print(f"\nSingle-version wheel created: {final_wheel}")

    print("\nBuild complete!")


if __name__ == "__main__":
    main()
