# Multi-CUDA Version Support for cuda-cccl

This document describes the multi-CUDA version support implemented in the cuda-cccl Python package.

## Overview

The cuda-cccl package now supports both CUDA 12 and CUDA 13 through a single wheel that automatically detects the available CUDA runtime and loads the appropriate binaries.

## Installation Options

### Option 1: Base Package (Requires Pre-installed CUDA)

```bash
pip install cuda-cccl
```

This installs the base package assuming you already have:
- CUDA Toolkit (12.x or 13.x) with nvrtc and nvjitlink
- Appropriate cuda-bindings version

### Option 2: CUDA 12 Bundle

```bash
pip install cuda-cccl[cu12]
```

This installs:
- cuda-cccl package
- cuda-bindings>=12.9.1,<13.0.0
- nvidia-cuda-nvrtc-cu12
- nvidia-nvjitlink-cu12

### Option 3: CUDA 13 Bundle

```bash
pip install cuda-cccl[cu13]
```

This installs:
- cuda-cccl package
- cuda-bindings>=13.0.0,<14.0.0
- nvidia-cuda-nvrtc-cu13
- nvidia-nvjitlink-cu13

## Runtime Detection

The package automatically detects the CUDA version at runtime using the following methods (in order):

1. **CUDA_PATH environment variable** - if set and points to valid CUDA installation
2. **nvcc in PATH** - if nvcc is available and can be executed
3. **Default installation paths** - checks `/usr/local/cuda` and `/opt/cuda`

Version detection tries multiple approaches:
- Reading `version.json` (CUDA 11.1+)
- Reading `version.txt` (older versions)
- Executing `nvcc --version`

## Architecture

### Directory Structure

```
cuda/cccl/parallel/experimental/
├── _bindings.py              # Main entry point with version detection
├── cu12/                     # CUDA 12 specific binaries
│   ├── _bindings_impl.so     # Cython extension for CUDA 12
│   ├── cccl/                 # C++ library binaries for CUDA 12
│   └── cuda_version_info.txt # Build information
├── cu13/                     # CUDA 13 specific binaries
│   ├── _bindings_impl.so     # Cython extension for CUDA 13
│   ├── cccl/                 # C++ library binaries for CUDA 13
│   └── cuda_version_info.txt # Build information
└── _cccl_interop.py          # Shared utilities
```

### Import Flow

1. User imports `cuda.cccl.parallel.experimental._bindings`
2. `_bindings.py` detects CUDA version using `_version_utils`
3. Appropriate version-specific directory (`cu12` or `cu13`) is added to Python path
4. NVRTC and nvJitLink libraries are loaded using `cuda.bindings.path_finder`
5. Version-specific `_bindings_impl` extension is imported

### Error Handling

The package provides clear error messages for common issues:

- **No CUDA detected**: Falls back to CUDA 12, warns user
- **Unsupported CUDA version**: Suggests installing appropriate extra
- **Library loading failure**: Provides specific troubleshooting steps
- **Missing binaries**: Indicates if wheel lacks version-specific binaries

## Build Process

### Local Development Build

For single CUDA version:
```bash
cd python/cuda_cccl
python -m build --wheel
```

For multi-CUDA build (requires both CUDA 12 and 13 installed):
```bash
cd python/cuda_cccl
python build_multi_cuda.py
```

### CI/CD Pipeline

The GitHub Actions workflow builds and tests for both CUDA versions:

1. **Matrix Build**: Builds wheels for CUDA 12 and 13 across Python 3.9-3.12
2. **Installation Testing**: Tests all installation scenarios
3. **Deployment**: Publishes unified wheel to PyPI

### Build Script Features

The `build_multi_cuda.py` script:
- Automatically detects available CUDA installations
- Builds separate wheels for each CUDA version
- Merges wheels into single distribution with version-specific binaries
- Handles build environment setup for each CUDA version

## Version Utilities

The `cuda.cccl._version_utils` module provides shared utilities:

- `detect_cuda_version()`: Runtime CUDA version detection
- `get_cuda_path()`: Find CUDA installation path
- `validate_cuda_version()`: Check version compatibility
- `get_recommended_extra()`: Suggest appropriate pip extra

## Testing

### Unit Tests

Tests verify:
- Correct CUDA version detection across environments
- Proper library loading for each CUDA version
- Error handling for unsupported configurations
- Include path resolution

### Integration Tests

CI tests verify:
- Package installation with different extras
- Runtime behavior with various CUDA configurations
- Cross-compatibility between CUDA versions

## Troubleshooting

### Common Issues

**ImportError on package import**:
- Check CUDA installation with `nvcc --version`
- Verify cuda-bindings compatibility
- Install appropriate extra: `pip install cuda-cccl[cu12]` or `cuda-cccl[cu13]`

**Wrong CUDA version detected**:
- Set `CUDA_PATH` environment variable to correct installation
- Ensure correct nvcc is first in PATH
- Check for conflicting CUDA installations

**Library loading errors**:
- Verify CUDA toolkit components are installed
- Check LD_LIBRARY_PATH includes CUDA libraries
- Try installing bundled extras instead of system CUDA

### Debug Information

Enable verbose output:
```python
import cuda.cccl
# Version information is printed during import

from cuda.cccl import get_include_paths
paths = get_include_paths()
print(f"CUDA version: {paths.cuda_version}")
print(f"CUDA path: {paths.cuda}")
```

## Migration Guide

### From Single-CUDA Version

Existing code remains compatible. The package will automatically detect and use the appropriate CUDA version.

### Pinning CUDA Version

To ensure specific CUDA version compatibility:
```bash
# Force CUDA 12
pip install cuda-cccl[cu12]

# Force CUDA 13
pip install cuda-cccl[cu13]
```

### Environment Variables

Set to override detection:
```bash
export CUDA_PATH=/usr/local/cuda-12
export CUDA_PATH=/usr/local/cuda-13
```

## Implementation Details

### Key Changes

1. **Dependency Management**: Moved CUDA-specific dependencies to optional extras
2. **Runtime Detection**: Added comprehensive CUDA version detection
3. **Binary Organization**: Version-specific binary directories in wheel
4. **Import System**: Dynamic path manipulation for version-specific imports
5. **Build System**: Enhanced CMake configuration for multi-version builds

### Backwards Compatibility

- Existing installations continue to work
- API remains unchanged
- Single-CUDA builds still supported for development

### Performance Considerations

- Version detection adds ~10ms startup time
- Binary organization has no runtime performance impact
- Library loading follows same patterns as before
