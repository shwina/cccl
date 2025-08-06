# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)

project(
    cuda_cccl
    DESCRIPTION "Python package cuda_cccl"
    LANGUAGES CUDA CXX C
)

find_package(CUDAToolkit)

# Detect CUDA version and set up version-specific build
if(CUDAToolkit_VERSION)
    string(REGEX MATCH "^([0-9]+)" CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION}")
    message(STATUS "Detected CUDA version: ${CUDAToolkit_VERSION} (major: ${CUDA_VERSION_MAJOR})")

    if(NOT CUDA_VERSION_MAJOR MATCHES "^(12|13)$")
        message(FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION_MAJOR}. Only CUDA 12 and 13 are supported.")
    endif()
else()
    message(WARNING "Could not detect CUDA version, defaulting to CUDA 12")
    set(CUDA_VERSION_MAJOR "12")
endif()

# Set version-specific output directory
set(CUDA_VERSION_DIR "cu${CUDA_VERSION_MAJOR}")
message(STATUS "Building for CUDA ${CUDA_VERSION_MAJOR}, output directory: ${CUDA_VERSION_DIR}")

set(_cccl_root ../..)

include(${_cccl_root}/cmake/AppendOptionIfAvailable.cmake)
include(${_cccl_root}/cmake/CCCLConfigureTarget.cmake)
include(${_cccl_root}/cmake/CCCLBuildCompilerTargets.cmake)
include(${_cccl_root}/cmake/CCCLGetDependencies.cmake)
cccl_build_compiler_targets()

# Build and install C++ library first
set(CCCL_ENABLE_C ON)
set(CCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY ${SKBUILD_PROJECT_NAME})
add_subdirectory(${_cccl_root} _parent_cccl)

# Now we can find CUB and other components
find_package(CUB REQUIRED)
find_package(Thrust REQUIRED)
find_package(libcudacxx REQUIRED)

# Install headers (shared across all CUDA versions)
set(_dest_incl_dir cuda/cccl/headers/include)

# No end slash: create ${_dest_inc_dir}/cub
install(
    DIRECTORY ${CUB_SOURCE_DIR}/cub
    DESTINATION ${_dest_incl_dir}
)
# No end slash: create ${_dest_inc_dir}/thrust
install(
    DIRECTORY ${Thrust_SOURCE_DIR}/thrust
    DESTINATION ${_dest_incl_dir}
)
# Slash at the end: copy content of
#                   include/ into ${_dest_inc_dir}/
install(
    DIRECTORY ${libcudacxx_SOURCE_DIR}/include/
    DESTINATION ${_dest_incl_dir}
)

# Install version-specific binaries
install(
    TARGETS cccl.c.parallel
    DESTINATION cuda/cccl/parallel/experimental/${CUDA_VERSION_DIR}/cccl
)

# Build and install Cython extension
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

get_filename_component(_python_path "${Python3_EXECUTABLE}" PATH)

set(CYTHON_version_command "${Python3_EXECUTABLE}" -m cython --version)
execute_process(COMMAND ${CYTHON_version_command}
    RESULT_VARIABLE _cython_version_result
    OUTPUT_VARIABLE _cython_version_output
    ERROR_VARIABLE _cython_version_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(NOT _cython_version_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to run cython version command:\n"
        "Command:  ${CYTHON_version_command}\n"
        "Output:   ${_cython_version_output}\n"
        "Error:    ${_cython_version_error}"
    )
endif()

message(STATUS "Cython version: ${_cython_version_output}")

# Create version-specific Cython extension
python3_add_library(_bindings_impl_${CUDA_VERSION_MAJOR} MODULE WITH_SOABI
    cuda/cccl/parallel/experimental/_bindings_impl.pyx
)

set_target_properties(_bindings_impl_${CUDA_VERSION_MAJOR} PROPERTIES
    OUTPUT_NAME "_bindings_impl"
)

target_include_directories(_bindings_impl_${CUDA_VERSION_MAJOR} PRIVATE
    ${_cccl_root}
    ${_cccl_root}/c/parallel/include
)

target_link_libraries(_bindings_impl_${CUDA_VERSION_MAJOR} PRIVATE
    cccl.c.parallel
    CUDA::cudart
)

# Configure compile options for the CUDA version
if(CUDA_VERSION_MAJOR EQUAL 13)
    target_compile_definitions(_bindings_impl_${CUDA_VERSION_MAJOR} PRIVATE
        CCCL_CUDA_VERSION_MAJOR=13
    )
elseif(CUDA_VERSION_MAJOR EQUAL 12)
    target_compile_definitions(_bindings_impl_${CUDA_VERSION_MAJOR} PRIVATE
        CCCL_CUDA_VERSION_MAJOR=12
    )
endif()

# Install version-specific Cython extension
install(
    TARGETS _bindings_impl_${CUDA_VERSION_MAJOR}
    DESTINATION cuda/cccl/parallel/experimental/${CUDA_VERSION_DIR}
)

# Install version detection information
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_version_info.txt.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cuda_version_info.txt"
    @ONLY
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/cuda_version_info.txt"
    DESTINATION cuda/cccl/parallel/experimental/${CUDA_VERSION_DIR}
)
