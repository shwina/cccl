cmake_minimum_required(VERSION 3.20)

# --------------------------------------------------------------------------
# LLVM/Clang/LLD — fetched via CPM as static libraries
# --------------------------------------------------------------------------
# CPM.cmake is at the cccl repo root: cccl/cmake/CPM.cmake
# From c/parallel/src/clangjit/ that's ../../../../cmake/CPM.cmake
set(_cccl_cmake_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake")
if (EXISTS "${_cccl_cmake_dir}/CPM.cmake")
  include("${_cccl_cmake_dir}/CPM.cmake")
else()
  message(FATAL_ERROR "CPM.cmake not found at ${_cccl_cmake_dir}/CPM.cmake")
endif()

if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(
    FATAL_ERROR
    "clangjit does not support Debug builds on Windows. "
    "The statically-linked LLVM Debug build is too large and causes stack "
    "overflows at runtime. Use MinSizeRel, Release, or RelWithDebInfo instead."
  )
endif()

set(CLANGJIT_LLVM_VERSION "llvmorg-22.1.1" CACHE STRING "LLVM git tag to fetch")

# List options must be set before CPMAddPackage
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX" CACHE STRING "" FORCE)

CPMAddPackage(
  NAME llvm_project
  GIT_REPOSITORY https://github.com/llvm/llvm-project.git
  GIT_TAG ${CLANGJIT_LLVM_VERSION}
  GIT_SHALLOW ON
  SOURCE_SUBDIR llvm
  EXCLUDE_FROM_ALL YES
  OPTIONS
    "LLVM_BUILD_LLVM_C_DYLIB OFF"
    "LLVM_BUILD_TOOLS OFF"
    "LLVM_BUILD_UTILS OFF"
    "LLVM_BUILD_RUNTIME OFF"
    "LLVM_BUILD_RUNTIMES OFF"
    "LLVM_INCLUDE_BENCHMARKS OFF"
    "LLVM_INCLUDE_DOCS OFF"
    "LLVM_INCLUDE_EXAMPLES OFF"
    "LLVM_INCLUDE_RUNTIMES OFF"
    "LLVM_INCLUDE_TESTS OFF"
    "LLVM_INCLUDE_TOOLS ON"
    "LLVM_INCLUDE_UTILS OFF"
    "LLVM_ENABLE_ZLIB OFF"
    "LLVM_ENABLE_ZSTD OFF"
    "LLVM_ENABLE_TERMINFO OFF"
    "LLVM_ENABLE_BINDINGS OFF"
    "CLANG_BUILD_TOOLS OFF"
    "CLANG_ENABLE_ARCMT OFF"
    "CLANG_ENABLE_STATIC_ANALYZER OFF"
)

# Ensure the clang resource directory exists
file(
  MAKE_DIRECTORY "${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
)

# Find CUDA toolkit (may already be found by parent)
if (NOT CUDAToolkit_FOUND)
  find_package(CUDAToolkit)
endif()

# --------------------------------------------------------------------------
# clangjit library
# --------------------------------------------------------------------------
add_library(
  clangjit_lib
  compiler.cpp
  config.cpp
  loader.cpp
  jit_compiler.cpp
  codegen/types.cpp
  codegen/iterators.cpp
  codegen/operators.cpp
  codegen/bitcode.cpp
  codegen/cub_call.cpp
)

# CCCL_SOURCE_DIR points to the cccl repo root
# From c/parallel/src/clangjit -> c/parallel/src -> c/parallel -> c -> cccl
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH _src_dir) # c/parallel/src
cmake_path(GET _src_dir PARENT_PATH _c_parallel_dir) # c/parallel
cmake_path(GET _c_parallel_dir PARENT_PATH _c_dir) # c
cmake_path(GET _c_dir PARENT_PATH _cccl_root) # cccl

target_include_directories(
  clangjit_lib
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_c_parallel_dir}/include
    ${llvm_project_SOURCE_DIR}/llvm/include
    ${llvm_project_BINARY_DIR}/include
    ${llvm_project_SOURCE_DIR}/clang/include
    ${llvm_project_BINARY_DIR}/tools/clang/include
    ${llvm_project_SOURCE_DIR}/lld/include
    ${llvm_project_BINARY_DIR}/tools/lld/include
)

target_compile_definitions(
  clangjit_lib
  PRIVATE
    CCCL_C_EXPERIMENTAL=1
    CCCL_SOURCE_DIR="${_cccl_root}"
    CLANG_RESOURCE_DIR="${llvm_project_BINARY_DIR}/lib/clang/${LLVM_VERSION_MAJOR}"
    CLANG_HEADERS_DIR="${llvm_project_SOURCE_DIR}/clang/lib/Headers"
    CLANGJIT_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if (CUDAToolkit_FOUND)
  target_include_directories(clangjit_lib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH CUDA_TOOLKIT_ROOT_FROM_CMAKE)
  target_compile_definitions(
    clangjit_lib
    PRIVATE
      CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_ROOT_FROM_CMAKE}"
      CUDA_SDK_VERSION="${CUDAToolkit_VERSION_MAJOR}.0"
  )
endif()

# Link against LLVM/Clang/LLD
target_link_libraries(
  clangjit_lib
  PUBLIC
    # LLVM
    LLVMCore
    LLVMSupport
    LLVMIRReader
    LLVMMC
    LLVMObject
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86Info
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMLinker
    LLVMPasses
    # Clang
    clangAST
    clangBasic
    clangCodeGen
    clangDriver
    clangFrontend
    clangFrontendTool
    clangLex
    clangParse
    clangSema
    clangEdit
    clangAnalysis
    clangRewrite
    clangSerialization
    # LLD
    $<IF:$<PLATFORM_ID:Windows>,lldCOFF,lldELF>
    lldCommon
)

if (NOT WIN32)
  target_link_libraries(clangjit_lib PUBLIC dl)
endif()

if (CUDAToolkit_FOUND)
  target_link_libraries(clangjit_lib PUBLIC CUDA::cuda_driver CUDA::cudart)
  if (WIN32)
    # On Windows, static CUDA libs are built with /MT which conflicts with
    # the project's dynamic CRT (/MD). Use dynamic variants instead.
    target_link_libraries(clangjit_lib PUBLIC CUDA::nvJitLink CUDA::nvfatbin)
  else()
    target_link_libraries(
      clangjit_lib
      PUBLIC
        CUDA::nvJitLink_static
        CUDA::nvptxcompiler_static
        CUDA::nvfatbin_static
    )
  endif()
endif()

if (NOT MSVC)
  target_compile_options(clangjit_lib PRIVATE -fno-rtti)
endif()

set_target_properties(
  clangjit_lib
  PROPERTIES CXX_STANDARD 20 POSITION_INDEPENDENT_CODE ON
)

# --------------------------------------------------------------------------
# Install clang headers into wheel (for self-sufficient packaging)
# --------------------------------------------------------------------------
# Clang CUDA headers we still use from the LLVM source tree.
# We DON'T install device_functions, math, or libdevice_declares — our local
# copies in cuda_minimal/ replace them.
set(
  _clang_cuda_headers_needed
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_math_forward_declares.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_builtin_vars.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_cmath.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_intrinsics.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_complex_builtins.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__clang_cuda_texture_intrinsics.h"
)
install(
  FILES ${_clang_cuda_headers_needed}
  DESTINATION "cuda/cccl/headers/clang"
)

# Clang builtin C headers needed by our stubs and CUDA toolkit headers.
file(
  GLOB _clang_stddef_headers
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__stddef_*.h"
)
set(
  _clang_c_headers
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/limits.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/stddef.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/stdint.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__stddef_header_macro.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/float.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/__float_header_macro.h"
  "${llvm_project_SOURCE_DIR}/clang/lib/Headers/inttypes.h"
  ${_clang_stddef_headers}
)
install(FILES ${_clang_c_headers} DESTINATION "cuda/cccl/headers/clang")

# Clangjit's minimal CUDA runtime headers (replacements for upstream clang headers)
set(
  _clangjit_cuda_minimal_dir
  "${CMAKE_CURRENT_SOURCE_DIR}/include/clangjit/cuda_minimal"
)
file(GLOB _clangjit_cuda_minimal_headers "${_clangjit_cuda_minimal_dir}/*.h")
install(
  FILES ${_clangjit_cuda_minimal_headers}
  DESTINATION "cuda/cccl/headers/clangjit/cuda_minimal"
)

# Clangjit's stub headers (minimal C++ standard library stubs for device compilation)
# Use GLOB_RECURSE + DIRECTORY so subdirectory overrides (e.g. cuda/std/__cstdlib/)
# are also installed alongside the top-level stubs.
install(
  DIRECTORY "${_clangjit_cuda_minimal_dir}/stubs/"
  DESTINATION "cuda/cccl/headers/clangjit/cuda_minimal/stubs"
)

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
add_executable(
  example_cub_reduce
  ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/cub_reduce.cpp
)
target_link_libraries(example_cub_reduce clangjit_lib cccl.c.parallel)
if (NOT MSVC)
  target_compile_options(example_cub_reduce PRIVATE -fno-rtti)
endif()
set_target_properties(example_cub_reduce PROPERTIES CXX_STANDARD 20)

add_executable(
  example_cub_reduce_deterministic
  ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/cub_reduce_deterministic.cpp
)
target_link_libraries(example_cub_reduce_deterministic clangjit_lib)
if (NOT MSVC)
  target_compile_options(example_cub_reduce_deterministic PRIVATE -fno-rtti)
endif()
set_target_properties(
  example_cub_reduce_deterministic
  PROPERTIES CXX_STANDARD 20
)

add_executable(
  example_cub_reduce_custom_op
  ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/cub_reduce_custom_op.cpp
)
target_link_libraries(example_cub_reduce_custom_op clangjit_lib)
if (NOT MSVC)
  target_compile_options(example_cub_reduce_custom_op PRIVATE -fno-rtti)
endif()
set_target_properties(example_cub_reduce_custom_op PROPERTIES CXX_STANDARD 20)

add_executable(
  example_cub_adjacent_difference
  ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/cub_adjacent_difference.cpp
)
target_link_libraries(example_cub_adjacent_difference clangjit_lib)
if (NOT MSVC)
  target_compile_options(example_cub_adjacent_difference PRIVATE -fno-rtti)
endif()
set_target_properties(
  example_cub_adjacent_difference
  PROPERTIES CXX_STANDARD 20
)

# On Windows with multi-config generators (Visual Studio), exclude clangjit
# targets from Debug builds — the LLVM Debug build causes stack overflows.
if (MSVC)
  set_target_properties(
    clangjit_lib
    example_cub_reduce
    example_cub_reduce_deterministic
    example_cub_reduce_custom_op
    example_cub_adjacent_difference
    PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD_DEBUG TRUE
  )
endif()
