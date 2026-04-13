//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>

#include <cccl/c/types.h>

namespace cccl::detail
{
// Stable strings for NVRTC PCH flags — must not be temporaries since
// NVRTC holds pointers to them for the lifetime of the compilation.
inline const char* nvrtc_pch_flag()
{
  static const char flag[] = "-pch";
  return flag;
}

inline const char* nvrtc_pch_dir_flag()
{
  static const char flag[] = "-pch-dir=/tmp/nvrtc_pch";
  return flag;
}

/**
 * @brief Extends a vector of compilation arguments with extra flags and include directories from a build config
 *
 * @param args The vector of arguments to extend
 * @param config The build configuration containing extra flags and include directories (can be nullptr)
 */
inline void extend_args_with_build_config(std::vector<const char*>& args, const cccl_build_config* config)
{
  if (config)
  {
    // Add extra compile flags
    for (size_t i = 0; i < config->num_extra_compile_flags; ++i)
    {
      args.push_back(config->extra_compile_flags[i]);
    }
    // Add include directories
    for (size_t i = 0; i < config->num_extra_include_dirs; ++i)
    {
      args.push_back("-I");
      args.push_back(config->extra_include_dirs[i]);
    }
    // Enable NVRTC automatic PCH mode: NVRTC will generate and reuse a PCH
    // for the common preamble headers within this process's NVRTC instance.
    if (config->enable_pch)
    {
      args.push_back(nvrtc_pch_flag());
      args.push_back(nvrtc_pch_dir_flag());
    }
  }
}
} // namespace cccl::detail
