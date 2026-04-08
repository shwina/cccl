//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <cccl/c/types.h>

namespace cccl::detail
{
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
  }
}

// Parse path arguments from the Python layer for use with clangjit.
// Returns the bare CCCL include path (strips "-I" prefix if present).
inline std::string parse_cccl_include_path(const char* libcudacxx_path)
{
  if (!libcudacxx_path || libcudacxx_path[0] == '\0')
  {
    return {};
  }
  std::string p = libcudacxx_path;
  if (p.substr(0, 2) == "-I")
  {
    p = p.substr(2);
  }
  return p;
}

// Returns the CTK root directory (strips "-I" prefix and "/include" suffix if present).
inline std::string parse_ctk_root(const char* ctk_path)
{
  if (!ctk_path || ctk_path[0] == '\0')
  {
    return {};
  }
  std::string p = ctk_path;
  if (p.substr(0, 2) == "-I")
  {
    p = p.substr(2);
  }
  std::filesystem::path fp(p);
  if (fp.filename() == "include")
  {
    p = fp.parent_path().string();
  }
  return p;
}
} // namespace cccl::detail
