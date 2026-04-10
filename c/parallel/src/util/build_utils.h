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

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cccl/c/types.h>
#include <clangjit/config.hpp>
#include <clangjit/jit_compiler.hpp>

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

// Build a CompilerConfig from the standard set of path parameters.
// Mirrors the configuration logic in CubCall::compile().
inline clangjit::CompilerConfig make_jit_config(
  int cc_major,
  int cc_minor,
  const char* clang_path,
  const char* ctk_root, // already parsed (bare CTK root)
  const char* cccl_include_path, // already parsed (bare CCCL include path)
  cccl_build_config* config,
  const char* entry_point_name = nullptr)
{
  auto jit_config       = clangjit::detectDefaultConfig();
  jit_config.sm_version = cc_major * 10 + cc_minor;
  jit_config.verbose    = false;
  if (entry_point_name)
  {
    jit_config.entry_point_name = entry_point_name;
  }
  if (clang_path && clang_path[0] != '\0')
  {
    jit_config.clang_headers_path = clang_path;
  }
  if (ctk_root && ctk_root[0] != '\0')
  {
    jit_config.cuda_toolkit_path = ctk_root;
    jit_config.library_paths.clear();
    for (const char* subdir : {"lib64", "lib"})
    {
      auto candidate = std::filesystem::path(ctk_root) / subdir;
      if (std::filesystem::exists(candidate))
      {
        jit_config.library_paths.push_back(candidate.string());
      }
    }
  }
  if (cccl_include_path && cccl_include_path[0] != '\0')
  {
    jit_config.cccl_include_path = cccl_include_path;
    if (jit_config.clangjit_include_path.empty()
        || !std::filesystem::exists(jit_config.clangjit_include_path + "/clangjit/cuda_minimal"))
    {
      auto parent = std::filesystem::path(cccl_include_path).parent_path().string();
      if (std::filesystem::exists(parent + "/clangjit/cuda_minimal"))
      {
        jit_config.clangjit_include_path = parent;
      }
    }
  }
  if (config)
  {
    for (size_t i = 0; i < config->num_extra_include_dirs; ++i)
    {
      jit_config.include_paths.push_back(config->extra_include_dirs[i]);
    }
    for (size_t i = 0; i < config->num_extra_compile_flags; ++i)
    {
      std::string flag = config->extra_compile_flags[i];
      if (flag.size() >= 2 && flag.substr(0, 2) == "-D")
      {
        auto eq = flag.find('=', 2);
        if (eq != std::string::npos)
        {
          jit_config.macro_definitions[flag.substr(2, eq - 2)] = flag.substr(eq + 1);
        }
        else
        {
          jit_config.macro_definitions[flag.substr(2)] = "";
        }
      }
    }
  }
  return jit_config;
}

// Build a JITCompiler from the standard set of path parameters.
inline clangjit::JITCompiler* make_jit_compiler(
  int cc_major,
  int cc_minor,
  const char* clang_path,
  const char* ctk_root,
  const char* cccl_include_path,
  cccl_build_config* config,
  const char* entry_point_name = nullptr)
{
  return new clangjit::JITCompiler(
    make_jit_config(cc_major, cc_minor, clang_path, ctk_root, cccl_include_path, config, entry_point_name));
}

// Compile a CUDA source string and return (compiler, fn_ptr, cubin).
// On failure, deletes the compiler and returns {nullptr, nullptr, {}}.
struct JITResult
{
  clangjit::JITCompiler* compiler = nullptr;
  void* fn_ptr                    = nullptr;
  std::vector<char> cubin;
};

inline JITResult compile_jit_source(
  const std::string& source,
  const char* fn_name,
  int cc_major,
  int cc_minor,
  const char* clang_path,
  const char* ctk_root,
  const char* cccl_include_path,
  cccl_build_config* config)
{
  auto* compiler = make_jit_compiler(cc_major, cc_minor, clang_path, ctk_root, cccl_include_path, config, fn_name);
  if (!compiler->compile(source))
  {
    fprintf(stderr, "\nJIT compilation failed: %s\n", compiler->getLastError().c_str());
    delete compiler;
    return {};
  }
  void* fn_ptr = compiler->getFunction<void*>(fn_name);
  if (!fn_ptr)
  {
    fprintf(stderr, "\nJIT symbol lookup failed for '%s': %s\n", fn_name, compiler->getLastError().c_str());
    delete compiler;
    return {};
  }
  JITResult result;
  result.compiler = compiler;
  result.fn_ptr   = fn_ptr;
  result.cubin    = compiler->getCubin();
  return result;
}

// Copy cubin data into a heap-allocated buffer and store size; returns pointer (caller frees with delete[]).
inline void* copy_cubin(const std::vector<char>& cubin, size_t* out_size)
{
  if (cubin.empty())
  {
    if (out_size)
    {
      *out_size = 0;
    }
    return nullptr;
  }
  auto* buf = new char[cubin.size()];
  std::memcpy(buf, cubin.data(), cubin.size());
  if (out_size)
  {
    *out_size = cubin.size();
  }
  return buf;
}
} // namespace cccl::detail
