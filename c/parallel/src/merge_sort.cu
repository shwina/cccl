//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>

#include <cccl/c/merge_sort.h>
#include <clangjit/codegen/bitcode.hpp>
#include <clangjit/codegen/iterators.hpp>
#include <clangjit/codegen/operators.hpp>
#include <clangjit/codegen/types.hpp>
#include <clangjit/config.hpp>
#include <clangjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace clangjit;
using namespace clangjit::codegen;

// All builds use the same 9-arg function signature.
// Keys-only builds ignore d_in_items / d_out_items (passed as nullptr).
using sort_fn_t = int (*)(void*, size_t*, void*, void*, void*, void*, unsigned long long, void*, void*);

static bool is_null_items(cccl_iterator_t it)
{
  return it.type == CCCL_POINTER && it.state == nullptr;
}

static CompilerConfig make_merge_sort_jit_config(
  const char* entry_name,
  int cc_major,
  int cc_minor,
  const char* clang_path,
  cccl_build_config* config,
  const char* ctk_root,
  const char* cccl_include_path)
{
  auto jit_config             = detectDefaultConfig();
  jit_config.sm_version       = cc_major * 10 + cc_minor;
  jit_config.verbose          = false;
  jit_config.entry_point_name = entry_name;

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
      if (flag.substr(0, 2) == "-D")
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

// ---------------------------------------------------------------------------
// Source generation
// ---------------------------------------------------------------------------

static std::string make_merge_sort_source(cccl_iterator_t d_in_keys, cccl_iterator_t d_in_items, cccl_op_t op)
{
  const bool has_items = !is_null_items(d_in_items);

  // Resolve key type (uses storage_t struct for unknown types)
  std::string key_preamble;
  std::string key_type = resolve_type(d_in_keys.value_type, "key_t", key_preamble);

  // Resolve item type
  std::string item_preamble;
  std::string item_type;
  if (has_items)
  {
    item_type = resolve_type(d_in_items.value_type, "item_t", item_preamble);
  }

  // Key input iterator (pointer or custom struct with state)
  auto key_in_code = make_input_iterator(d_in_keys, key_type, key_type, "in_keys_t", "in_keys", "d_in_keys");

  // Item input iterator
  IteratorCode item_in_code;
  if (has_items)
  {
    item_in_code = make_input_iterator(d_in_items, item_type, item_type, "in_items_t", "in_items", "d_in_items");
  }

  // Comparison operator (returns bool)
  const bool has_bc = BitcodeCollector::is_bitcode_op(op);
  auto cmp_code     = make_comparison_op(op, key_type, "cmp_op_t", "cmp_op", "op_state", has_bc);

  std::string src;
  src += "#include <cuda_runtime.h>\n";
  src += "#include <cuda_fp16.h>\n";
  src += "#include <cuda/std/iterator>\n";
  src += "#include <cuda/std/functional>\n";
  src += "#include <cuda/functional>\n";
  src += "#include <cub/device/device_merge_sort.cuh>\n\n";

  src += key_preamble;
  if (has_items)
  {
    src += item_preamble;
  }
  src += key_in_code.preamble;
  if (has_items)
  {
    src += item_in_code.preamble;
  }
  src += cmp_code.preamble;

  src += "#ifdef _WIN32\n"
         "#define EXPORT __declspec(dllexport)\n"
         "#else\n"
         "#define EXPORT __attribute__((visibility(\"default\")))\n"
         "#endif\n\n";

  // Unified 9-arg signature for both keys-only and pairs builds.
  // Keys-only: d_in_items / d_out_items are passed as nullptr and ignored.
  src +=
    "extern \"C\" EXPORT int cccl_jit_merge_sort(\n"
    "    void* d_temp_storage,\n"
    "    size_t* temp_storage_bytes,\n"
    "    void* d_in_keys,\n"
    "    void* d_in_items,\n"
    "    void* d_out_keys,\n"
    "    void* d_out_items,\n"
    "    unsigned long long num_items,\n"
    "    void* op_state,\n"
    "    void* stream)\n"
    "{\n";

  // Set up key input iterator
  src += "    " + key_in_code.setup_code + "\n";

  // Output key pointer (merge sort output is always a raw pointer)
  src += std::format("    {}* out_keys = static_cast<{}*>(d_out_keys);\n", key_type, key_type);

  if (has_items)
  {
    src += "    " + item_in_code.setup_code + "\n";
    src += std::format("    {}* out_items = static_cast<{}*>(d_out_items);\n", item_type, item_type);
  }
  else
  {
    src += "    (void)d_in_items;\n";
    src += "    (void)d_out_items;\n";
  }

  // Set up comparison op
  src += "    " + cmp_code.setup_code + "\n\n";

  if (has_items)
  {
    src += "    cudaError_t err = cub::DeviceMergeSort::SortPairsCopy(\n"
           "        d_temp_storage, *temp_storage_bytes,\n"
           "        in_keys, in_items, out_keys, out_items,\n"
           "        (unsigned long long)num_items, cmp_op,\n"
           "        (cudaStream_t)stream);\n";
  }
  else
  {
    src += "    cudaError_t err = cub::DeviceMergeSort::SortKeysCopy(\n"
           "        d_temp_storage, *temp_storage_bytes,\n"
           "        in_keys, out_keys,\n"
           "        (unsigned long long)num_items, cmp_op,\n"
           "        (cudaStream_t)stream);\n";
  }

  src += "    return (int)err;\n}\n";

  return src;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort_build_ex(
  cccl_device_merge_sort_build_result_t* build_ptr,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* /*cub_path*/,
  const char* /*thrust_path*/,
  const char* libcudacxx_path,
  const char* ctk_path,
  const char* clang_path,
  cccl_build_config* config)
try
{
  if (d_out_keys.type == CCCL_ITERATOR || d_out_items.type == CCCL_ITERATOR)
  {
    fprintf(stderr, "\nERROR in cccl_device_merge_sort_build(): merge sort output cannot be an iterator\n");
    return CUDA_ERROR_UNKNOWN;
  }

  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();

  auto jit_config = make_merge_sort_jit_config(
    "cccl_jit_merge_sort", cc_major, cc_minor, clang_path, config, ctk_root, cccl_include_path);

  uintptr_t unique_id = reinterpret_cast<uintptr_t>(build_ptr);
  BitcodeCollector bitcode(jit_config, unique_id);
  bitcode.add_op(op, "compare_op");
  bitcode.add_iterator(d_in_keys, "in_keys");
  if (!is_null_items(d_in_items))
  {
    bitcode.add_iterator(d_in_items, "in_items");
  }

  std::string cuda_source = make_merge_sort_source(d_in_keys, d_in_items, op);

  auto* compiler = new JITCompiler(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    delete compiler;
    bitcode.cleanup();
    throw std::runtime_error("merge_sort compilation failed: " + err);
  }
  bitcode.cleanup();

  auto fn = compiler->getFunction<sort_fn_t>("cccl_jit_merge_sort");
  if (!fn)
  {
    std::string err = compiler->getLastError();
    delete compiler;
    throw std::runtime_error("merge_sort function lookup failed: " + err);
  }

  auto cubin = compiler->getCubin();

  build_ptr->cc         = cc_major * 10 + cc_minor;
  build_ptr->cubin      = nullptr;
  build_ptr->cubin_size = 0;
  if (!cubin.empty())
  {
    auto* cubin_copy = new char[cubin.size()];
    std::memcpy(cubin_copy, cubin.data(), cubin.size());
    build_ptr->cubin      = cubin_copy;
    build_ptr->cubin_size = cubin.size();
  }
  build_ptr->jit_compiler = compiler;
  build_ptr->sort_fn      = reinterpret_cast<void*>(fn);
  build_ptr->key_type     = d_in_keys.value_type;
  build_ptr->item_type    = d_in_items.value_type;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_merge_sort_build(
  cccl_device_merge_sort_build_result_t* build,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  const char* clang_path)
{
  return cccl_device_merge_sort_build_ex(
    build,
    d_in_keys,
    d_in_items,
    d_out_keys,
    d_out_items,
    op,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    clang_path,
    nullptr);
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort(
  cccl_device_merge_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in_keys,
  cccl_iterator_t d_in_items,
  cccl_iterator_t d_out_keys,
  cccl_iterator_t d_out_items,
  uint64_t num_items,
  cccl_op_t op,
  CUstream stream)
{
  try
  {
    auto sort_fn = reinterpret_cast<sort_fn_t>(build.sort_fn);
    if (!sort_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    // Parameter order matches the generated function signature:
    // (temp, temp_bytes, in_keys, in_items, out_keys, out_items, num_items, op_state, stream)
    int status = sort_fn(
      d_temp_storage,
      temp_storage_bytes,
      d_in_keys.state,
      d_in_items.state,
      d_out_keys.state,
      d_out_items.state,
      num_items,
      op.state,
      reinterpret_cast<void*>(stream));

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_merge_sort_cleanup(cccl_device_merge_sort_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (build_ptr->jit_compiler)
  {
    delete static_cast<JITCompiler*>(build_ptr->jit_compiler);
    build_ptr->jit_compiler = nullptr;
  }
  if (build_ptr->cubin)
  {
    delete[] static_cast<char*>(build_ptr->cubin);
    build_ptr->cubin = nullptr;
  }
  build_ptr->cubin_size = 0;
  build_ptr->sort_fn    = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_merge_sort_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
