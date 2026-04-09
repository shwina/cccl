//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
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

#include <cccl/c/transform.h>
#include <clangjit/codegen/bitcode.hpp>
#include <clangjit/codegen/iterators.hpp>
#include <clangjit/codegen/operators.hpp>
#include <clangjit/codegen/types.hpp>
#include <clangjit/config.hpp>
#include <clangjit/jit_compiler.hpp>
#include <util/build_utils.h>

using namespace clangjit;
using namespace clangjit::codegen;

using unary_transform_fn_t  = int (*)(void*, void*, unsigned long long, void*);
using binary_transform_fn_t = int (*)(void*, void*, void*, unsigned long long, void*);

// ---------------------------------------------------------------------------
// Unary transform: custom source generation
// ---------------------------------------------------------------------------

static std::string make_unary_transform_source(cccl_iterator_t d_in, cccl_iterator_t d_out, cccl_op_t op)
{
  const auto in_type        = get_type_name(d_in.value_type.type);
  const auto out_type       = get_type_name(d_out.value_type.type);
  const bool has_bc         = BitcodeCollector::is_bitcode_op(op);
  const bool stateful       = (op.type == CCCL_STATEFUL);
  const std::string op_name = (op.name && op.name[0]) ? op.name : "op";

  std::string in_storage_preamble, out_storage_preamble;
  const std::string in_type_actual  = in_type.empty() ? "in_storage_t" : in_type;
  const std::string out_type_actual = out_type.empty() ? "out_storage_t" : out_type;
  if (in_type.empty())
  {
    in_storage_preamble = make_storage_type("in_storage_t", d_in.value_type.size, d_in.value_type.alignment);
  }
  if (out_type.empty())
  {
    out_storage_preamble = make_storage_type("out_storage_t", d_out.value_type.size, d_out.value_type.alignment);
  }

  auto in_code  = make_input_iterator(d_in, in_type, in_type_actual, "in_0_it_t", "in_0", "d_in_0");
  auto out_code = make_output_iterator(d_out, out_type_actual, "out_0_it_t", "out_0", "d_out_0");

  std::string src;
  src += "#include <cuda_runtime.h>\n";
  src += "#include <cuda_fp16.h>\n";
  src += "#include <cuda/std/iterator>\n";
  src += "#include <cuda/std/functional>\n";
  src += "#include <cuda/functional>\n";
  src += "#include <cub/device/device_transform.cuh>\n\n";

  src += "#ifdef _WIN32\n"
         "#define EXPORT __declspec(dllexport)\n"
         "#else\n"
         "#define EXPORT __attribute__((visibility(\"default\")))\n"
         "#endif\n\n";

  if (!in_storage_preamble.empty())
  {
    src += in_storage_preamble;
  }
  if (!out_storage_preamble.empty())
  {
    src += out_storage_preamble;
  }

  src += in_code.preamble;
  src += out_code.preamble;

  const bool is_negate     = (op.type == CCCL_NEGATE);
  const bool is_identity   = (op.type == CCCL_IDENTITY);
  const bool is_well_known = is_negate || is_identity;

  if (!is_well_known)
  {
    if (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0)
    {
      src += std::string(op.code, op.code_size);
      src += "\n\n";
    }
    else if (has_bc)
    {
      if (stateful)
      {
        src += std::format("extern \"C\" __device__ void {}(void* state, void* a_ptr, void* result_ptr);\n\n", op_name);
      }
      else
      {
        src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* result_ptr);\n\n", op_name);
      }
    }
  }

  // Generate unary functor
  if (is_negate)
  {
    src += "using UnaryOp_0 = ::cuda::std::negate<>;\n\n";
  }
  else if (is_identity)
  {
    src += "using UnaryOp_0 = ::cuda::std::identity;\n\n";
  }
  else if (stateful)
  {
    src += std::format(
      "struct UnaryOp_0 {{\n"
      "  void* state;\n"
      "  __device__ __forceinline__\n"
      "  {} operator()(const {}& a) const {{\n"
      "    {} result;\n"
      "    {}(state, (void*)&a, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      out_type_actual,
      in_type_actual,
      out_type_actual,
      op_name);
  }
  else
  {
    src += std::format(
      "struct UnaryOp_0 {{\n"
      "  __device__ __forceinline__\n"
      "  {} operator()(const {}& a) const {{\n"
      "    {} result;\n"
      "    {}((void*)&a, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      out_type_actual,
      in_type_actual,
      out_type_actual,
      op_name);
  }

  src += "extern \"C\" EXPORT int cccl_jit_unary_transform(\n"
         "    void* d_in_0, void* d_out_0, unsigned long long num_items, void* op_0_state\n"
         ") {\n";
  src += "    " + in_code.setup_code + "\n";
  src += "    " + out_code.setup_code + "\n";
  if (is_well_known || !stateful)
  {
    src += "    UnaryOp_0 op_0{};\n";
  }
  else
  {
    src += "    UnaryOp_0 op_0{op_0_state};\n";
  }
  src += "    cudaError_t err = cub::DeviceTransform::Transform(in_0, out_0, (unsigned long long)num_items, op_0);\n";
  src += "    return (int)err;\n"
         "}\n";

  return src;
}

// ---------------------------------------------------------------------------
// Binary transform: custom source generation (tuple API)
// ---------------------------------------------------------------------------

static std::string
make_binary_transform_source(cccl_iterator_t d_in1, cccl_iterator_t d_in2, cccl_iterator_t d_out, cccl_op_t op)
{
  const auto in1_type = get_type_name(d_in1.value_type.type);
  const auto in2_type = get_type_name(d_in2.value_type.type);
  const auto out_type = get_type_name(d_out.value_type.type);
  const bool has_bc   = BitcodeCollector::is_bitcode_op(op);

  // For custom output types, emit a storage struct; for known types, use a type alias.
  std::string accum_preamble;
  std::string accum_type;
  if (out_type.empty())
  {
    accum_preamble = make_storage_type("accum_t", d_out.value_type.size, d_out.value_type.alignment);
    accum_type     = "accum_t";
  }
  else
  {
    accum_type = out_type;
  }

  auto in0_code = make_input_iterator(d_in1, in1_type, accum_type, "in_0_it_t", "in_0", "d_in_0");
  auto in1_code = make_input_iterator(d_in2, in2_type, accum_type, "in_1_it_t", "in_1", "d_in_1");
  auto out_code = make_output_iterator(d_out, accum_type, "out_0_it_t", "out_0", "d_out_0");
  auto op_code  = make_binary_op(op, accum_type, "Op_0", "op_0", "op_0_state", has_bc);

  std::string src;
  src += "#include <cuda_runtime.h>\n";
  src += "#include <cuda_fp16.h>\n";
  src += "#include <cuda/std/iterator>\n";
  src += "#include <cuda/std/functional>\n";
  src += "#include <cuda/functional>\n";
  src += "#include <cuda/std/tuple>\n";
  src += "#include <cub/device/device_transform.cuh>\n\n";

  src += "#ifdef _WIN32\n"
         "#define EXPORT __declspec(dllexport)\n"
         "#else\n"
         "#define EXPORT __attribute__((visibility(\"default\")))\n"
         "#endif\n\n";

  // Emit accum_t definition (storage struct for custom types, type alias otherwise)
  if (!accum_preamble.empty())
  {
    src += accum_preamble;
  }
  else
  {
    src += std::format("using accum_t = {};\n\n", accum_type);
  }

  src += in0_code.preamble;
  src += in1_code.preamble;
  src += out_code.preamble;
  src += op_code.preamble;

  src += "extern \"C\" EXPORT int cccl_jit_binary_transform(\n"
         "    void* d_in_0, void* d_in_1, void* d_out_0, unsigned long long num_items, void* op_0_state\n"
         ") {\n";
  src += "    " + in0_code.setup_code + "\n";
  src += "    " + in1_code.setup_code + "\n";
  src += "    " + out_code.setup_code + "\n";
  src += "    " + op_code.setup_code + "\n";
  src += "    cudaError_t err = cub::DeviceTransform::Transform(\n"
         "        ::cuda::std::make_tuple(in_0, in_1), out_0, (unsigned long long)num_items, op_0);\n";
  src += "    return (int)err;\n"
         "}\n";

  return src;
}

// Set up JITCompiler config for transform — mirrors binary_search.cu pattern
static CompilerConfig make_transform_jit_config(
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
// Build functions
// ---------------------------------------------------------------------------

CUresult cccl_device_unary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
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
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();

  auto jit_config = make_transform_jit_config(
    "cccl_jit_unary_transform", cc_major, cc_minor, clang_path, config, ctk_root, cccl_include_path);

  // Collect bitcode from op and iterators
  uintptr_t unique_id = reinterpret_cast<uintptr_t>(build_ptr);
  BitcodeCollector bitcode(jit_config, unique_id);
  bitcode.add_op(op, "op_0");
  bitcode.add_iterator(d_in, "in_0");
  bitcode.add_iterator(d_out, "out_0");

  // Generate source
  std::string cuda_source = make_unary_transform_source(d_in, d_out, op);

  // Compile
  auto* compiler = new JITCompiler(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    delete compiler;
    bitcode.cleanup();
    throw std::runtime_error("unary_transform compilation failed: " + err);
  }
  bitcode.cleanup();

  // Extract function pointer
  using fn_t = int (*)(void*, ...);
  auto fn    = compiler->getFunction<fn_t>("cccl_jit_unary_transform");
  if (!fn)
  {
    std::string err = compiler->getLastError();
    delete compiler;
    throw std::runtime_error("unary_transform function lookup failed: " + err);
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
  build_ptr->transform_fn = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_unary_transform_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_binary_transform_build_ex(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
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
  std::string cccl_include_str  = cccl::detail::parse_cccl_include_path(libcudacxx_path);
  std::string ctk_root_str      = cccl::detail::parse_ctk_root(ctk_path);
  const char* cccl_include_path = cccl_include_str.empty() ? nullptr : cccl_include_str.c_str();
  const char* ctk_root          = ctk_root_str.empty() ? nullptr : ctk_root_str.c_str();

  auto jit_config = make_transform_jit_config(
    "cccl_jit_binary_transform", cc_major, cc_minor, clang_path, config, ctk_root, cccl_include_path);

  // Collect bitcode from op and iterators
  uintptr_t unique_id = reinterpret_cast<uintptr_t>(build_ptr);
  BitcodeCollector bitcode(jit_config, unique_id);
  bitcode.add_op(op, "op_0");
  bitcode.add_iterator(d_in1, "in_0");
  bitcode.add_iterator(d_in2, "in_1");
  bitcode.add_iterator(d_out, "out_0");

  // Generate source
  std::string cuda_source = make_binary_transform_source(d_in1, d_in2, d_out, op);

  // Compile
  auto* compiler = new JITCompiler(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    delete compiler;
    bitcode.cleanup();
    throw std::runtime_error("binary_transform compilation failed: " + err);
  }
  bitcode.cleanup();

  // Extract function pointer
  using fn_t = int (*)(void*, ...);
  auto fn    = compiler->getFunction<fn_t>("cccl_jit_binary_transform");
  if (!fn)
  {
    std::string err = compiler->getLastError();
    delete compiler;
    throw std::runtime_error("binary_transform function lookup failed: " + err);
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
  build_ptr->transform_fn = reinterpret_cast<void*>(fn);

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_binary_transform_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

// ---------------------------------------------------------------------------
// Non-ex wrappers (call _ex with nullptr config)
// ---------------------------------------------------------------------------

CUresult cccl_device_unary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  const char* clang_path)
{
  return cccl_device_unary_transform_build_ex(
    build_ptr,
    d_in,
    d_out,
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

CUresult cccl_device_binary_transform_build(
  cccl_device_transform_build_result_t* build_ptr,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  cccl_op_t op,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  const char* clang_path)
{
  return cccl_device_binary_transform_build_ex(
    build_ptr,
    d_in1,
    d_in2,
    d_out,
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
// Runtime functions
// ---------------------------------------------------------------------------

CUresult cccl_device_unary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream /*stream*/)
{
  try
  {
    auto fn = reinterpret_cast<unary_transform_fn_t>(build.transform_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    int status = fn(d_in.state, d_out.state, num_items, op.state);
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_unary_transform(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_binary_transform(
  cccl_device_transform_build_result_t build,
  cccl_iterator_t d_in1,
  cccl_iterator_t d_in2,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  CUstream /*stream*/)
{
  try
  {
    auto fn = reinterpret_cast<binary_transform_fn_t>(build.transform_fn);
    if (!fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }
    int status = fn(d_in1.state, d_in2.state, d_out.state, num_items, op.state);
    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_binary_transform(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

CUresult cccl_device_transform_cleanup(cccl_device_transform_build_result_t* build_ptr)
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
  build_ptr->cubin_size   = 0;
  build_ptr->transform_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_transform_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}
