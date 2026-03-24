//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cccl/c/reduce.h>

#include <clangjit/config.hpp>
#include <clangjit/jit_compiler.hpp>

#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>

namespace
{

std::string get_cpp_type_name(cccl_type_enum type)
{
  switch (type)
  {
    case CCCL_INT8:
      return "char";
    case CCCL_INT16:
      return "short";
    case CCCL_INT32:
      return "int";
    case CCCL_INT64:
      return "long long";
    case CCCL_UINT8:
      return "unsigned char";
    case CCCL_UINT16:
      return "unsigned short";
    case CCCL_UINT32:
      return "unsigned int";
    case CCCL_UINT64:
      return "unsigned long long";
    case CCCL_FLOAT32:
      return "float";
    case CCCL_FLOAT64:
      return "double";
    default:
      return "";
  }
}

std::string get_well_known_op_body(cccl_op_kind_t kind, const std::string& type_name)
{
  switch (kind)
  {
    case CCCL_PLUS:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
        "    *out = *a + *b;\n",
        type_name);
    case CCCL_MINIMUM:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
        "    *out = (*a < *b) ? *a : *b;\n",
        type_name);
    case CCCL_MAXIMUM:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
        "    *out = (*a > *b) ? *a : *b;\n",
        type_name);
    case CCCL_BIT_AND:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
        "    *out = *a & *b;\n",
        type_name);
    case CCCL_BIT_OR:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
        "    *out = *a | *b;\n",
        type_name);
    case CCCL_BIT_XOR:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
        "    *out = *a ^ *b;\n",
        type_name);
    case CCCL_MULTIPLIES:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; {0}* out = ({0}*)out_ptr;\n"
        "    *out = *a * *b;\n",
        type_name);
    default:
      return "";
  }
}

// Generate the CUDA source for the reduction
std::string generate_reduce_source(
  cccl_iterator_t input_it, cccl_iterator_t /*output_it*/, cccl_op_t op, cccl_value_t init, bool has_extern_op)
{
  const size_t value_size      = init.type.size;
  const size_t value_alignment = init.type.alignment;
  const auto type_name         = get_cpp_type_name(init.type.type);
  const bool use_typed         = !type_name.empty() && input_it.type == CCCL_POINTER;

  // Use the operation's name, falling back to "user_op"
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";

  std::string src;

  src += "#include <cuda_runtime.h>\n";
  src += "#include <cub/device/device_reduce.cuh>\n\n";

  if (has_extern_op)
  {
    // Operation provided as linked bitcode - declare extern with matching name
    src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* b_ptr, void* out_ptr);\n\n", op_name);
  }
  else if (op.type >= CCCL_PLUS && op.type <= CCCL_MAXIMUM)
  {
    // Well-known operation - generate inline
    src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* b_ptr, void* out_ptr) {{\n", op_name);
    src += get_well_known_op_body(op.type, type_name.empty() ? "char" : type_name);
    src += "}\n\n";
  }

  if (use_typed)
  {
    // Use the actual C++ type for better codegen
    src += std::format("using value_t = {};\n\n", type_name);
  }
  else
  {
    // Generic storage type for custom types
    src += std::format(
      "struct __align__({}) value_t {{\n"
      "  char data[{}];\n"
      "}};\n\n",
      value_alignment,
      value_size);
  }

  src += std::format(
    "struct ReduceOp {{\n"
    "  __device__ __forceinline__\n"
    "  value_t operator()(const value_t& a, const value_t& b) const {{\n"
    "    value_t result;\n"
    "    {}((void*)&a, (void*)&b, (void*)&result);\n"
    "    return result;\n"
    "  }}\n"
    "}};\n\n",
    op_name);

  // Export the reduce function
  src += "#ifdef _WIN32\n"
         "#define EXPORT __declspec(dllexport)\n"
         "#else\n"
         "#define EXPORT __attribute__((visibility(\"default\")))\n"
         "#endif\n\n";

  src += "extern \"C\" EXPORT int cccl_jit_reduce(\n"
         "    void* d_temp_storage,\n"
         "    size_t* temp_storage_bytes,\n"
         "    void* d_input,\n"
         "    void* d_output,\n"
         "    unsigned long long num_items,\n"
         "    void* init_ptr)\n"
         "{\n"
         "    value_t* input = static_cast<value_t*>(d_input);\n"
         "    value_t* output = static_cast<value_t*>(d_output);\n"
         "    value_t init = *static_cast<value_t*>(init_ptr);\n"
         "    ReduceOp op;\n"
         "\n"
         "    cudaError_t err = cub::DeviceReduce::Reduce(\n"
         "        d_temp_storage, *temp_storage_bytes,\n"
         "        input, output, (int)num_items, op, init);\n"
         "\n"
         "    if (d_temp_storage != nullptr) {\n"
         "        cudaError_t sync_err = cudaDeviceSynchronize();\n"
         "        if (err == cudaSuccess) err = sync_err;\n"
         "    }\n"
         "\n"
         "    return (int)err;\n"
         "}\n";

  return src;
}

bool write_bitcode_file(const char* data, size_t size, const std::string& path)
{
  std::ofstream f(path, std::ios::binary);
  if (!f)
  {
    return false;
  }
  f.write(data, static_cast<std::streamsize>(size));
  return f.good();
}

} // anonymous namespace

CUresult cccl_device_reduce_build_ex(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* /*cub_path*/,
  const char* /*thrust_path*/,
  const char* /*libcudacxx_path*/,
  const char* /*ctk_path*/,
  cccl_build_config* /*config*/)
try
{
  // Determine if the operation is provided as linked bitcode
  const bool has_extern_op = (op.type == CCCL_STATELESS || op.type == CCCL_STATEFUL) && op.code != nullptr
                          && op.code_size > 0;

  // Generate the CUDA source
  std::string cuda_source = generate_reduce_source(input_it, output_it, op, init, has_extern_op);

  // Set up clangjit compiler
  clangjit::CompilerConfig config = clangjit::detectDefaultConfig();
  config.sm_version               = cc_major * 10 + cc_minor;
  config.verbose                  = false;

  // If operation has bitcode, write it to a temp file and add to config
  std::string bitcode_path;
  if (has_extern_op)
  {
    bitcode_path =
      (std::filesystem::temp_directory_path() / ("cccl_reduce_op_" + std::to_string(reinterpret_cast<uintptr_t>(build))
                                                 + ".bc"))
        .string();
    if (!write_bitcode_file(op.code, op.code_size, bitcode_path))
    {
      fprintf(stderr, "\nERROR: Failed to write operation bitcode file\n");
      return CUDA_ERROR_UNKNOWN;
    }
    config.device_bitcode_files.push_back(bitcode_path);
  }

  // Compile
  auto* compiler = new clangjit::JITCompiler(config);

  if (!compiler->compile(cuda_source))
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_build(): %s\n", compiler->getLastError().c_str());
    delete compiler;
    if (!bitcode_path.empty())
    {
      std::filesystem::remove(bitcode_path);
    }
    return CUDA_ERROR_UNKNOWN;
  }

  // Clean up temp bitcode file
  if (!bitcode_path.empty())
  {
    std::filesystem::remove(bitcode_path);
  }

  // Get the reduce function pointer
  using reduce_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*);
  auto reduce_fn    = compiler->getFunction<reduce_fn_t>("cccl_jit_reduce");
  if (!reduce_fn)
  {
    fprintf(stderr, "\nERROR: Failed to get reduce function: %s\n", compiler->getLastError().c_str());
    delete compiler;
    return CUDA_ERROR_UNKNOWN;
  }

  build->cc               = cc_major * 10 + cc_minor;
  build->cubin            = nullptr;
  build->cubin_size       = 0;
  build->jit_compiler     = compiler;
  build->reduce_fn        = reinterpret_cast<void*>(reduce_fn);
  build->accumulator_size = init.type.size;
  build->determinism      = determinism;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_build(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

CUresult cccl_device_reduce(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t /*op*/,
  cccl_value_t init,
  CUstream /*stream*/)
{
  try
  {
    using reduce_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*);
    auto reduce_fn    = reinterpret_cast<reduce_fn_t>(build.reduce_fn);

    if (!reduce_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    int status = reduce_fn(d_temp_storage, temp_storage_bytes, d_in.state, d_out.state, num_items, init.state);

    return (status == 0) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  }
  catch (const std::exception& exc)
  {
    fprintf(stderr, "\nEXCEPTION in cccl_device_reduce(): %s\n", exc.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cccl_device_reduce_nondeterministic(
  cccl_device_reduce_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  uint64_t num_items,
  cccl_op_t op,
  cccl_value_t init,
  CUstream stream)
{
  // For now, nondeterministic uses the same path as deterministic
  return cccl_device_reduce(build, d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, op, init, stream);
}

CUresult cccl_device_reduce_cleanup(cccl_device_reduce_build_result_t* build_ptr)
try
{
  if (build_ptr == nullptr)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  if (build_ptr->jit_compiler)
  {
    delete static_cast<clangjit::JITCompiler*>(build_ptr->jit_compiler);
    build_ptr->jit_compiler = nullptr;
  }
  build_ptr->reduce_fn = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

// Backward compatibility wrapper
CUresult cccl_device_reduce_build(
  cccl_device_reduce_build_result_t* build,
  cccl_iterator_t d_in,
  cccl_iterator_t d_out,
  cccl_op_t op,
  cccl_value_t init,
  cccl_determinism_t determinism,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_reduce_build_ex(
    build, d_in, d_out, op, init, determinism, cc_major, cc_minor, cub_path, thrust_path, libcudacxx_path, ctk_path,
    nullptr);
}
