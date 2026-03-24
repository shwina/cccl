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

#include <clangjit/compiler.hpp>
#include <clangjit/config.hpp>
#include <clangjit/jit_compiler.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

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

std::string get_storage_type_def(size_t size, size_t alignment)
{
  return std::format(
    "struct __align__({}) storage_t {{\n"
    "  char data[{}];\n"
    "}};\n",
    alignment,
    size);
}

// Generate op source: either inline well-known, embed C++ source, or declare extern for bitcode linkage
std::string generate_op_source(
  cccl_op_t op, const std::string& accum_type_name, bool has_bitcode_op, bool is_stateful)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  std::string src;

  if (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0)
  {
    // Embed C++ source directly
    src += std::string(op.code, op.code_size) + "\n\n";
  }
  else if (has_bitcode_op)
  {
    // Extern declaration for bitcode-linked operation
    if (is_stateful)
    {
      src += std::format(
        "extern \"C\" __device__ void {}(void* state, void* a_ptr, void* b_ptr, void* out_ptr);\n\n", op_name);
    }
    else
    {
      src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* b_ptr, void* out_ptr);\n\n", op_name);
    }
  }
  else if (op.type >= CCCL_PLUS && op.type <= CCCL_MAXIMUM)
  {
    // Well-known operation - generate inline
    src += std::format("extern \"C\" __device__ void {}(void* a_ptr, void* b_ptr, void* out_ptr) {{\n", op_name);
    src += get_well_known_op_body(op.type, accum_type_name);
    src += "}\n\n";
  }

  return src;
}

// Generate the functor that wraps the op for CUB
std::string generate_op_functor(cccl_op_t op, bool is_stateful)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  std::string src;

  if (is_stateful)
  {
    src += std::format(
      "struct ReduceOp {{\n"
      "  void* state;\n"
      "  __device__ __forceinline__\n"
      "  accum_t operator()(const accum_t& a, const accum_t& b) const {{\n"
      "    accum_t result;\n"
      "    {}(state, (void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      op_name);
  }
  else
  {
    src += std::format(
      "struct ReduceOp {{\n"
      "  __device__ __forceinline__\n"
      "  accum_t operator()(const accum_t& a, const accum_t& b) const {{\n"
      "    accum_t result;\n"
      "    {}((void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      op_name);
  }

  return src;
}

// Generate input iterator wrapper
std::string generate_input_iterator(cccl_iterator_t it)
{
  std::string src;

  if (it.type == CCCL_POINTER)
  {
    // For pointer iterators, the input type matches value_type
    auto input_type = get_cpp_type_name(it.value_type.type);
    if (input_type.empty())
    {
      // Custom type - use accum_t (which should match)
      src += "using input_it_t = accum_t*;\n\n";
    }
    else
    {
      src += std::format("using input_it_t = {}*;\n\n", input_type);
    }
  }
  else
  {
    // Custom iterator: state + advance + dereference as extern "C" functions
    const std::string adv_name  = (it.advance.name && it.advance.name[0]) ? it.advance.name : "input_advance";
    const std::string deref_name =
      (it.dereference.name && it.dereference.name[0]) ? it.dereference.name : "input_dereference";

    // Determine the input value type
    auto input_val_type = get_cpp_type_name(it.value_type.type);
    if (input_val_type.empty())
    {
      input_val_type = "accum_t";
    }

    // Define a separate input_value_t if different from accum_t
    src += std::format(
      "using input_value_t = {};\n",
      input_val_type);

    src += std::format(
      "extern \"C\" __device__ void {}(void* state, const void* offset);\n"
      "extern \"C\" __device__ void {}(const void* state, input_value_t* result);\n\n",
      adv_name,
      deref_name);

    src += std::format(
      "struct input_it_t {{\n"
      "  using value_type = input_value_t;\n"
      "  using difference_type = long long;\n"
      "  using pointer = input_value_t*;\n"
      "  using reference = input_value_t;\n"
      "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
      "\n"
      "  char state[{}];\n"
      "\n"
      "  __device__ input_it_t operator+(difference_type n) const {{\n"
      "    input_it_t copy = *this;\n"
      "    unsigned long long offset = static_cast<unsigned long long>(n);\n"
      "    {}(copy.state, &offset);\n"
      "    return copy;\n"
      "  }}\n"
      "  __device__ difference_type operator-(const input_it_t&) const {{ return 0; }}\n"
      "  __device__ reference operator*() const {{\n"
      "    input_value_t result;\n"
      "    {}(state, &result);\n"
      "    return result;\n"
      "  }}\n"
      "  __device__ reference operator[](difference_type n) const {{ return *(*this + n); }}\n"
      "  __device__ bool operator==(const input_it_t&) const {{ return false; }}\n"
      "  __device__ bool operator!=(const input_it_t&) const {{ return true; }}\n"
      "}};\n\n",
      it.size,
      adv_name,
      deref_name);
  }

  return src;
}

// Generate output iterator wrapper
std::string generate_output_iterator(cccl_iterator_t it)
{
  if (it.type == CCCL_POINTER)
  {
    return "using output_it_t = accum_t*;\n\n";
  }

  const std::string adv_name  = (it.advance.name && it.advance.name[0]) ? it.advance.name : "output_advance";
  const std::string deref_name =
    (it.dereference.name && it.dereference.name[0]) ? it.dereference.name : "output_dereference";

  std::string src;
  src += std::format(
    "extern \"C\" __device__ void {}(void* state, const void* offset);\n"
    "extern \"C\" __device__ void {}(void* state, const void* value);\n\n",
    adv_name,
    deref_name);

  src += std::format(
    "struct output_proxy_t {{\n"
    "  void* state;\n"
    "  __device__ void operator=(const accum_t& val) {{\n"
    "    {}(state, &val);\n"
    "  }}\n"
    "}};\n"
    "struct output_it_t {{\n"
    "  using value_type = accum_t;\n"
    "  using difference_type = long long;\n"
    "  using pointer = accum_t*;\n"
    "  using reference = output_proxy_t;\n"
    "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
    "\n"
    "  char state[{}];\n"
    "\n"
    "  __device__ output_it_t operator+(difference_type n) const {{\n"
    "    output_it_t copy = *this;\n"
    "    unsigned long long offset = static_cast<unsigned long long>(n);\n"
    "    {}(copy.state, &offset);\n"
    "    return copy;\n"
    "  }}\n"
    "  __device__ difference_type operator-(const output_it_t&) const {{ return 0; }}\n"
    "  __device__ reference operator*() {{ return output_proxy_t{{state}}; }}\n"
    "  __device__ reference operator[](difference_type n) {{ return *(*this + n); }}\n"
    "}};\n\n",
    deref_name,
    it.size,
    adv_name);

  return src;
}

// Generate the full CUDA source
std::string generate_reduce_source(
  cccl_iterator_t input_it, cccl_iterator_t output_it, cccl_op_t op, cccl_value_t init, bool has_bitcode_op)
{
  const bool is_stateful = (op.type == CCCL_STATEFUL);

  // Determine accum type
  const auto accum_type_name = get_cpp_type_name(init.type.type);
  const auto storage_def     = get_storage_type_def(init.type.size, init.type.alignment);

  std::string src;
  src += "#include <cuda_runtime.h>\n";
  src += "#include <cuda/std/iterator>\n";
  src += "#include <cub/device/device_reduce.cuh>\n\n";

  // Accum type
  if (!accum_type_name.empty())
  {
    src += std::format("using accum_t = {};\n\n", accum_type_name);
  }
  else
  {
    src += storage_def + "using accum_t = storage_t;\n\n";
  }

  // Op
  src += generate_op_source(op, accum_type_name.empty() ? "storage_t" : accum_type_name, has_bitcode_op, is_stateful);
  src += generate_op_functor(op, is_stateful);

  // Iterators
  src += generate_input_iterator(input_it);
  src += generate_output_iterator(output_it);

  // Export
  src += "#ifdef _WIN32\n"
         "#define EXPORT __declspec(dllexport)\n"
         "#else\n"
         "#define EXPORT __attribute__((visibility(\"default\")))\n"
         "#endif\n\n";

  // The exported reduce function
  // Takes: d_temp_storage, temp_storage_bytes, d_input_state, d_output_state,
  //        num_items, init_ptr, op_state_ptr
  src += "extern \"C\" EXPORT int cccl_jit_reduce(\n"
         "    void* d_temp_storage,\n"
         "    size_t* temp_storage_bytes,\n"
         "    void* d_input_state,\n"
         "    void* d_output_state,\n"
         "    unsigned long long num_items,\n"
         "    void* init_ptr,\n"
         "    void* op_state_ptr)\n"
         "{\n";

  // Input iterator
  if (input_it.type == CCCL_POINTER)
  {
    src += "    input_it_t input = static_cast<accum_t*>(d_input_state);\n";
  }
  else
  {
    src += std::format(
      "    input_it_t input;\n"
      "    __builtin_memcpy(input.state, d_input_state, {});\n",
      input_it.size);
  }

  // Output iterator
  if (output_it.type == CCCL_POINTER)
  {
    src += "    output_it_t output = static_cast<accum_t*>(d_output_state);\n";
  }
  else
  {
    src += std::format(
      "    output_it_t output;\n"
      "    __builtin_memcpy(output.state, d_output_state, {});\n",
      output_it.size);
  }

  src += "    accum_t init;\n"
         "    __builtin_memcpy(&init, init_ptr, sizeof(accum_t));\n";

  if (is_stateful)
  {
    src += "    ReduceOp op{op_state_ptr};\n";
  }
  else
  {
    src += "    ReduceOp op;\n";
  }

  src += "\n"
         "    cudaError_t err = cub::DeviceReduce::Reduce(\n"
         "        d_temp_storage, *temp_storage_bytes,\n"
         "        input, output, (unsigned long long)num_items, op, init);\n"
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

bool write_file(const char* data, size_t size, const std::string& path)
{
  std::ofstream f(path, std::ios::binary);
  if (!f)
  {
    return false;
  }
  f.write(data, static_cast<std::streamsize>(size));
  return f.good();
}

std::string make_temp_path(const std::string& prefix, uintptr_t id, const std::string& ext)
{
  return (std::filesystem::temp_directory_path() / (prefix + std::to_string(id) + ext)).string();
}

// Compile C++ source to LLVM bitcode, write to temp file
bool compile_cpp_source_to_bitcode(
  const char* source,
  size_t source_size,
  const std::string& output_path,
  const clangjit::CompilerConfig& jit_config)
{
  clangjit::CUDACompiler compiler;
  std::string src(source, source_size);
  auto result = compiler.compileToDeviceBitcode(src, jit_config);
  if (!result.success)
  {
    fprintf(stderr, "\nERROR compiling iterator op to bitcode: %s\n", result.diagnostics.c_str());
    return false;
  }
  return write_file(result.bitcode.data(), result.bitcode.size(), output_path);
}

// Collect all bitcode files to link (op + iterator advance/dereference functions)
void collect_bitcode_files(
  cccl_op_t op,
  cccl_iterator_t input_it,
  cccl_iterator_t output_it,
  uintptr_t unique_id,
  std::vector<std::string>& bitcode_paths,
  clangjit::CompilerConfig& config)
{
  auto add_bitcode = [&](const char* data, size_t size, const std::string& name) {
    if (data && size > 0)
    {
      auto path = make_temp_path("cccl_" + name + "_", unique_id, ".bc");
      if (write_file(data, size, path))
      {
        config.device_bitcode_files.push_back(path);
        bitcode_paths.push_back(path);
      }
    }
  };

  auto add_op_code = [&](cccl_op_t& the_op, const std::string& name) {
    if (!the_op.code || the_op.code_size == 0)
    {
      return;
    }
    if (the_op.code_type == CCCL_OP_CPP_SOURCE)
    {
      // Compile C++ source to LLVM bitcode on the fly
      auto path = make_temp_path("cccl_" + name + "_", unique_id, ".bc");
      if (compile_cpp_source_to_bitcode(the_op.code, the_op.code_size, path, config))
      {
        config.device_bitcode_files.push_back(path);
        bitcode_paths.push_back(path);
      }
    }
    else
    {
      // LLVM_IR or LTOIR — already bitcode
      add_bitcode(the_op.code, the_op.code_size, name);
    }
  };

  // Op bitcode (only for LLVM_IR or LTOIR code types, not CPP_SOURCE — CPP_SOURCE is embedded inline)
  bool is_bitcode = op.code_type == CCCL_OP_LLVM_IR || op.code_type == CCCL_OP_LTOIR;
  if (is_bitcode && op.code && op.code_size > 0)
  {
    add_bitcode(op.code, op.code_size, "op");
  }

  // Input iterator advance/dereference
  if (input_it.type == CCCL_ITERATOR)
  {
    add_op_code(input_it.advance, "in_adv");
    add_op_code(input_it.dereference, "in_deref");
  }

  // Output iterator advance/dereference
  if (output_it.type == CCCL_ITERATOR)
  {
    add_op_code(output_it.advance, "out_adv");
    add_op_code(output_it.dereference, "out_deref");
  }
}

void cleanup_temp_files(const std::vector<std::string>& paths)
{
  for (const auto& p : paths)
  {
    std::filesystem::remove(p);
  }
}

} // anonymous namespace

using reduce_fn_t = int (*)(void*, size_t*, void*, void*, unsigned long long, void*, void*);

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
  cccl_build_config* build_config)
try
{
  const uintptr_t unique_id = reinterpret_cast<uintptr_t>(build);

  // Determine if the operation is provided as linked bitcode
  const bool is_bitcode_op = (op.code_type == CCCL_OP_LLVM_IR || op.code_type == CCCL_OP_LTOIR) && op.code != nullptr
                          && op.code_size > 0;
  const bool has_bitcode_op = is_bitcode_op;

  // Generate the CUDA source
  std::string cuda_source = generate_reduce_source(input_it, output_it, op, init, has_bitcode_op);

  // Set up clangjit compiler
  clangjit::CompilerConfig jit_config = clangjit::detectDefaultConfig();
  jit_config.sm_version               = cc_major * 10 + cc_minor;
  jit_config.verbose                  = false;

  // Apply extra build configuration
  if (build_config)
  {
    for (size_t i = 0; i < build_config->num_extra_include_dirs; ++i)
    {
      jit_config.include_paths.push_back(build_config->extra_include_dirs[i]);
    }
    for (size_t i = 0; i < build_config->num_extra_compile_flags; ++i)
    {
      // Parse -D flags into macro_definitions
      std::string flag = build_config->extra_compile_flags[i];
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

  // Collect bitcode files
  std::vector<std::string> bitcode_paths;
  collect_bitcode_files(op, input_it, output_it, unique_id, bitcode_paths, jit_config);

  // Compile
  auto* compiler = new clangjit::JITCompiler(jit_config);

  if (!compiler->compile(cuda_source))
  {
    fprintf(stderr, "\nERROR in cccl_device_reduce_build(): %s\n", compiler->getLastError().c_str());
    delete compiler;
    cleanup_temp_files(bitcode_paths);
    return CUDA_ERROR_UNKNOWN;
  }

  cleanup_temp_files(bitcode_paths);

  // Get the reduce function pointer
  auto reduce_fn = compiler->getFunction<reduce_fn_t>("cccl_jit_reduce");
  if (!reduce_fn)
  {
    fprintf(stderr, "\nERROR: Failed to get reduce function: %s\n", compiler->getLastError().c_str());
    delete compiler;
    return CUDA_ERROR_UNKNOWN;
  }

  build->cc               = cc_major * 10 + cc_minor;
  // Store cubin for SASS inspection
  const auto& cubin = compiler->getCubin();
  if (!cubin.empty())
  {
    auto* cubin_copy = new char[cubin.size()];
    std::memcpy(cubin_copy, cubin.data(), cubin.size());
    build->cubin      = cubin_copy;
    build->cubin_size = cubin.size();
  }
  else
  {
    build->cubin      = nullptr;
    build->cubin_size = 0;
  }
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
  cccl_op_t op,
  cccl_value_t init,
  CUstream /*stream*/)
{
  try
  {
    auto reduce_fn = reinterpret_cast<reduce_fn_t>(build.reduce_fn);

    if (!reduce_fn)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    int status = reduce_fn(d_temp_storage, temp_storage_bytes, d_in.state, d_out.state, num_items, init.state, op.state);

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
  if (build_ptr->cubin)
  {
    delete[] static_cast<char*>(build_ptr->cubin);
    build_ptr->cubin = nullptr;
  }
  build_ptr->cubin_size = 0;
  build_ptr->reduce_fn  = nullptr;

  return CUDA_SUCCESS;
}
catch (const std::exception& exc)
{
  fprintf(stderr, "\nEXCEPTION in cccl_device_reduce_cleanup(): %s\n", exc.what());
  return CUDA_ERROR_UNKNOWN;
}

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
