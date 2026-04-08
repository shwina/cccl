#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>

#include <clangjit/codegen/bitcode.hpp>
#include <clangjit/codegen/cub_call.hpp>
#include <clangjit/codegen/iterators.hpp>
#include <clangjit/codegen/operators.hpp>
#include <clangjit/codegen/types.hpp>

namespace clangjit::codegen
{
CubCall CubCall::from(const char* include_header)
{
  CubCall c;
  c.include_ = include_header;
  return c;
}

CubCall& CubCall::run(const char* cub_function)
{
  cub_function_ = cub_function;
  return *this;
}

CubCall& CubCall::name(const char* export_name)
{
  fn_name_ = export_name;
  return *this;
}

// Helper to find the accumulator type from the argument list.
// Priority: first cccl_value_t, then first input_t's value_type.
namespace
{
cccl_type_info find_accum_type(const std::vector<Arg>& args)
{
  // First: look for cccl_value_t (init value defines accum type)
  for (const auto& arg : args)
  {
    if (auto* val = std::get_if<cccl_value_t>(&arg))
    {
      return val->type;
    }
  }
  // Fallback: first input iterator's value_type
  for (const auto& arg : args)
  {
    if (auto* inp = std::get_if<input_t>(&arg))
    {
      return inp->it.value_type;
    }
  }
  // Last resort: first output iterator
  for (const auto& arg : args)
  {
    if (auto* outp = std::get_if<output_t>(&arg))
    {
      return outp->it.value_type;
    }
  }
  return cccl_type_info{sizeof(int), alignof(int), CCCL_INT32};
}
} // anonymous namespace

std::string CubCall::source() const
{
  // Pass 1: determine accumulator type
  cccl_type_info accum_info = find_accum_type(args_);
  std::string accum_preamble;
  std::string accum_type = resolve_type(accum_info, "storage_t", accum_preamble);

  // Counters for unique naming
  int in_count  = 0;
  int out_count = 0;
  int op_count  = 0;
  int val_count = 0;

  // Accumulated sections
  std::string preamble;
  std::vector<std::string> params;
  std::vector<std::string> setup_lines;
  std::vector<std::string> cub_args;

  // Emit accum type
  if (!accum_preamble.empty())
  {
    preamble += accum_preamble;
  }
  preamble += std::format("using accum_t = {};\n\n", accum_type);

  // Pass 2: process each argument
  for (const auto& arg : args_)
  {
    std::visit(
      [&](auto&& a) {
        using T = std::decay_t<decltype(a)>;

        if constexpr (std::is_same_v<T, temp_storage_t>)
        {
          params.push_back("void* d_temp_storage");
          cub_args.push_back("d_temp_storage");
        }
        else if constexpr (std::is_same_v<T, temp_bytes_t>)
        {
          params.push_back("size_t* temp_storage_bytes");
          cub_args.push_back("*temp_storage_bytes");
        }
        else if constexpr (std::is_same_v<T, num_items_t>)
        {
          params.push_back("unsigned long long num_items");
          cub_args.push_back("(unsigned long long)num_items");
        }
        else if constexpr (std::is_same_v<T, input_t>)
        {
          auto idx         = in_count++;
          auto struct_name = std::format("in_{}_it_t", idx);
          auto var_name    = std::format("in_{}", idx);
          auto param_name  = std::format("d_in_{}", idx);

          auto value_type = get_type_name(a.it.value_type.type);
          auto code       = make_input_iterator(a.it, value_type, "accum_t", struct_name, var_name, param_name);

          preamble += code.preamble;
          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, output_t>)
        {
          auto idx         = out_count++;
          auto struct_name = std::format("out_{}_it_t", idx);
          auto var_name    = std::format("out_{}", idx);
          auto param_name  = std::format("d_out_{}", idx);

          auto code = make_output_iterator(a.it, "accum_t", struct_name, var_name, param_name);

          preamble += code.preamble;
          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, cccl_op_t>)
        {
          auto idx          = op_count++;
          auto functor_name = std::format("Op_{}", idx);
          auto var_name     = std::format("op_{}", idx);
          auto state_param  = std::format("op_{}_state", idx);
          bool has_bc       = BitcodeCollector::is_bitcode_op(a);

          auto code = make_binary_op(a, accum_type, functor_name, var_name, state_param, has_bc);

          preamble += code.preamble;
          // Always emit op_state param for ABI stability (unused for stateless ops)
          params.push_back(std::format("void* {}", state_param));
          setup_lines.push_back(code.setup_code);
          cub_args.push_back(var_name);
        }
        else if constexpr (std::is_same_v<T, cccl_value_t>)
        {
          auto idx        = val_count++;
          auto var_name   = std::format("val_{}", idx);
          auto param_name = std::format("val_{}_ptr", idx);

          params.push_back(std::format("void* {}", param_name));
          setup_lines.push_back(std::format(
            "accum_t {};\n    __builtin_memcpy(&{}, {}, sizeof(accum_t));", var_name, var_name, param_name));
          cub_args.push_back(var_name);
        }
      },
      arg);
  }

  // Assemble the complete source
  std::string src;
  src += "#include <cuda_runtime.h>\n";
  src += "#include <cuda_fp16.h>\n";
  src += "#include <cuda/std/iterator>\n";
  src += "#include <cuda/std/functional>\n";
  src += "#include <cuda/functional>\n";
  src += std::format("#include <{}>\n\n", include_);

  src += preamble;

  src += "#ifdef _WIN32\n"
         "#define EXPORT __declspec(dllexport)\n"
         "#else\n"
         "#define EXPORT __attribute__((visibility(\"default\")))\n"
         "#endif\n\n";

  // Function signature
  src += std::format("extern \"C\" EXPORT int {}(\n", fn_name_);
  for (size_t i = 0; i < params.size(); ++i)
  {
    src += "    " + params[i];
    if (i + 1 < params.size())
    {
      src += ",\n";
    }
  }
  src += ")\n{\n";

  // Setup code
  for (const auto& line : setup_lines)
  {
    src += "    " + line + "\n";
  }
  src += "\n";

  // CUB call
  src += std::format("    cudaError_t err = {}(\n", cub_function_);
  for (size_t i = 0; i < cub_args.size(); ++i)
  {
    src += "        " + cub_args[i];
    if (i + 1 < cub_args.size())
    {
      src += ",\n";
    }
  }
  src += ");\n\n";

  // Error return
  src += "    return (int)err;\n"
         "}\n";

  return src;
}

CubCallResult CubCall::compile(
  int cc_major,
  int cc_minor,
  const char* clang_path,
  cccl_build_config* config,
  const char* ctk_path,
  const char* cccl_include_path) const
{
  // 1. Configure compiler
  auto jit_config             = clangjit::detectDefaultConfig();
  jit_config.sm_version       = cc_major * 10 + cc_minor;
  jit_config.verbose          = false;
  jit_config.entry_point_name = fn_name_;

  if (clang_path)
  {
    jit_config.clang_headers_path = clang_path;
  }
  if (ctk_path && ctk_path[0] != '\0')
  {
    jit_config.cuda_toolkit_path = ctk_path;
    // Rebuild library_paths from the new toolkit root so the linker
    // can find libcudart.so in the pip-installed layout.
    jit_config.library_paths.clear();
    for (const char* subdir : {"lib64", "lib"})
    {
      auto candidate = std::filesystem::path(ctk_path) / subdir;
      if (std::filesystem::exists(candidate))
      {
        jit_config.library_paths.push_back(candidate.string());
      }
    }
  }
  if (cccl_include_path && cccl_include_path[0] != '\0')
  {
    jit_config.cccl_include_path = cccl_include_path;
    // When CCCL headers are pip-installed, the clangjit cuda_minimal headers
    // are installed alongside them under the parent directory:
    //   cccl_include_path = .../cuda/cccl/headers/include/
    //   clangjit headers  = .../cuda/cccl/headers/clangjit/cuda_minimal/
    // So derive clangjit_include_path as the parent of cccl_include_path.
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

  // Apply extra build configuration
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
    jit_config.enable_pch = config->enable_pch != 0;
  }

  // 2. Auto-collect bitcode from ops and iterators
  uintptr_t unique_id = reinterpret_cast<uintptr_t>(this);
  BitcodeCollector bitcode(jit_config, unique_id);

  int op_idx  = 0;
  int in_idx  = 0;
  int out_idx = 0;
  for (const auto& arg : args_)
  {
    std::visit(
      [&](auto&& a) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, cccl_op_t>)
        {
          bitcode.add_op(a, std::format("op_{}", op_idx++));
        }
        else if constexpr (std::is_same_v<T, input_t>)
        {
          bitcode.add_iterator(a.it, std::format("in_{}", in_idx++));
        }
        else if constexpr (std::is_same_v<T, output_t>)
        {
          bitcode.add_iterator(a.it, std::format("out_{}", out_idx++));
        }
      },
      arg);
  }

  // 3. Generate source
  std::string cuda_source = source();

  // 4. Compile
  auto* compiler = new JITCompiler(jit_config);
  if (!compiler->compile(cuda_source))
  {
    std::string err = compiler->getLastError();
    delete compiler;
    bitcode.cleanup();
    throw std::runtime_error("CubCall compilation failed: " + err);
  }

  bitcode.cleanup();

  // 5. Extract function pointer
  using fn_t = int (*)(void*, ...);
  auto fn    = compiler->getFunction<fn_t>(fn_name_);
  if (!fn)
  {
    std::string err = compiler->getLastError();
    delete compiler;
    throw std::runtime_error("CubCall function lookup failed: " + err);
  }

  // 6. Copy cubin
  auto cubin = compiler->getCubin();

  return CubCallResult{compiler, reinterpret_cast<void*>(fn), std::move(cubin)};
}
} // namespace clangjit::codegen
