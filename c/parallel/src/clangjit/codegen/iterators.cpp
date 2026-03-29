#include <clangjit/codegen/iterators.hpp>

#include <format>

namespace clangjit::codegen
{

IteratorCode make_input_iterator(
  cccl_iterator_t it,
  const std::string& value_type_name,
  const std::string& accum_type_name,
  const std::string& struct_name,
  const std::string& var_name,
  const std::string& state_param)
{
  IteratorCode result;
  result.local_var = var_name;

  if (it.type == CCCL_POINTER)
  {
    // For pointer iterators, the input type matches value_type
    if (value_type_name.empty())
    {
      result.type_name = accum_type_name + "*";
      result.preamble  = std::format("using {} = {}*;\n\n", struct_name, accum_type_name);
    }
    else
    {
      result.type_name = value_type_name + "*";
      result.preamble  = std::format("using {} = {}*;\n\n", struct_name, value_type_name);
    }
    result.setup_code = std::format("{} {} = static_cast<{}>({}); ", struct_name, var_name, struct_name, state_param);
  }
  else
  {
    // Custom iterator with state + advance + dereference
    const std::string adv_name =
      (it.advance.name && it.advance.name[0]) ? it.advance.name : (var_name + "_advance");
    const std::string deref_name =
      (it.dereference.name && it.dereference.name[0]) ? it.dereference.name : (var_name + "_dereference");

    auto input_val_type = value_type_name.empty() ? accum_type_name : value_type_name;
    auto val_alias      = var_name + "_value_t";

    result.type_name = struct_name;
    result.preamble  = std::format("using {} = {};\n", val_alias, input_val_type);

    result.preamble += std::format(
      "extern \"C\" __device__ void {}(void* state, const void* offset);\n"
      "extern \"C\" __device__ void {}(const void* state, {}* result);\n\n",
      adv_name,
      deref_name,
      val_alias);

    result.preamble += std::format(
      "struct {} {{\n"
      "  using value_type = {};\n"
      "  using difference_type = long long;\n"
      "  using pointer = {}*;\n"
      "  using reference = {};\n"
      "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
      "\n"
      "  char state[{}];\n"
      "\n"
      "  __device__ {} operator+(difference_type n) const {{\n"
      "    {} copy = *this;\n"
      "    unsigned long long offset = static_cast<unsigned long long>(n);\n"
      "    {}(copy.state, &offset);\n"
      "    return copy;\n"
      "  }}\n"
      "  __device__ difference_type operator-(const {}&) const {{ return 0; }}\n"
      "  __device__ reference operator*() const {{\n"
      "    {} result;\n"
      "    {}(state, &result);\n"
      "    return result;\n"
      "  }}\n"
      "  __device__ reference operator[](difference_type n) const {{ return *(*this + n); }}\n"
      "  __device__ bool operator==(const {}&) const {{ return false; }}\n"
      "  __device__ bool operator!=(const {}&) const {{ return true; }}\n"
      "}};\n\n",
      struct_name, // struct name
      val_alias, // value_type
      val_alias, // pointer
      val_alias, // reference
      it.size, // state size
      struct_name, // operator+ return type
      struct_name, // copy type
      adv_name, // advance function
      struct_name, // operator- param
      val_alias, // operator* result type
      deref_name, // dereference function
      struct_name, // operator== param
      struct_name); // operator!= param

    result.setup_code = std::format(
      "{} {};\n"
      "    __builtin_memcpy({}.state, {}, {});",
      struct_name,
      var_name,
      var_name,
      state_param,
      it.size);
  }

  return result;
}

IteratorCode make_output_iterator(
  cccl_iterator_t it,
  const std::string& accum_type_name,
  const std::string& struct_name,
  const std::string& var_name,
  const std::string& state_param)
{
  IteratorCode result;
  result.local_var = var_name;

  if (it.type == CCCL_POINTER)
  {
    result.type_name  = accum_type_name + "*";
    result.preamble   = std::format("using {} = {}*;\n\n", struct_name, accum_type_name);
    result.setup_code = std::format(
      "{} {} = static_cast<{}*>({});", struct_name, var_name, accum_type_name, state_param);
  }
  else
  {
    const std::string adv_name =
      (it.advance.name && it.advance.name[0]) ? it.advance.name : (var_name + "_advance");
    const std::string deref_name =
      (it.dereference.name && it.dereference.name[0]) ? it.dereference.name : (var_name + "_dereference");

    auto proxy_name = var_name + "_proxy_t";

    result.type_name = struct_name;
    result.preamble  = std::format(
      "extern \"C\" __device__ void {}(void* state, const void* offset);\n"
       "extern \"C\" __device__ void {}(void* state, const void* value);\n\n",
      adv_name,
      deref_name);

    result.preamble += std::format(
      "struct {} {{\n"
      "  void* state;\n"
      "  __device__ void operator=(const {}& val) {{\n"
      "    {}(state, &val);\n"
      "  }}\n"
      "}};\n",
      proxy_name,
      accum_type_name,
      deref_name);

    result.preamble += std::format(
      "struct {} {{\n"
      "  using value_type = {};\n"
      "  using difference_type = long long;\n"
      "  using pointer = {}*;\n"
      "  using reference = {};\n"
      "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
      "\n"
      "  char state[{}];\n"
      "\n"
      "  __device__ {} operator+(difference_type n) const {{\n"
      "    {} copy = *this;\n"
      "    unsigned long long offset = static_cast<unsigned long long>(n);\n"
      "    {}(copy.state, &offset);\n"
      "    return copy;\n"
      "  }}\n"
      "  __device__ difference_type operator-(const {}&) const {{ return 0; }}\n"
      "  __device__ reference operator*() {{ return {}{{state}}; }}\n"
      "  __device__ reference operator[](difference_type n) {{ return *(*this + n); }}\n"
      "}};\n\n",
      struct_name, // struct name
      accum_type_name, // value_type
      accum_type_name, // pointer
      proxy_name, // reference (proxy type)
      it.size, // state size
      struct_name, // operator+ return
      struct_name, // copy type
      adv_name, // advance function
      struct_name, // operator- param
      proxy_name); // proxy constructor

    result.setup_code = std::format(
      "{} {};\n"
      "    __builtin_memcpy({}.state, {}, {});",
      struct_name,
      var_name,
      var_name,
      state_param,
      it.size);
  }

  return result;
}

} // namespace clangjit::codegen
