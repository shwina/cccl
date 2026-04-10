#include <format>

#include <clangjit/codegen/iterators.hpp>
#include <clangjit/codegen/types.hpp>

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
    // For pointer iterators, the element type is value_type.
    // When value_type_name is empty (unknown/struct type), resolve it from the iterator's
    // value_type info to get a correctly-sized storage struct — falling back to accum_t
    // would use the wrong element size if the value type differs from the accumulator.
    std::string elem_type;
    if (value_type_name.empty())
    {
      auto elem_alias = struct_name + "_elem_t";
      elem_type       = resolve_type(it.value_type, elem_alias.c_str(), result.preamble);
    }
    else
    {
      elem_type = value_type_name;
    }
    result.type_name = elem_type + "*";
    result.preamble += std::format("using {} = {}*;\n\n", struct_name, elem_type);
    result.setup_code = std::format("{} {} = static_cast<{}>({}); ", struct_name, var_name, struct_name, state_param);
  }
  else
  {
    // Custom iterator with state + advance + dereference
    const std::string adv_name = (it.advance.name && it.advance.name[0]) ? it.advance.name : (var_name + "_advance");
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

    // Positional args: {0}=struct_name, {1}=val_alias, {2}=it.size, {3}=adv_name, {4}=deref_name
    result.preamble += std::format(
      "struct {0} {{\n"
      "  using value_type = {1};\n"
      "  using difference_type = long long;\n"
      "  using pointer = {1}*;\n"
      "  using reference = {1};\n"
      "  using iterator_category = cuda::std::random_access_iterator_tag;\n"
      "\n"
      "  char state[{2}];\n"
      "\n"
      "  __device__ {0} operator+(difference_type n) const {{\n"
      "    {0} copy = *this;\n"
      "    unsigned long long offset = static_cast<unsigned long long>(n);\n"
      "    {3}(copy.state, &offset);\n"
      "    return copy;\n"
      "  }}\n"
      "  __device__ {0}& operator+=(difference_type n) {{\n"
      "    unsigned long long offset = static_cast<unsigned long long>(n);\n"
      "    {3}(state, &offset);\n"
      "    return *this;\n"
      "  }}\n"
      "  __device__ {0}& operator++() {{ return *this += 1; }}\n"
      "  __device__ {0}  operator++(int) {{ {0} tmp = *this; ++(*this); return tmp; }}\n"
      "  __device__ difference_type operator-(const {0}&) const {{ return 0; }}\n"
      "  __device__ {1} operator*() const {{\n"
      "    {1} result;\n"
      "    {4}(state, &result);\n"
      "    return result;\n"
      "  }}\n"
      "  __device__ {1} operator[](difference_type n) const {{ return *(*this + n); }}\n"
      "  __device__ bool operator==(const {0}&) const {{ return false; }}\n"
      "  __device__ bool operator!=(const {0}&) const {{ return true; }}\n"
      "}};\n\n",
      struct_name, // {0}
      val_alias, // {1}
      it.size, // {2}
      adv_name, // {3}
      deref_name); // {4}

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
  const std::string& state_param,
  const std::string& value_type_name)
{
  IteratorCode result;
  result.local_var = var_name;

  // For custom iterators the element type comes from the dereference function so the
  // accum_t fallback is fine; for pointer iterators we resolve the actual value_type
  // below to get the correct element size.
  const std::string elem_type = value_type_name.empty() ? accum_type_name : value_type_name;

  if (it.type == CCCL_POINTER)
  {
    // When value_type_name is empty (unknown/struct type), resolve from the iterator's own
    // value_type info so the element size is correct — not from accum_t which may differ.
    std::string ptr_elem_type;
    if (value_type_name.empty())
    {
      auto elem_alias = struct_name + "_elem_t";
      ptr_elem_type   = resolve_type(it.value_type, elem_alias.c_str(), result.preamble);
    }
    else
    {
      ptr_elem_type = value_type_name;
    }
    result.type_name = ptr_elem_type + "*";
    result.preamble += std::format("using {} = {}*;\n\n", struct_name, ptr_elem_type);
    result.setup_code = std::format("{} {} = static_cast<{}*>({});", struct_name, var_name, ptr_elem_type, state_param);
  }
  else
  {
    const std::string adv_name = (it.advance.name && it.advance.name[0]) ? it.advance.name : (var_name + "_advance");
    const std::string deref_name =
      (it.dereference.name && it.dereference.name[0]) ? it.dereference.name : (var_name + "_dereference");

    auto proxy_name = var_name + "_proxy_t";

    result.type_name = struct_name;
    result.preamble  = std::format(
      "extern \"C\" __device__ void {}(void* state, const void* offset);\n"
       "extern \"C\" __device__ void {}(void* state, const void* value);\n\n",
      adv_name,
      deref_name);

    // The proxy carries a COPY of the iterator state, not a pointer to it.
    // This is critical for indexed writes (output_it[i] = val): operator[] creates
    // a temporary advanced iterator, calls operator* on it, and returns the proxy
    // by value.  After operator[] returns the temporary is destroyed, so a pointer
    // to its state would be dangling.  Storing the state bytes in the proxy itself
    // makes the proxy self-contained and safe across that return.
    result.preamble += std::format(
      "struct {} {{\n"
      "  char state[{}];\n"
      "  __device__ void operator=(const {}& val) {{\n"
      "    {}(state, &val);\n"
      "  }}\n"
      "}};\n",
      proxy_name,
      it.size,
      elem_type,
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
      "  __device__ reference operator*() {{\n"
      "    {} proxy;\n"
      "    __builtin_memcpy(proxy.state, state, {});\n"
      "    return proxy;\n"
      "  }}\n"
      "  __device__ reference operator[](difference_type n) {{ return *(*this + n); }}\n"
      "}};\n\n",
      struct_name, // struct name
      elem_type, // value_type
      elem_type, // pointer
      proxy_name, // reference (proxy type)
      it.size, // state size
      struct_name, // operator+ return
      struct_name, // copy type
      adv_name, // advance function
      struct_name, // operator- param
      proxy_name, // proxy type for operator*
      it.size); // memcpy size

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
