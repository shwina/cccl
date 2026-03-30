#include <clangjit/codegen/operators.hpp>

#include <format>

namespace clangjit::codegen
{

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
    case CCCL_LESS:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; bool* out = (bool*)out_ptr;\n"
        "    *out = *a < *b;\n",
        type_name);
    case CCCL_GREATER:
      return std::format(
        "    {0}* a = ({0}*)a_ptr; {0}* b = ({0}*)b_ptr; bool* out = (bool*)out_ptr;\n"
        "    *out = *a > *b;\n",
        type_name);
    default:
      return "";
  }
}

namespace
{

std::string generate_op_source(
  cccl_op_t op, const std::string& accum_type, bool has_bitcode, bool is_stateful, bool is_comparison)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  std::string src;

  if (op.code_type == CCCL_OP_CPP_SOURCE && op.code && op.code_size > 0)
  {
    // Embed C++ source directly
    src += std::string(op.code, op.code_size) + "\n\n";
  }
  else if (has_bitcode)
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
    src += get_well_known_op_body(op.type, accum_type);
    src += "}\n\n";
  }

  return src;
}

std::string generate_binary_functor(cccl_op_t op, const std::string& accum_type, const std::string& functor_name)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  const bool is_stateful    = (op.type == CCCL_STATEFUL);

  if (is_stateful)
  {
    return std::format(
      "struct {} {{\n"
      "  void* state;\n"
      "  __device__ __forceinline__\n"
      "  {} operator()(const {}& a, const {}& b) const {{\n"
      "    {} result;\n"
      "    {}(state, (void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      accum_type,
      accum_type,
      accum_type,
      accum_type,
      op_name);
  }
  else
  {
    return std::format(
      "struct {} {{\n"
      "  __device__ __forceinline__\n"
      "  {} operator()(const {}& a, const {}& b) const {{\n"
      "    {} result;\n"
      "    {}((void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      accum_type,
      accum_type,
      accum_type,
      accum_type,
      op_name);
  }
}

std::string generate_comparison_functor(cccl_op_t op, const std::string& key_type, const std::string& functor_name)
{
  const std::string op_name = (op.name && op.name[0]) ? op.name : "user_op";
  const bool is_stateful    = (op.type == CCCL_STATEFUL);

  if (is_stateful)
  {
    return std::format(
      "struct {} {{\n"
      "  void* state;\n"
      "  __device__ __forceinline__\n"
      "  bool operator()(const {}& a, const {}& b) const {{\n"
      "    bool result;\n"
      "    {}(state, (void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      key_type,
      key_type,
      op_name);
  }
  else
  {
    return std::format(
      "struct {} {{\n"
      "  __device__ __forceinline__\n"
      "  bool operator()(const {}& a, const {}& b) const {{\n"
      "    bool result;\n"
      "    {}((void*)&a, (void*)&b, (void*)&result);\n"
      "    return result;\n"
      "  }}\n"
      "}};\n\n",
      functor_name,
      key_type,
      key_type,
      op_name);
  }
}

// Returns the cuda::std (or cuda::) functor type string for a well-known op, or empty if not well-known.
const char* get_well_known_functor_type(cccl_op_kind_t kind)
{
  switch (kind)
  {
    case CCCL_PLUS:
      return "::cuda::std::plus<>";
    case CCCL_MINUS:
      return "::cuda::std::minus<>";
    case CCCL_MULTIPLIES:
      return "::cuda::std::multiplies<>";
    case CCCL_DIVIDES:
      return "::cuda::std::divides<>";
    case CCCL_MODULUS:
      return "::cuda::std::modulus<>";
    case CCCL_EQUAL_TO:
      return "::cuda::std::equal_to<>";
    case CCCL_NOT_EQUAL_TO:
      return "::cuda::std::not_equal_to<>";
    case CCCL_GREATER:
      return "::cuda::std::greater<>";
    case CCCL_LESS:
      return "::cuda::std::less<>";
    case CCCL_GREATER_EQUAL:
      return "::cuda::std::greater_equal<>";
    case CCCL_LESS_EQUAL:
      return "::cuda::std::less_equal<>";
    case CCCL_BIT_AND:
      return "::cuda::std::bit_and<>";
    case CCCL_BIT_OR:
      return "::cuda::std::bit_or<>";
    case CCCL_BIT_XOR:
      return "::cuda::std::bit_xor<>";
    case CCCL_MINIMUM:
      return "::cuda::minimum<>";
    case CCCL_MAXIMUM:
      return "::cuda::maximum<>";
    default:
      return nullptr;
  }
}

} // anonymous namespace

OperatorCode make_binary_op(
  cccl_op_t op,
  const std::string& accum_type,
  const std::string& functor_name,
  const std::string& var_name,
  const std::string& state_param,
  bool has_bitcode)
{
  // For well-known operations, use cuda::std functors directly instead of generating wrapper code.
  const char* well_known_type = get_well_known_functor_type(op.type);
  if (well_known_type)
  {
    OperatorCode result;
    result.local_var  = var_name;
    result.setup_code = std::format("{} {}{{}};", well_known_type, var_name);
    return result;
  }

  const bool is_stateful = (op.type == CCCL_STATEFUL);

  OperatorCode result;
  result.local_var = var_name;
  result.preamble  = generate_op_source(op, accum_type, has_bitcode, is_stateful, false);
  result.preamble += generate_binary_functor(op, accum_type, functor_name);

  if (is_stateful)
  {
    result.setup_code = std::format("{} {}{{{}}};", functor_name, var_name, state_param);
  }
  else
  {
    result.setup_code = std::format("{} {};", functor_name, var_name);
  }

  return result;
}

OperatorCode make_comparison_op(
  cccl_op_t op,
  const std::string& key_type,
  const std::string& functor_name,
  const std::string& var_name,
  const std::string& state_param,
  bool has_bitcode)
{
  // For well-known operations, use cuda::std functors directly.
  const char* well_known_type = get_well_known_functor_type(op.type);
  if (well_known_type)
  {
    OperatorCode result;
    result.local_var  = var_name;
    result.setup_code = std::format("{} {}{{}};", well_known_type, var_name);
    return result;
  }

  const bool is_stateful = (op.type == CCCL_STATEFUL);

  OperatorCode result;
  result.local_var = var_name;
  result.preamble  = generate_op_source(op, key_type, has_bitcode, is_stateful, true);
  result.preamble += generate_comparison_functor(op, key_type, functor_name);

  if (is_stateful)
  {
    result.setup_code = std::format("{} {}{{{}}};", functor_name, var_name, state_param);
  }
  else
  {
    result.setup_code = std::format("{} {};", functor_name, var_name);
  }

  return result;
}

} // namespace clangjit::codegen
