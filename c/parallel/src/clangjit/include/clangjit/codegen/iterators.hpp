#pragma once

#include <cccl/c/types.h>

#include <string>

namespace clangjit::codegen
{

// Result of generating iterator code.
struct IteratorCode
{
  std::string preamble;   // type alias or struct definition (goes at file scope)
  std::string setup_code; // initialization inside function body
  std::string local_var;  // e.g., "in_0"
  std::string type_name;  // e.g., "in_0_it_t" or "accum_t*"
};

// Generate code for an input iterator.
// For CCCL_POINTER: emits a type alias and pointer cast.
// For CCCL_ITERATOR: emits a full iterator struct with advance/dereference.
IteratorCode make_input_iterator(
  cccl_iterator_t it,
  const std::string& value_type_name, // resolved C++ type of iterator's value
  const std::string& accum_type_name, // accumulator type alias (for pointer fallback)
  const std::string& struct_name, // e.g., "in_0_it_t"
  const std::string& var_name, // e.g., "in_0"
  const std::string& state_param); // e.g., "d_in_0" (void* param name)

// Generate code for an output iterator.
IteratorCode make_output_iterator(
  cccl_iterator_t it,
  const std::string& accum_type_name,
  const std::string& struct_name,
  const std::string& var_name,
  const std::string& state_param);

} // namespace clangjit::codegen
