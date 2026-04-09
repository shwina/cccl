#pragma once

#include <string>
#include <variant>
#include <vector>

#include <cccl/c/types.h>
#include <clangjit/config.hpp>
#include <clangjit/jit_compiler.hpp>

namespace clangjit::codegen
{
// Tags for non-cccl arguments (no runtime data, just control code generation)
struct temp_storage_t
{};
struct temp_bytes_t
{};
// num_items_t carries a name so the same tag type can express num_segments,
// num_needles, etc. — each becomes its own unsigned long long parameter.
struct num_items_t
{
  const char* name = "num_items";
};
struct stream_t
{};

inline constexpr temp_storage_t temp_storage{};
inline constexpr temp_bytes_t temp_bytes{};
inline constexpr num_items_t num_items{};
inline constexpr num_items_t num_segments{"num_segments"};
inline constexpr num_items_t num_needles{"num_needles"};
inline constexpr num_items_t num_haystack{"num_haystack"};
inline constexpr stream_t stream{};

// Direction wrappers for iterators (cccl_iterator_t doesn't encode direction)
struct input_t
{
  cccl_iterator_t it;
};
struct output_t
{
  cccl_iterator_t it;
};

inline input_t in(cccl_iterator_t it)
{
  return {it};
}
inline output_t out(cccl_iterator_t it)
{
  return {it};
}

// cmp_t: wraps a cccl_op_t that should generate a comparison functor
// (bool operator()(const T&, const T&)) rather than the default binary reduce
// functor (T operator()(T, T)).  Use cmp(op) where sort/search operators go.
struct cmp_t
{
  cccl_op_t op;
};
inline cmp_t cmp(cccl_op_t op)
{
  return {op};
}

// Argument variant: everything that can appear in .with()
using Arg =
  std::variant<temp_storage_t, temp_bytes_t, num_items_t, stream_t, input_t, output_t, cccl_op_t, cmp_t, cccl_value_t>;

// Result of a successful compilation.
struct CubCallResult
{
  JITCompiler* compiler; // caller takes ownership
  void* fn_ptr; // the exported function
  std::vector<char> cubin; // for SASS inspection
};

class CubCall
{
public:
  // Start building: specify the CUB header to include.
  static CubCall from(const char* include_header);

  // Specify the CUB function to call (e.g., "cub::DeviceReduce::Reduce").
  CubCall& run(const char* cub_function);

  // Optionally override the exported function name (default: "cccl_jit_fn").
  CubCall& name(const char* export_name);

  // Add arguments in CUB call order. Each argument is dispatched by type.
  template <typename... Args>
  CubCall& with(Args&&... args)
  {
    (args_.emplace_back(Arg{std::forward<Args>(args)}), ...);
    return *this;
  }

  // Generate the complete CUDA source string (useful for debugging).
  std::string source() const;

  // Compile the generated source and return the function pointer.
  CubCallResult compile(
    int cc_major,
    int cc_minor,
    const char* clang_path        = nullptr,
    cccl_build_config* config     = nullptr,
    const char* ctk_path          = nullptr,
    const char* cccl_include_path = nullptr) const;

private:
  std::string include_;
  std::string cub_function_;
  std::string fn_name_ = "cccl_jit_fn";
  std::vector<Arg> args_;
};
} // namespace clangjit::codegen
