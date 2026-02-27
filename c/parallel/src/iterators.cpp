//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//
// This file intentionally avoids C++ stdlib (no <string>, <vector>,
// <format>, <atomic>, <stdexcept>, etc.).  See util/iterator_codegen.h.
//

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <cccl/c/iterators.h>
#include <cccl/c/types.h>
#include <util/iterator_codegen.h>
#include <util/types.h>

using namespace cccl::iterators;

// ============================================================================
// Internal helpers
// ============================================================================

// Maximum extra LTOIRs we handle for any single op.
// For zip: up to 32 children, each with up to 4 extra LTOIRs = 128 + 32 primary.
static constexpr std::size_t MAX_EXTRAS = 256;

// Maximum supported zip/permutation children (state offsets, value offsets).
static constexpr std::size_t MAX_ZIP_CHILDREN = 32;

/**
 * Copy `size` bytes from `src` into a fresh malloc buffer stored in
 * `out->state`.  Sets `out->size` and `out->alignment`.
 */
static CUresult set_state(cccl_iterator_t* out, const void* src, std::size_t size, std::size_t alignment)
{
  if (src && size > 0)
  {
    out->state = std::malloc(size);
    if (!out->state)
    {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    std::memcpy(out->state, src, size);
  }
  else
  {
    out->state = nullptr;
  }
  out->size      = size;
  out->alignment = alignment;
  return CUDA_SUCCESS;
}

// ============================================================================
// cccl_make_counting_iterator
// ============================================================================

CUresult cccl_make_counting_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  auto type_str_counting  = cccl_type_enum_to_name(value_type.type);
  const char* cpp_type    = type_str_counting.c_str();

  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "CountingIterator_advance_%d", static_cast<int>(value_type.type));
  std::snprintf(deref_sym, sizeof(deref_sym), "CountingIterator_deref_%d", static_cast<int>(value_type.type));

  CUresult r = CUDA_SUCCESS;

  // --- Advance ---
  strbuf adv_src;
  r = adv_src.append(CUDA_PREAMBLE);
  if (r != CUDA_SUCCESS) goto fail;
  r = adv_src.appendf(
    "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
    "    auto* s = static_cast<%s*>(state);\n"
    "    auto dist = *static_cast<uint64_t*>(offset);\n"
    "    *s += static_cast<%s>(dist);\n"
    "}\n",
    adv_sym,
    cpp_type,
    cpp_type);
  if (r != CUDA_SUCCESS) goto fail;

  {
    char* adv_ltoir   = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  // --- Dereference ---
  {
    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = deref_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* result) {\n"
      "    *static_cast<%s*>(result) = *static_cast<%s*>(state);\n"
      "}\n",
      deref_sym,
      cpp_type,
      cpp_type);
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = value_type;
  r               = set_state(out, state, value_type.size, value_type.alignment);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_constant_iterator
// ============================================================================

CUresult cccl_make_constant_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  auto type_str_constant  = cccl_type_enum_to_name(value_type.type);
  const char* cpp_type    = type_str_constant.c_str();

  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "ConstantIterator_advance_%d", static_cast<int>(value_type.type));
  std::snprintf(deref_sym, sizeof(deref_sym), "ConstantIterator_deref_%d", static_cast<int>(value_type.type));

  CUresult r = CUDA_SUCCESS;

  // --- Advance (no-op) ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = adv_src.appendf("extern \"C\" __device__ void %s(void*, void*) {}\n", adv_sym);
    if (r != CUDA_SUCCESS) goto fail;

    char* adv_ltoir   = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  // --- Dereference ---
  {
    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = deref_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* result) {\n"
      "    *static_cast<%s*>(result) = *static_cast<%s*>(state);\n"
      "}\n",
      deref_sym,
      cpp_type,
      cpp_type);
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = value_type;
  r               = set_state(out, state, value_type.size, value_type.alignment);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_discard_iterator
// ============================================================================

CUresult cccl_make_discard_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "DiscardIterator_advance_%d", static_cast<int>(value_type.type));
  std::snprintf(deref_sym, sizeof(deref_sym), "DiscardIterator_deref_%d", static_cast<int>(value_type.type));

  CUresult r = CUDA_SUCCESS;

  // All ops are no-ops; both use the same pattern
  auto compile_noop = [&](const char* sym, cccl_op_t* op) -> CUresult {
    strbuf src;
    CUresult ri = src.append(CUDA_PREAMBLE);
    if (ri != CUDA_SUCCESS) return ri;
    ri = src.appendf("extern \"C\" __device__ void %s(void*, void*) {}\n", sym);
    if (ri != CUDA_SUCCESS) return ri;

    char* ltoir   = nullptr;
    std::size_t sz = 0;
    ri = compile_iterator_source(src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &ltoir, &sz);
    if (ri != CUDA_SUCCESS) return ri;
    ri = fill_op(op, sym, ltoir, sz);
    std::free(ltoir);
    return ri;
  };

  r = compile_noop(adv_sym, &out->advance);
  if (r != CUDA_SUCCESS) goto fail;
  r = compile_noop(deref_sym, &out->dereference);
  if (r != CUDA_SUCCESS) goto fail;

  out->type       = CCCL_ITERATOR;
  out->value_type = value_type;
  r               = set_state(out, state, value_type.size, value_type.alignment);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_pointer_iterator
// ============================================================================

CUresult cccl_make_pointer_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int is_input,
  int is_output,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  const bool is_scalar        = (value_type.type != CCCL_STORAGE);
  const std::size_t elem_size = value_type.size;
  auto type_str_pointer       = is_scalar ? cccl_type_enum_to_name(value_type.type) : std::string{};
  const char* cpp_type        = is_scalar ? type_str_pointer.c_str() : nullptr;

  // Suffix encodes both type enum and element size (for STORAGE types)
  char type_suffix[32];
  std::snprintf(type_suffix, sizeof(type_suffix), "%d_s%zu", static_cast<int>(value_type.type), elem_size);

  char adv_sym[80];
  std::snprintf(adv_sym, sizeof(adv_sym), "PointerIterator_advance_%s", type_suffix);

  CUresult r = CUDA_SUCCESS;

  // --- Advance ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;

    if (is_scalar)
    {
      r = adv_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
        "    auto* ptr_state = static_cast<%s**>(state);\n"
        "    auto dist = *static_cast<int64_t*>(offset);\n"
        "    *ptr_state += dist;\n"
        "}\n",
        adv_sym,
        cpp_type);
    }
    else
    {
      r = adv_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
        "    auto* ptr_state = static_cast<char**>(state);\n"
        "    auto dist = *static_cast<int64_t*>(offset);\n"
        "    *ptr_state += dist * %zu;\n"
        "}\n",
        adv_sym,
        elem_size);
    }
    if (r != CUDA_SUCCESS) goto fail;

    char* adv_ltoir   = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  // --- Dereference ---
  if (is_output)
  {
    char out_sym[80];
    std::snprintf(out_sym, sizeof(out_sym), "PointerIterator_out_deref_%s", type_suffix);

    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;

    if (is_scalar)
    {
      r = deref_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* value) {\n"
        "    auto* ptr_state = static_cast<%s**>(state);\n"
        "    **ptr_state = *static_cast<%s*>(value);\n"
        "}\n",
        out_sym,
        cpp_type,
        cpp_type);
    }
    else
    {
      r = deref_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* value) {\n"
        "    auto* ptr_state = static_cast<char**>(state);\n"
        "    memcpy(*ptr_state, value, %zu);\n"
        "}\n",
        out_sym,
        elem_size);
    }
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->dereference, out_sym, deref_ltoir, deref_sz);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }
  else if (is_input)
  {
    char in_sym[80];
    std::snprintf(in_sym, sizeof(in_sym), "PointerIterator_in_deref_%s", type_suffix);

    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;

    if (is_scalar)
    {
      r = deref_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* result) {\n"
        "    auto* ptr_state = static_cast<%s**>(state);\n"
        "    *static_cast<%s*>(result) = **ptr_state;\n"
        "}\n",
        in_sym,
        cpp_type,
        cpp_type);
    }
    else
    {
      r = deref_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* result) {\n"
        "    auto* ptr_state = static_cast<char**>(state);\n"
        "    memcpy(result, *ptr_state, %zu);\n"
        "}\n",
        in_sym,
        elem_size);
    }
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->dereference, in_sym, deref_ltoir, deref_sz);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = value_type;
  // Pointer state: always 8 bytes
  r = set_state(out, state, 8, 8);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_cache_modified_input_iterator
// ============================================================================

CUresult cccl_make_cache_modified_input_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  cccl_cache_modifier_t modifier,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  const char* intrinsic = cache_modifier_intrinsic(modifier);
  if (!intrinsic)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  auto type_str_cache  = cccl_type_enum_to_name(value_type.type);
  const char* cpp_type = type_str_cache.c_str();

  char adv_sym[80], deref_sym[80];
  std::snprintf(adv_sym, sizeof(adv_sym), "CacheModifiedIterator_advance_%d_%d",
                static_cast<int>(value_type.type), static_cast<int>(modifier));
  std::snprintf(deref_sym, sizeof(deref_sym), "CacheModifiedIterator_deref_%d_%d",
                static_cast<int>(value_type.type), static_cast<int>(modifier));

  CUresult r = CUDA_SUCCESS;

  // --- Advance ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = adv_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
      "    auto* s = static_cast<%s**>(state);\n"
      "    auto dist = *static_cast<uint64_t*>(offset);\n"
      "    *s += dist;\n"
      "}\n",
      adv_sym,
      cpp_type);
    if (r != CUDA_SUCCESS) goto fail;

    char* adv_ltoir   = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  // --- Dereference ---
  {
    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = deref_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* result) {\n"
      "    auto* ptr = *static_cast<%s**>(state);\n"
      "    *static_cast<%s*>(result) = %s(ptr);\n"
      "}\n",
      deref_sym,
      cpp_type,
      cpp_type,
      intrinsic);
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;
    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = value_type;
  r               = set_state(out, state, 8, 8);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_reverse_iterator
// ============================================================================

CUresult cccl_make_reverse_iterator(
  cccl_iterator_t* out,
  cccl_iterator_t underlying,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  const int uid = next_unique_id();
  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "ReverseIterator_advance_%d", uid);
  std::snprintf(deref_sym, sizeof(deref_sym), "ReverseIterator_deref_%d", uid);

  const char* child_adv_name   = underlying.advance.name;
  const char* child_deref_name = underlying.dereference.name;

  CUresult r = CUDA_SUCCESS;

  // --- Advance (negate offset, call child) ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = adv_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* offset);\n"
      "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
      "    int64_t neg_offset = -static_cast<int64_t>(*static_cast<uint64_t*>(offset));\n"
      "    %s(state, &neg_offset);\n"
      "}\n",
      child_adv_name,
      adv_sym,
      child_adv_name);
    if (r != CUDA_SUCCESS) goto fail;

    char* adv_ltoir   = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS) goto fail;

    // Collect child advance LTOIRs as extras
    ltoir_ref child_extras[1 + MAX_EXTRAS];
    std::size_t n_child_extras = 0;
    collect_op_ltoirs(underlying.advance, child_extras, &n_child_extras);

    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz, child_extras, n_child_extras);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  // --- Dereference (delegate to child) ---
  {
    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = deref_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* result);\n"
      "extern \"C\" __device__ void %s(void* state, void* result) {\n"
      "    %s(state, result);\n"
      "}\n",
      child_deref_name,
      deref_sym,
      child_deref_name);
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;

    ltoir_ref child_extras[1 + MAX_EXTRAS];
    std::size_t n_child_extras = 0;
    collect_op_ltoirs(underlying.dereference, child_extras, &n_child_extras);

    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz, child_extras, n_child_extras);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = underlying.value_type;
  r               = set_state(out, underlying.state, underlying.size, underlying.alignment);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_zip_iterator
// ============================================================================

CUresult cccl_make_zip_iterator(
  cccl_iterator_t* out,
  const cccl_iterator_t* children,
  const size_t* state_offsets,
  const size_t* value_offsets,
  size_t n_children,
  const void* state,
  size_t state_size,
  size_t state_alignment,
  cccl_type_info value_type,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  if (n_children == 0 || n_children > MAX_ZIP_CHILDREN)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const int uid = next_unique_id();
  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "ZipIterator_advance_%d", uid);
  std::snprintf(deref_sym, sizeof(deref_sym), "ZipIterator_in_deref_%d", uid);

  CUresult r = CUDA_SUCCESS;

  // --- Advance ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;

    // extern declarations
    for (std::size_t i = 0; i < n_children; ++i)
    {
      r = adv_src.appendf("extern \"C\" __device__ void %s(void* state, void* offset);\n",
                           children[i].advance.name);
      if (r != CUDA_SUCCESS) goto fail;
    }

    r = adv_src.appendf("extern \"C\" __device__ void %s(void* state, void* offset) {\n", adv_sym);
    if (r != CUDA_SUCCESS) goto fail;

    for (std::size_t i = 0; i < n_children; ++i)
    {
      r = adv_src.appendf("    %s(static_cast<char*>(state) + %zu, offset);\n",
                           children[i].advance.name, state_offsets[i]);
      if (r != CUDA_SUCCESS) goto fail;
    }
    r = adv_src.append("}\n");
    if (r != CUDA_SUCCESS) goto fail;

    char* adv_ltoir   = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS) goto fail;

    // Gather extra LTOIRs from all child advance ops
    ltoir_ref adv_extras[MAX_EXTRAS];
    std::size_t n_adv_extras = 0;
    for (std::size_t i = 0; i < n_children && n_adv_extras < MAX_EXTRAS; ++i)
    {
      std::size_t nc = 0;
      collect_op_ltoirs(children[i].advance, adv_extras + n_adv_extras, &nc);
      n_adv_extras += nc;
    }

    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz, adv_extras, n_adv_extras);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  // --- Input Dereference ---
  {
    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;

    for (std::size_t i = 0; i < n_children; ++i)
    {
      r = deref_src.appendf("extern \"C\" __device__ void %s(void* state, void* result);\n",
                              children[i].dereference.name);
      if (r != CUDA_SUCCESS) goto fail;
    }

    r = deref_src.appendf("extern \"C\" __device__ void %s(void* state, void* result) {\n", deref_sym);
    if (r != CUDA_SUCCESS) goto fail;

    for (std::size_t i = 0; i < n_children; ++i)
    {
      r = deref_src.appendf(
        "    %s(static_cast<char*>(state) + %zu, static_cast<char*>(result) + %zu);\n",
        children[i].dereference.name,
        state_offsets[i],
        value_offsets[i]);
      if (r != CUDA_SUCCESS) goto fail;
    }
    r = deref_src.append("}\n");
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;

    ltoir_ref deref_extras[MAX_EXTRAS];
    std::size_t n_deref_extras = 0;
    for (std::size_t i = 0; i < n_children && n_deref_extras < MAX_EXTRAS; ++i)
    {
      std::size_t nc = 0;
      collect_op_ltoirs(children[i].dereference, deref_extras + n_deref_extras, &nc);
      n_deref_extras += nc;
    }

    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz, deref_extras, n_deref_extras);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = value_type;
  r               = set_state(out, state, state_size, state_alignment);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_permutation_iterator
// ============================================================================

CUresult cccl_make_permutation_iterator(
  cccl_iterator_t* out,
  cccl_iterator_t values_iter,
  cccl_iterator_t indices_iter,
  size_t values_offset,
  size_t indices_offset,
  const void* state,
  size_t state_size,
  size_t state_alignment,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  const int uid = next_unique_id();
  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "PermutationIterator_advance_%d", uid);
  std::snprintf(deref_sym, sizeof(deref_sym), "PermutationIterator_deref_%d", uid);

  const char* idx_adv_name   = indices_iter.advance.name;
  const char* idx_deref_name = indices_iter.dereference.name;
  const char* val_adv_name   = values_iter.advance.name;
  const char* val_deref_name = values_iter.dereference.name;

  const std::size_t values_state_size  = values_iter.size;
  const std::size_t values_state_align = values_iter.alignment;
  auto idx_type_str                    = cccl_type_enum_to_name(indices_iter.value_type.type);
  const char* idx_cpp_type             = idx_type_str.c_str();

  CUresult r = CUDA_SUCCESS;

  // --- Advance (advance only indices) ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = adv_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* offset);\n"
      "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
      "    char* indices_state = static_cast<char*>(state) + %zu;\n"
      "    %s(indices_state, offset);\n"
      "}\n",
      idx_adv_name,
      adv_sym,
      indices_offset,
      idx_adv_name);
    if (r != CUDA_SUCCESS) goto fail;

    char* adv_ltoir   = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS) goto fail;

    ltoir_ref idx_adv_extras[1 + MAX_EXTRAS];
    std::size_t n_idx_adv = 0;
    collect_op_ltoirs(indices_iter.advance, idx_adv_extras, &n_idx_adv);

    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz, idx_adv_extras, n_idx_adv);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  // --- Dereference (read index, copy values state, advance, deref values) ---
  {
    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS) goto fail;
    r = deref_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* result);\n"
      "extern \"C\" __device__ void %s(void* state, void* offset);\n"
      "extern \"C\" __device__ void %s(void* state, void* result);\n"
      "extern \"C\" __device__ void %s(void* state, void* result) {\n"
      "    char* values_state = static_cast<char*>(state) + %zu;\n"
      "    char* indices_state = static_cast<char*>(state) + %zu;\n"
      "    %s idx;\n"
      "    %s(indices_state, &idx);\n"
      "    alignas(%zu) char temp_values[%zu];\n"
      "    memcpy(temp_values, values_state, %zu);\n"
      "    uint64_t offset = static_cast<uint64_t>(idx);\n"
      "    %s(temp_values, &offset);\n"
      "    %s(temp_values, result);\n"
      "}\n",
      idx_deref_name,
      val_adv_name,
      val_deref_name,
      deref_sym,
      values_offset,
      indices_offset,
      idx_cpp_type,
      idx_deref_name,
      values_state_align,
      values_state_size,
      values_state_size,
      val_adv_name,
      val_deref_name);
    if (r != CUDA_SUCCESS) goto fail;

    char* deref_ltoir   = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS) goto fail;

    // Collect: values advance LTOIRs, indices deref LTOIRs, values deref LTOIRs
    ltoir_ref deref_extras[3 * (1 + MAX_EXTRAS)];
    std::size_t n_deref = 0;
    std::size_t nc      = 0;
    collect_op_ltoirs(values_iter.advance, deref_extras + n_deref, &nc);
    n_deref += nc;
    collect_op_ltoirs(indices_iter.dereference, deref_extras + n_deref, &nc);
    n_deref += nc;
    collect_op_ltoirs(values_iter.dereference, deref_extras + n_deref, &nc);
    n_deref += nc;

    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz, deref_extras, n_deref);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS) goto fail;
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = values_iter.value_type;
  r               = set_state(out, state, state_size, state_alignment);
  if (r != CUDA_SUCCESS) goto fail;
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_shuffle_iterator
// ============================================================================

CUresult cccl_make_shuffle_iterator(
  cccl_iterator_t* out,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  const int uid = next_unique_id();
  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "ShuffleIterator_advance_%d", uid);
  std::snprintf(deref_sym, sizeof(deref_sym), "ShuffleIterator_deref_%d", uid);

  static constexpr const char* SHUFFLE_STATE_STRUCT =
    "struct ShuffleState {\n"
    "    int64_t current_index;\n"
    "    uint64_t num_items;\n"
    "    uint64_t seed;\n"
    "};\n";

  CUresult r = CUDA_SUCCESS;

  // --- Advance ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = adv_src.append(SHUFFLE_STATE_STRUCT);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = adv_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
      "    auto s = static_cast<ShuffleState*>(state);\n"
      "    auto dist = *static_cast<int64_t*>(offset);\n"
      "    s->current_index += dist;\n"
      "}\n",
      adv_sym);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    char* adv_ltoir    = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
  }

  // --- Dereference ---
  {
    strbuf deref_src;
    r = deref_src.append(
      "#include <cuda/__random/random_bijection.h>\n"
      "#include <cuda/__random/pcg_engine.h>\n");
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = deref_src.append(SHUFFLE_STATE_STRUCT);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    // __noinline__ prevents register pressure spill from feistel_bijection UNROLL_FULL
    r = deref_src.appendf(
      "__device__ __noinline__ int64_t __shuffle_apply_%d("
      "uint64_t num_items, uint64_t seed, uint64_t idx) {\n"
      "    cuda::pcg64 rng(seed);\n"
      "    cuda::random_bijection<uint64_t> bijection(num_items, rng);\n"
      "    return static_cast<int64_t>(bijection(idx));\n"
      "}\n"
      "extern \"C\" __device__ void %s(void* state, void* result) {\n"
      "    const auto* s = static_cast<const ShuffleState*>(state);\n"
      "    *static_cast<int64_t*>(result) = __shuffle_apply_%d(\n"
      "        s->num_items, s->seed, static_cast<uint64_t>(s->current_index));\n"
      "}\n",
      uid,
      deref_sym,
      uid);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    char* deref_ltoir    = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
  }

  out->type                = CCCL_ITERATOR;
  out->value_type.type     = CCCL_INT64;
  out->value_type.size     = 8;
  out->value_type.alignment = 8;
  r                        = set_state(out, state, 24, 8);
  if (r != CUDA_SUCCESS)
  {
    goto fail;
  }
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_make_transform_iterator
// ============================================================================

CUresult cccl_make_transform_iterator(
  cccl_iterator_t* out,
  cccl_iterator_t underlying,
  cccl_op_t transform_op,
  cccl_type_info value_type,
  int is_input,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  std::memset(out, 0, sizeof(*out));

  const int uid = next_unique_id();
  char adv_sym[64], deref_sym[64];
  std::snprintf(adv_sym, sizeof(adv_sym), "TransformIterator_advance_%d", uid);
  std::snprintf(deref_sym, sizeof(deref_sym), "TransformIterator_deref_%d", uid);

  const char* child_adv_name   = underlying.advance.name;
  const char* child_deref_name = underlying.dereference.name;
  const char* transform_name   = transform_op.name;

  CUresult r = CUDA_SUCCESS;

  // --- Advance (call through to underlying advance) ---
  {
    strbuf adv_src;
    r = adv_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = adv_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* offset);\n"
      "extern \"C\" __device__ void %s(void* state, void* offset) {\n"
      "    %s(state, offset);\n"
      "}\n",
      child_adv_name,
      adv_sym,
      child_adv_name);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    char* adv_ltoir    = nullptr;
    std::size_t adv_sz = 0;
    r = compile_iterator_source(adv_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &adv_ltoir, &adv_sz);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    ltoir_ref adv_extras[1 + MAX_EXTRAS];
    std::size_t n_adv_extras = 0;
    collect_op_ltoirs(underlying.advance, adv_extras, &n_adv_extras);

    r = fill_op(&out->advance, adv_sym, adv_ltoir, adv_sz, adv_extras, n_adv_extras);
    std::free(adv_ltoir);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
  }

  // --- Dereference ---
  {
    strbuf deref_src;
    r = deref_src.append(CUDA_PREAMBLE);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
    r = deref_src.appendf(
      "extern \"C\" __device__ void %s(void* state, void* result);\n"
      "extern \"C\" __device__ void %s(void* input, void* output);\n",
      child_deref_name,
      transform_name);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    // Temp variable declaration for the underlying value type
    cccl_type_info underlying_vtype = underlying.value_type;
    if (underlying_vtype.type == CCCL_STORAGE)
    {
      r = deref_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* result) {\n"
        "    alignas(%zu) char temp[%zu];\n",
        deref_sym,
        underlying_vtype.alignment,
        underlying_vtype.size);
    }
    else
    {
      auto temp_type_str = cccl_type_enum_to_name(underlying_vtype.type);
      r               = deref_src.appendf(
        "extern \"C\" __device__ void %s(void* state, void* result) {\n"
        "    %s temp;\n",
        deref_sym,
        temp_type_str.c_str());
    }
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    if (is_input)
    {
      r = deref_src.appendf(
        "    %s(state, &temp);\n"
        "    %s(&temp, result);\n"
        "}\n",
        child_deref_name,
        transform_name);
    }
    else
    {
      r = deref_src.appendf(
        "    %s(result, &temp);\n"
        "    %s(state, &temp);\n"
        "}\n",
        transform_name,
        child_deref_name);
    }
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    char* deref_ltoir    = nullptr;
    std::size_t deref_sz = 0;
    r = compile_iterator_source(
      deref_src.c_str(), cc_major, cc_minor, libcudacxx_path, ctk_path, &deref_ltoir, &deref_sz);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }

    ltoir_ref deref_extras[2 + 2 * MAX_EXTRAS];
    std::size_t n_deref_extras = 0;
    collect_op_ltoirs(transform_op, deref_extras, &n_deref_extras);
    std::size_t nc = 0;
    collect_op_ltoirs(underlying.dereference, deref_extras + n_deref_extras, &nc);
    n_deref_extras += nc;

    r = fill_op(&out->dereference, deref_sym, deref_ltoir, deref_sz, deref_extras, n_deref_extras);
    std::free(deref_ltoir);
    if (r != CUDA_SUCCESS)
    {
      goto fail;
    }
  }

  out->type       = CCCL_ITERATOR;
  out->value_type = value_type;
  r               = set_state(out, underlying.state, underlying.size, underlying.alignment);
  if (r != CUDA_SUCCESS)
  {
    goto fail;
  }
  return CUDA_SUCCESS;

fail:
  cccl_destroy_iterator(out);
  return r;
}

// ============================================================================
// cccl_destroy_iterator
// ============================================================================

void cccl_destroy_iterator(cccl_iterator_t* it)
{
  if (!it)
  {
    return;
  }
  free_op(&it->advance);
  free_op(&it->dereference);
  std::free(it->state);
  it->state = nullptr;
  it->size  = 0;
}
