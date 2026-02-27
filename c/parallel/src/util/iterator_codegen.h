//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

//
// This header intentionally avoids C++ stdlib (no <string>, <vector>,
// <format>, <atomic>, <stdexcept>, etc.).  Only C standard headers and
// existing CCCL-internal headers (which already have their own stdlib usage)
// are included.
//

#include <stdarg.h>
#include <stdio.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <cuda.h>

#include <cccl/c/iterators.h>
#include <cccl/c/types.h>
#include <nvrtc/command_list.h>
#include <util/errors.h>
#include <util/types.h>

namespace cccl::iterators
{

// ---------------------------------------------------------------------------
// CUDA device code preamble used in all iterator kernels
// ---------------------------------------------------------------------------

static constexpr const char* CUDA_PREAMBLE = "#include <cuda/std/cstdint>\n"
                                             "#include <cuda_fp16.h>\n"
                                             "#include <cuda/std/cstring>\n"
                                             "using namespace cuda::std;\n";

// ---------------------------------------------------------------------------
// Simple growable string buffer (no stdlib)
// ---------------------------------------------------------------------------

struct strbuf
{
  char* data = nullptr;
  std::size_t len  = 0;
  std::size_t cap  = 0;

  strbuf() = default;

  ~strbuf()
  {
    std::free(data);
  }

  // Non-copyable, movable would complicate lifetime; just use in-place.
  strbuf(const strbuf&)            = delete;
  strbuf& operator=(const strbuf&) = delete;

  CUresult reserve(std::size_t extra)
  {
    if (len + extra + 1 <= cap)
    {
      return CUDA_SUCCESS;
    }
    std::size_t new_cap = (cap + extra + 1) * 2;
    if (new_cap < 512)
    {
      new_cap = 512;
    }
    char* p = static_cast<char*>(std::realloc(data, new_cap));
    if (!p)
    {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    data = p;
    cap  = new_cap;
    return CUDA_SUCCESS;
  }

  CUresult append(const char* s)
  {
    std::size_t slen = std::strlen(s);
    CUresult r       = reserve(slen);
    if (r != CUDA_SUCCESS)
    {
      return r;
    }
    std::memcpy(data + len, s, slen);
    len += slen;
    data[len] = '\0';
    return CUDA_SUCCESS;
  }

  CUresult appendf(const char* fmt, ...) __attribute__((format(printf, 2, 3)))
  {
    va_list args;
    va_start(args, fmt);
    int needed = vsnprintf(nullptr, 0, fmt, args);
    va_end(args);
    if (needed < 0)
    {
      return CUDA_ERROR_UNKNOWN;
    }

    CUresult r = reserve(static_cast<std::size_t>(needed));
    if (r != CUDA_SUCCESS)
    {
      return r;
    }

    va_start(args, fmt);
    vsnprintf(data + len, static_cast<std::size_t>(needed) + 1, fmt, args);
    va_end(args);
    len += static_cast<std::size_t>(needed);
    return CUDA_SUCCESS;
  }

  const char* c_str() const
  {
    return data ? data : "";
  }
};

// ---------------------------------------------------------------------------
// Unique ID counter for compound iterator symbol names (no std::atomic)
// ---------------------------------------------------------------------------

inline int next_unique_id()
{
  static int counter = 0;
  return __atomic_fetch_add(&counter, 1, __ATOMIC_RELAXED);
}

// ---------------------------------------------------------------------------
// LTOIR compilation
// ---------------------------------------------------------------------------

/**
 * @brief Compile a C++ device source string to LTOIR.
 *
 * Returns CUDA_SUCCESS on success.  On success, writes the LTOIR pointer
 * (heap-allocated; caller must free) and size into *ltoir_out / *size_out.
 *
 * @param source           Null-terminated C++ device source
 * @param cc_major         Compute capability major
 * @param cc_minor         Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx (may be nullptr)
 * @param ctk_path         Include flag for CUDA toolkit headers (may be nullptr)
 * @param ltoir_out        Receives heap-allocated LTOIR bytes (caller frees)
 * @param size_out         Receives LTOIR byte count
 */
inline CUresult compile_iterator_source(
  const char* source,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path,
  char** ltoir_out,
  std::size_t* size_out)
{
  char arch_buf[32];
  std::snprintf(arch_buf, sizeof(arch_buf), "-arch=sm_%d%d", cc_major, cc_minor);

  // Build args array (fixed-size; plenty for our use)
  const char* args[8];
  int num_args = 0;
  args[num_args++] = arch_buf;
  if (libcudacxx_path)
  {
    args[num_args++] = libcudacxx_path;
  }
  if (ctk_path)
  {
    args[num_args++] = ctk_path;
  }
  args[num_args++] = "-rdc=true";
  args[num_args++] = "-dlto";

  // Use existing command_list infrastructure (which uses stdlib internally —
  // that's pre-existing and out of scope for this change).
  auto [ltoir_size, ltoir_data] = begin_linking_nvrtc_program(0, nullptr)
                                    ->add_program(nvrtc_translation_unit{source, "iterator_codegen"})
                                    ->compile_program({args, static_cast<std::size_t>(num_args)})
                                    ->get_program_ltoir();

  // Copy out of unique_ptr into a plain malloc'd buffer so callers don't
  // need stdlib to manage the lifetime.
  char* out = static_cast<char*>(std::malloc(ltoir_size));
  if (!out)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(out, ltoir_data.get(), ltoir_size);
  *ltoir_out = out;
  *size_out  = ltoir_size;
  return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------
// op helpers
// ---------------------------------------------------------------------------

/**
 * @brief (data, size) pair used for extra LTOIR references.
 */
struct ltoir_ref
{
  const char* data;
  std::size_t size;
};

/**
 * @brief Fill a cccl_op_t from compiled LTOIR and optional extra LTOIRs.
 *
 * All strings/buffers are deep-copied.  Call free_op() to release.
 *
 * @param op           Output op (zeroed before writing)
 * @param name         Symbol name
 * @param ltoir_data   Primary LTOIR bytes
 * @param ltoir_size   Primary LTOIR size
 * @param extras       Array of extra LTOIR (data, size) pairs — may be nullptr
 * @param n_extras     Number of entries in extras[]
 */
inline CUresult fill_op(
  cccl_op_t* op,
  const char* name,
  const char* ltoir_data,
  std::size_t ltoir_size,
  const ltoir_ref* extras = nullptr,
  std::size_t n_extras    = 0)
{
  std::memset(op, 0, sizeof(*op));
  op->type      = CCCL_STATELESS;
  op->code_type = CCCL_OP_LTOIR;

  // Copy name
  std::size_t nlen = std::strlen(name) + 1;
  char* name_copy  = static_cast<char*>(std::malloc(nlen));
  if (!name_copy)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(name_copy, name, nlen);
  op->name = name_copy;

  // Copy primary LTOIR
  char* code_copy = static_cast<char*>(std::malloc(ltoir_size));
  if (!code_copy)
  {
    std::free(name_copy);
    op->name = nullptr;
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(code_copy, ltoir_data, ltoir_size);
  op->code      = code_copy;
  op->code_size = ltoir_size;

  // Copy extra LTOIRs
  if (n_extras > 0 && extras)
  {
    op->extra_ltoirs      = static_cast<const char**>(std::malloc(n_extras * sizeof(const char*)));
    op->extra_ltoir_sizes = static_cast<std::size_t*>(std::malloc(n_extras * sizeof(std::size_t)));
    if (!op->extra_ltoirs || !op->extra_ltoir_sizes)
    {
      // Partially constructed; free_op will clean up what's set
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    op->num_extra_ltoirs = n_extras;
    for (std::size_t i = 0; i < n_extras; ++i)
    {
      char* copy = static_cast<char*>(std::malloc(extras[i].size));
      if (!copy)
      {
        // Mark remaining entries as null so free_op handles partial init
        for (std::size_t j = i; j < n_extras; ++j)
        {
          op->extra_ltoirs[j]      = nullptr;
          op->extra_ltoir_sizes[j] = 0;
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      std::memcpy(copy, extras[i].data, extras[i].size);
      op->extra_ltoirs[i]      = copy;
      op->extra_ltoir_sizes[i] = extras[i].size;
    }
  }

  return CUDA_SUCCESS;
}

/**
 * @brief Free all memory owned by a cccl_op_t filled by fill_op().
 */
inline void free_op(cccl_op_t* op)
{
  if (!op)
  {
    return;
  }
  std::free(const_cast<char*>(op->name));
  std::free(const_cast<char*>(op->code));
  for (std::size_t i = 0; i < op->num_extra_ltoirs; ++i)
  {
    std::free(const_cast<char*>(op->extra_ltoirs[i]));
  }
  std::free(const_cast<const char**>(op->extra_ltoirs));
  std::free(op->extra_ltoir_sizes);
  std::memset(op, 0, sizeof(*op));
}

/**
 * @brief Collect all LTOIRs from a cccl_op_t into a caller-supplied array.
 *
 * @param op        Source op
 * @param out       Output array of ltoir_ref — must have room for at least
 *                  1 + op.num_extra_ltoirs entries
 * @param n_out     Receives the number of entries written
 */
inline void collect_op_ltoirs(const cccl_op_t& op, ltoir_ref* out, std::size_t* n_out)
{
  std::size_t n = 0;
  if (op.code && op.code_size > 0)
  {
    out[n++] = {op.code, op.code_size};
  }
  for (std::size_t i = 0; i < op.num_extra_ltoirs; ++i)
  {
    if (op.extra_ltoirs[i] && op.extra_ltoir_sizes[i] > 0)
    {
      out[n++] = {op.extra_ltoirs[i], op.extra_ltoir_sizes[i]};
    }
  }
  *n_out = n;
}

/**
 * @brief Return the C++ intrinsic string for a cache modifier.
 *
 * Returns nullptr for unsupported modifiers.
 */
inline const char* cache_modifier_intrinsic(cccl_cache_modifier_t modifier)
{
  switch (modifier)
  {
    case CCCL_LOAD_CS:
      return "__ldcs";
    case CCCL_LOAD_CG:
      return "__ldcg";
    case CCCL_LOAD_CV:
      return "__ldcv";
    default:
      return nullptr;
  }
}

} // namespace cccl::iterators
