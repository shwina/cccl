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

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <stddef.h>

#include <cuda.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

/**
 * @brief Cache modifier values for CacheModifiedInputIterator.
 */
typedef enum cccl_cache_modifier_t
{
  CCCL_LOAD_DEFAULT  = 0,
  CCCL_LOAD_CA       = 1,
  CCCL_LOAD_CG       = 2,
  CCCL_LOAD_CS       = 3,
  CCCL_LOAD_CV       = 4,
  CCCL_LOAD_LDG      = 5,
  CCCL_LOAD_VOLATILE = 6,
} cccl_cache_modifier_t;

/**
 * @brief Construct a counting iterator.
 *
 * A counting iterator yields a sequence of values starting at `state` and
 * incrementing by 1 on each advance.
 *
 * @param out          Output iterator struct (caller-owned, call
 *                     cccl_destroy_iterator to free)
 * @param value_type   Type info for the value type
 * @param state        Pointer to initial value bytes (value_type.size bytes)
 * @param cc_major     Compute capability major
 * @param cc_minor     Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx (e.g. "-I/path/libcudacxx")
 * @param ctk_path     Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_counting_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Construct a constant iterator.
 *
 * A constant iterator yields the same value on every dereference.
 *
 * @param out          Output iterator struct
 * @param value_type   Type info for the value type
 * @param state        Pointer to the constant value bytes (value_type.size bytes)
 * @param cc_major     Compute capability major
 * @param cc_minor     Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path     Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_constant_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Construct a discard iterator.
 *
 * A discard iterator silently drops all reads and writes.
 *
 * @param out          Output iterator struct
 * @param value_type   Type info for the value type (used to size state)
 * @param state        Pointer to state bytes (value_type.size bytes, may be zeroes)
 * @param cc_major     Compute capability major
 * @param cc_minor     Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path     Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_discard_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Construct a pointer iterator.
 *
 * A pointer iterator wraps a device pointer, supporting both input and output.
 * For scalar types (non-CCCL_STORAGE), typed pointer arithmetic is used.
 * For struct types (CCCL_STORAGE), byte-level pointer arithmetic is used.
 *
 * @param out          Output iterator struct
 * @param value_type   Type info for the value type
 * @param state        Pointer to an 8-byte pointer value
 * @param is_input     Non-zero to include input dereference, zero to skip
 * @param is_output    Non-zero to include output dereference, zero to skip
 * @param cc_major     Compute capability major
 * @param cc_minor     Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path     Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_pointer_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  int is_input,
  int is_output,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Construct a cache-modified input iterator.
 *
 * Uses PTX cache modifiers (cs/cg/cv) for loads.
 * Only supports types of size 1, 2, 4, 8, or 16 bytes.
 *
 * @param out          Output iterator struct
 * @param value_type   Type info for the value type
 * @param state        Pointer to an 8-byte pointer value
 * @param modifier     Cache modifier: CCCL_LOAD_CS, CCCL_LOAD_CG, or CCCL_LOAD_CV
 * @param cc_major     Compute capability major
 * @param cc_minor     Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path     Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_cache_modified_input_iterator(
  cccl_iterator_t* out,
  cccl_type_info value_type,
  const void* state,
  cccl_cache_modifier_t modifier,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Construct a reverse iterator wrapping an underlying iterator.
 *
 * Advancing with a positive offset moves backward in the underlying iterator.
 * The underlying iterator's op names and LTOIRs are embedded in the new ops.
 *
 * @param out          Output iterator struct
 * @param underlying   Fully constructed underlying iterator
 * @param cc_major     Compute capability major
 * @param cc_minor     Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path     Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_reverse_iterator(
  cccl_iterator_t* out,
  cccl_iterator_t underlying,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Construct a zip iterator from multiple child iterators.
 *
 * At each position, yields values from all child iterators packed into a struct.
 * The value_type should be CCCL_STORAGE with a size equal to the sum of child
 * value sizes (plus padding for alignment).
 *
 * @param out            Output iterator struct
 * @param children       Array of pre-constructed child iterator structs
 * @param state_offsets  Byte offsets of each child's state within the combined state
 * @param value_offsets  Byte offsets of each child's value within the combined value
 * @param n_children     Number of child iterators
 * @param state          Combined state bytes (state_size bytes)
 * @param state_size     Total combined state size in bytes
 * @param state_alignment  Alignment of the combined state
 * @param value_type     Type info for the combined value (typically CCCL_STORAGE)
 * @param cc_major       Compute capability major
 * @param cc_minor       Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path       Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_zip_iterator(
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
  const char* ctk_path);

/**
 * @brief Construct a permutation iterator.
 *
 * At position i, yields values[indices[i]]. Only the indices iterator advances;
 * the values iterator is used for random access.
 *
 * @param out              Output iterator struct
 * @param values_iter      Fully constructed values iterator
 * @param indices_iter     Fully constructed indices iterator
 * @param values_offset    Byte offset of values state within the combined state
 * @param indices_offset   Byte offset of indices state within the combined state
 * @param state            Combined state bytes
 * @param state_size       Combined state size in bytes
 * @param state_alignment  Alignment of combined state
 * @param cc_major         Compute capability major
 * @param cc_minor         Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path         Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_permutation_iterator(
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
  const char* ctk_path);

/**
 * @brief Construct a shuffle iterator.
 *
 * At position i, yields a deterministic random permutation of [0, num_items)
 * parameterized by seed.  The state is a 24-byte struct laid out as:
 *   { int64_t current_index; uint64_t num_items; uint64_t seed; }
 *
 * @param out          Output iterator struct
 * @param state        Pointer to the 24-byte ShuffleState
 * @param cc_major     Compute capability major
 * @param cc_minor     Compute capability minor
 * @param libcudacxx_path  Include flag for libcudacxx
 * @param ctk_path     Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_shuffle_iterator(
  cccl_iterator_t* out,
  const void* state,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Construct a transform iterator.
 *
 * For input (is_input != 0): reads from underlying, applies transform, yields result.
 * For output (is_input == 0): applies transform to input value, writes to underlying.
 *
 * The transform_op must have its name and LTOIR set; any extra LTOIRs it carries
 * are automatically linked in.  The underlying iterator's dereference op must be
 * configured for the desired direction (input or output) before calling this function.
 *
 * @param out                  Output iterator struct
 * @param underlying           Fully constructed underlying iterator
 *                             (dereference = input or output deref as appropriate)
 * @param transform_op         User-compiled transform operation (name + LTOIR)
 * @param value_type           Type info for the iterator's value type
 *                             (output type for input iterator, input type for output)
 * @param is_input             Non-zero for input iterator, zero for output
 * @param cc_major             Compute capability major
 * @param cc_minor             Compute capability minor
 * @param libcudacxx_path      Include flag for libcudacxx
 * @param ctk_path             Include flag for CUDA toolkit headers
 */
CCCL_C_API CUresult cccl_make_transform_iterator(
  cccl_iterator_t* out,
  cccl_iterator_t underlying,
  cccl_op_t transform_op,
  cccl_type_info value_type,
  int is_input,
  int cc_major,
  int cc_minor,
  const char* libcudacxx_path,
  const char* ctk_path);

/**
 * @brief Free all memory owned by a C-constructed iterator.
 *
 * Must be called for any iterator created with a cccl_make_* function.
 * Sets all freed pointers to NULL. Safe to call with NULL or on an already-
 * destroyed iterator (no-op for NULL pointers).
 *
 * @param it   Pointer to the iterator to destroy. May be NULL.
 */
CCCL_C_API void cccl_destroy_iterator(cccl_iterator_t* it);

CCCL_C_EXTERN_C_END
