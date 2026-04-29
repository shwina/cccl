//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#ifndef CCCL_C_EXPERIMENTAL
#  define CCCL_C_EXPERIMENTAL
#endif
#include <cccl/c/types.h>

enum class CclbTag : uint32_t
{
  reduce              = 0,
  segmented_reduce    = 1,
  binary_search       = 2,
  scan                = 3,
  unique_by_key       = 4,
  merge_sort          = 5,
  transform           = 6,
  radix_sort          = 7,
  histogram           = 8,
  segmented_sort      = 9,
  three_way_partition = 10,
  for_each            = 11,
};

// Writer: opens file, writes fields in sequence, closes on destruction.
struct AotWriter
{
  FILE* fp_;
  explicit AotWriter(const char* path);
  ~AotWriter();

  void write_header(CclbTag tag); // magic + version + tag
  void write_i32(int32_t v);
  void write_u32(uint32_t v);
  void write_u64(uint64_t v);
  void write_bool(bool v);
  void write_blob(const void* data, uint64_t size); // writes u64 size then bytes
  void write_string(const char* s); // writes u16 length + chars (null ptr → length 0)
  void write_type_info(cccl_type_info ti); // u64 size, u64 alignment, i32 type

private:
  void write_raw(const void* data, size_t size);
};

// Reader: opens file, validates magic+version, reads fields in sequence.
struct AotReader
{
  FILE* fp_;
  explicit AotReader(const char* path); // throws std::runtime_error if bad magic/version
  ~AotReader();

  CclbTag read_tag();
  int32_t read_i32();
  uint32_t read_u32();
  uint64_t read_u64();
  bool read_bool();
  // Reads a size-prefixed blob. Returns malloc'd pointer + size.
  // Caller must free() the returned pointer.
  void* read_blob(uint64_t* size_out);
  // Reads a length-prefixed string. Returns malloc'd null-terminated string.
  // Returns nullptr if stored length was 0.
  char* read_string_heap();
  cccl_type_info read_type_info();

private:
  void read_raw(void* buf, size_t size);
};
