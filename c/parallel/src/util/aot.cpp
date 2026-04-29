//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "aot.h"

#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// AotWriter
// ---------------------------------------------------------------------------

AotWriter::AotWriter(const char* path)
{
  fp_ = std::fopen(path, "wb");
  if (!fp_)
  {
    throw std::runtime_error(std::string("AotWriter: cannot open file for writing: ") + path);
  }
}

AotWriter::~AotWriter()
{
  if (fp_)
  {
    std::fclose(fp_);
    fp_ = nullptr;
  }
}

void AotWriter::write_raw(const void* data, size_t size)
{
  if (size == 0)
  {
    return;
  }
  size_t written = std::fwrite(data, 1, size, fp_);
  if (written != size)
  {
    throw std::runtime_error("AotWriter: short write");
  }
}

void AotWriter::write_header(CclbTag tag)
{
  // magic: 4 bytes
  write_raw("CCLB", 4);
  // version: uint32_t = 1
  uint32_t version = 1;
  write_raw(&version, sizeof(version));
  // tag: uint32_t enum value
  uint32_t tag_val = static_cast<uint32_t>(tag);
  write_raw(&tag_val, sizeof(tag_val));
}

void AotWriter::write_i32(int32_t v)
{
  write_raw(&v, sizeof(v));
}

void AotWriter::write_u32(uint32_t v)
{
  write_raw(&v, sizeof(v));
}

void AotWriter::write_u64(uint64_t v)
{
  write_raw(&v, sizeof(v));
}

void AotWriter::write_bool(bool v)
{
  uint8_t b = v ? 1 : 0;
  write_raw(&b, sizeof(b));
}

void AotWriter::write_blob(const void* data, uint64_t size)
{
  write_u64(size);
  if (size > 0 && data != nullptr)
  {
    write_raw(data, static_cast<size_t>(size));
  }
}

void AotWriter::write_string(const char* s)
{
  uint16_t len = (s != nullptr) ? static_cast<uint16_t>(std::strlen(s)) : 0;
  write_raw(&len, sizeof(len));
  if (len > 0)
  {
    write_raw(s, len);
  }
}

void AotWriter::write_type_info(cccl_type_info ti)
{
  uint64_t sz  = static_cast<uint64_t>(ti.size);
  uint64_t aln = static_cast<uint64_t>(ti.alignment);
  int32_t tp   = static_cast<int32_t>(ti.type);
  write_raw(&sz, sizeof(sz));
  write_raw(&aln, sizeof(aln));
  write_raw(&tp, sizeof(tp));
}

// ---------------------------------------------------------------------------
// AotReader
// ---------------------------------------------------------------------------

AotReader::AotReader(const char* path)
{
  fp_ = std::fopen(path, "rb");
  if (!fp_)
  {
    throw std::runtime_error(std::string("AotReader: cannot open file for reading: ") + path);
  }
  // Validate magic
  char magic[4];
  read_raw(magic, 4);
  if (std::memcmp(magic, "CCLB", 4) != 0)
  {
    std::fclose(fp_);
    fp_ = nullptr;
    throw std::runtime_error("AotReader: bad magic (not a CCLB file)");
  }
  // Validate version
  uint32_t version = 0;
  read_raw(&version, sizeof(version));
  if (version != 1)
  {
    std::fclose(fp_);
    fp_ = nullptr;
    throw std::runtime_error(std::string("AotReader: unsupported version: ") + std::to_string(version));
  }
}

AotReader::~AotReader()
{
  if (fp_)
  {
    std::fclose(fp_);
    fp_ = nullptr;
  }
}

void AotReader::read_raw(void* buf, size_t size)
{
  if (size == 0)
  {
    return;
  }
  size_t got = std::fread(buf, 1, size, fp_);
  if (got != size)
  {
    throw std::runtime_error("AotReader: short read (truncated file?)");
  }
}

CclbTag AotReader::read_tag()
{
  uint32_t val = 0;
  read_raw(&val, sizeof(val));
  return static_cast<CclbTag>(val);
}

int32_t AotReader::read_i32()
{
  int32_t v = 0;
  read_raw(&v, sizeof(v));
  return v;
}

uint32_t AotReader::read_u32()
{
  uint32_t v = 0;
  read_raw(&v, sizeof(v));
  return v;
}

uint64_t AotReader::read_u64()
{
  uint64_t v = 0;
  read_raw(&v, sizeof(v));
  return v;
}

bool AotReader::read_bool()
{
  uint8_t b = 0;
  read_raw(&b, sizeof(b));
  return b != 0;
}

void* AotReader::read_blob(uint64_t* size_out)
{
  uint64_t sz = read_u64();
  *size_out   = sz;
  if (sz == 0)
  {
    return nullptr;
  }
  void* buf = std::malloc(static_cast<size_t>(sz));
  if (!buf)
  {
    throw std::runtime_error("AotReader: malloc failed for blob");
  }
  read_raw(buf, static_cast<size_t>(sz));
  return buf;
}

char* AotReader::read_string_heap()
{
  uint16_t len = 0;
  read_raw(&len, sizeof(len));
  if (len == 0)
  {
    return nullptr;
  }
  char* buf = static_cast<char*>(std::malloc(static_cast<size_t>(len) + 1));
  if (!buf)
  {
    throw std::runtime_error("AotReader: malloc failed for string");
  }
  read_raw(buf, len);
  buf[len] = '\0';
  return buf;
}

cccl_type_info AotReader::read_type_info()
{
  uint64_t sz  = read_u64();
  uint64_t aln = read_u64();
  int32_t tp   = read_i32();
  cccl_type_info ti;
  ti.size      = static_cast<size_t>(sz);
  ti.alignment = static_cast<size_t>(aln);
  ti.type      = static_cast<cccl_type_enum>(tp);
  return ti;
}
