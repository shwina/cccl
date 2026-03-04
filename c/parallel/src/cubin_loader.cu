//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <new>

#include <cuda.h>

#include "util/cubin_loader.h"

CUresult cccl_load_cubin_and_get_kernels(
  const void* cubin_in,
  size_t cubin_size,
  void** cubin_copy_out,
  CUlibrary* library_out,
  const char** kernel_names,
  CUkernel* kernel_handles,
  int num_kernels)
{
  *cubin_copy_out = nullptr;
  *library_out    = nullptr;

  if (cubin_in == nullptr || cubin_size == 0)
  {
    return CUDA_ERROR_INVALID_VALUE;
  }

  char* cubin_copy = new (std::nothrow) char[cubin_size];
  if (!cubin_copy)
  {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(cubin_copy, cubin_in, cubin_size);

  CUresult status = cuLibraryLoadData(library_out, cubin_copy, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (status != CUDA_SUCCESS)
  {
    delete[] cubin_copy;
    return status;
  }

  for (int i = 0; i < num_kernels; ++i)
  {
    if (kernel_names[i] == nullptr)
    {
      kernel_handles[i] = nullptr;
      continue;
    }
    status = cuLibraryGetKernel(&kernel_handles[i], *library_out, kernel_names[i]);
    if (status != CUDA_SUCCESS)
    {
      cuLibraryUnload(*library_out);
      *library_out = nullptr;
      delete[] cubin_copy;
      return status;
    }
  }

  // Eagerly call cuKernelGetFunction for each kernel.
  // This triggers the one-time cubin→SASS finalisation now (at load time)
  // rather than deferring it to the first kernel launch.  The CUfunction is
  // discarded; CUDA caches it internally so the execution path's own
  // cuKernelGetFunction call becomes a free cache hit.
  //
  // If there is no active CUDA context yet (e.g. the cubin is being loaded
  // at Python import time before any CUDA API has been called), skip the
  // warm-up gracefully — the first kernel launch will pay the cost instead.
  CUdevice dev;
  if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
  {
    for (int i = 0; i < num_kernels; ++i)
    {
      if (kernel_handles[i] == nullptr)
      {
        continue;
      }
      CUfunction func;
      status = cuKernelGetFunction(&func, kernel_handles[i]);
      if (status != CUDA_SUCCESS)
      {
        cuLibraryUnload(*library_out);
        *library_out = nullptr;
        delete[] cubin_copy;
        return status;
      }
    }
  }

  *cubin_copy_out = cubin_copy;
  return CUDA_SUCCESS;
}
