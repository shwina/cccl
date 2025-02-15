//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_EXTENDED_FLOATING_POINT_H
#define __CCCL_EXTENDED_FLOATING_POINT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/diagnostic.h>
#include <cuda/std/__cccl/preprocessor.h>

#if !defined(_CCCL_HAS_NVFP16)
#  if _CCCL_HAS_INCLUDE(<cuda_fp16.h>) && (defined(_CCCL_CUDA_COMPILER) || defined(LIBCUDACXX_ENABLE_HOST_NVFP16)) \
                        && !defined(CCCL_DISABLE_FP16_SUPPORT)
#    define _CCCL_HAS_NVFP16 1
#  endif
#endif // !_CCCL_HAS_NVFP16

#if !defined(_CCCL_HAS_NVBF16)
#  if _CCCL_HAS_INCLUDE(<cuda_bf16.h>) && defined(_CCCL_HAS_NVFP16) && !defined(CCCL_DISABLE_BF16_SUPPORT) \
                        && !defined(CUB_DISABLE_BF16_SUPPORT)
#    define _CCCL_HAS_NVBF16 1
#  endif
#endif // !_CCCL_HAS_NVBF16

#endif // __CCCL_EXTENDED_FLOATING_POINT_H
