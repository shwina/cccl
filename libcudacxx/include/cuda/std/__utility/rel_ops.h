//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_REL_OPS_H
#define _LIBCUDACXX___UTILITY_REL_OPS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace rel_ops
{

template <class _Tp>
_CCCL_API inline bool operator!=(const _Tp& __x, const _Tp& __y)
{
  return !(__x == __y);
}

template <class _Tp>
_CCCL_API inline bool operator>(const _Tp& __x, const _Tp& __y)
{
  return __y < __x;
}

template <class _Tp>
_CCCL_API inline bool operator<=(const _Tp& __x, const _Tp& __y)
{
  return !(__y < __x);
}

template <class _Tp>
_CCCL_API inline bool operator>=(const _Tp& __x, const _Tp& __y)
{
  return !(__x < __y);
}

} // namespace rel_ops

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_REL_OPS_H
