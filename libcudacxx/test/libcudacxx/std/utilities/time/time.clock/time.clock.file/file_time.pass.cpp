//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++17

// <cuda/std/chrono>

// file_time

#include <cuda/std/chrono>

#include "test_macros.h"

template <class Dur>
__host__ __device__ void test()
{
  static_assert(cuda::std::is_same_v<cuda::std::chrono::file_time<Dur>,
                                     cuda::std::chrono::time_point<cuda::std::chrono::file_clock, Dur>>);
}

int main(int, char**)
{
  test<cuda::std::chrono::nanoseconds>();
  test<cuda::std::chrono::minutes>();
  test<cuda::std::chrono::hours>();

  return 0;
}
