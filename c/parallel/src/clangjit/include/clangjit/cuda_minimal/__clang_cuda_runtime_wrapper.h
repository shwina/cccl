/*===---- Clangjit preamble + real LLVM wrapper ----------------------------===*/
#ifndef __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#define __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#pragma clang system_header
#if defined(__CUDA__) && defined(__clang__)

// Forward declarations needed by libcudacxx before the LLVM wrapper defines them
__attribute__((device)) unsigned int __isClusterShared(const void *);
__attribute__((device)) bool __nv_fp128_isnan(__float128);
__attribute__((device)) __float128 __nv_fp128_fmax(__float128, __float128);
__attribute__((device)) __float128 __nv_fp128_fmin(__float128, __float128);

#define __MATH_FUNCTIONS_HPP__
#include_next <__clang_cuda_runtime_wrapper.h>
#undef __MATH_FUNCTIONS_HPP__

#endif
#endif
