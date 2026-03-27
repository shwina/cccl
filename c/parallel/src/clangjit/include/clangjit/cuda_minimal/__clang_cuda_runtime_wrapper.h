/*===---- Clangjit preamble + real LLVM wrapper ----------------------------===*/
#ifndef __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#define __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#pragma clang system_header
#if defined(__CUDA__) && defined(__clang__)

// Forward declarations for CUDA intrinsics defined later in the LLVM wrapper
// (line ~294) and sm_20_intrinsics.hpp. Needed because <cuda/std/utility>
// transitively includes address_space.h via memcpy.h → check_address.h,
// and that happens from cuda_runtime.h (line 112) before the definitions.
#define __FWD_DEVICE static __attribute__((device)) __attribute__((always_inline))
__FWD_DEVICE __attribute__((const)) unsigned int __isGlobal(const void *);
__FWD_DEVICE __attribute__((const)) unsigned int __isShared(const void *);
__FWD_DEVICE __attribute__((const)) unsigned int __isConstant(const void *);
__FWD_DEVICE __attribute__((const)) unsigned int __isLocal(const void *);
__FWD_DEVICE unsigned int __isClusterShared(const void *);
__FWD_DEVICE __SIZE_TYPE__ __cvta_generic_to_shared(const void *);
__FWD_DEVICE __SIZE_TYPE__ __cvta_generic_to_global(const void *);
__FWD_DEVICE void * __cvta_shared_to_generic(__SIZE_TYPE__);
__FWD_DEVICE void * __cvta_global_to_generic(__SIZE_TYPE__);
#undef __FWD_DEVICE

// Forward declarations for CUDA 13 fp128 builtins
__attribute__((device)) bool __nv_fp128_isnan(__float128);
__attribute__((device)) __float128 __nv_fp128_fmax(__float128, __float128);
__attribute__((device)) __float128 __nv_fp128_fmin(__float128, __float128);

#define __MATH_FUNCTIONS_HPP__
#include_next <__clang_cuda_runtime_wrapper.h>
#undef __MATH_FUNCTIONS_HPP__

#endif
#endif
