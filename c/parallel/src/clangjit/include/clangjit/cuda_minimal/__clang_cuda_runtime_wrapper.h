/*===---- Clangjit preamble + real LLVM wrapper ----------------------------===*/
#ifndef __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#define __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#if defined(__CUDA__) && defined(__clang__)

#define _CCCL_ENABLE_FREESTANDING 1
#define CCCL_DISABLE_FP16_SUPPORT 1
#define CCCL_DISABLE_BF16_SUPPORT 1
#define CCCL_DISABLE_NVTX 1
#define CCCL_DISABLE_EXCEPTIONS 1
// Skip CUDA toolkit's math_functions.hpp host implementations — they need
// system math (sqrt, etc.) which we don't provide. Device math comes from
// __clang_cuda_math.h and libdevice.
#define __MATH_FUNCTIONS_HPP__
#include_next <__clang_cuda_runtime_wrapper.h>
#undef __MATH_FUNCTIONS_HPP__

#endif
#endif
