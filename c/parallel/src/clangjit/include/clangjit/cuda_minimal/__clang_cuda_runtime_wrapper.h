/*===---- Clangjit preamble + real LLVM wrapper ----------------------------===*/
#ifndef __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#define __CLANGJIT_CUDA_RUNTIME_PREAMBLE_H__
#pragma clang system_header
#if defined(__CUDA__) && defined(__clang__)
#define __MATH_FUNCTIONS_HPP__
#include_next <__clang_cuda_runtime_wrapper.h>
#undef __MATH_FUNCTIONS_HPP__
#endif
#endif
