/*===---- __clang_cuda_runtime_wrapper.h - CUDA runtime support -------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 *
 * CUDA headers are implemented in a way that currently makes it
 * impossible for user code to #include directly when compiling with
 * Clang. They present different view of CUDA-supplied functions
 * depending on where in NVCC's compilation pipeline the headers are
 * included. Neither of these modes provides function definitions with
 * correct attributes, so we use preprocessor to force the headers
 * into a form that Clang can use.
 *
 * Similarly to NVCC which -include's cuda_runtime.h, Clang -include's
 * this file during every CUDA compilation.
 */

#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__
#define __CLANG_CUDA_RUNTIME_WRAPPER_H__

#if defined(__CUDA__) && defined(__clang__)

// CLANGJIT: Define NVRTC mode only for device compilation to make libcu++
// skip system header includes. For host compilation, we need the full CUDA
// runtime API (cudaLaunchKernel, etc.) which is excluded when __CUDACC_RTC__
// is defined. __CLANGJIT_DEVICE_COMPILATION__ is passed by compiler.cpp.
#ifdef __CLANGJIT_DEVICE_COMPILATION__
#define __CUDACC_RTC__ 1
#endif
#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ 12
#endif
#ifndef __CUDACC_VER_MINOR__
#define __CUDACC_VER_MINOR__ 9
#endif
#ifndef __CUDACC_VER_BUILD__
#define __CUDACC_VER_BUILD__ 0
#endif

// CLANGJIT: CCCL/CUB workarounds for freestanding compilation
// These must be defined before any CCCL headers are included
#define _CCCL_ENABLE_FREESTANDING 1
#define CCCL_DISABLE_FP16_SUPPORT 1
#define CCCL_DISABLE_BF16_SUPPORT 1
#define CCCL_DISABLE_NVTX 1
#define CCCL_DISABLE_EXCEPTIONS 1

// CLANGJIT: Skip clang math forward declares - requires system headers
// #include <__clang_cuda_math_forward_declares.h>

// Define __CUDACC__ early as libstdc++ standard headers with GNU extensions
// enabled depend on it to avoid using __float128, which is unsupported in
// CUDA.
#define __CUDACC__

// CLANGJIT: Skip system headers entirely for JIT compilation.
#undef __CUDACC__

// Preserve common macros that will be changed below by us or by CUDA
// headers.
#pragma push_macro("__THROW")
#pragma push_macro("__CUDA_ARCH__")

// WARNING: Preprocessor hacks below are based on specific details of
// CUDA-7.x headers and are not expected to work with any other
// version of CUDA headers.
// CLANGJIT: Define CUDA_VERSION directly to avoid including cuda.h
// which pulls in system headers. Adjust this for your CUDA version.
#ifndef CUDA_VERSION
#define CUDA_VERSION 12090
#endif

#pragma push_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")
#if CUDA_VERSION >= 10000
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif

// Make largest subset of device functions available during host
// compilation.
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 9999
#endif

#include "__clang_cuda_builtin_vars.h"

// No need for device_launch_parameters.h as __clang_cuda_builtin_vars.h above
// has taken care of builtin variables declared in the file.
#define __DEVICE_LAUNCH_PARAMETERS_H__

// {math,device}_functions.h only have declarations of the
// functions. We don't need them as we're going to pull in their
// definitions from .hpp files.
#define __DEVICE_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__
#define __COMMON_FUNCTIONS_H__
// device_functions_decls is replaced by __clang_cuda_device_functions.h
// included below.
#define __DEVICE_FUNCTIONS_DECLS_H__

#undef __CUDACC__
// CLANGJIT: Ensure basic types are defined before any CUDA headers
typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

#ifndef NULL
#define NULL nullptr
#endif

// CLANGJIT: Stub includes (if needed) should go after CUDA headers define
// vector types

#if CUDA_VERSION < 9000
#define __CUDABE__
#else
#define __CUDACC__
#define __CUDA_LIBDEVICE__
#endif
// Disables definitions of device-side runtime support stubs in
// cuda_device_runtime_api.h
#include "host_defines.h"
#undef __CUDACC__
#include "driver_types.h"
#include "host_config.h"

// Temporarily replace "nv_weak" with weak, so __attribute__((nv_weak)) in
// cuda_device_runtime_api.h ends up being __attribute__((weak)) which is the
// functional equivalent of what we need.
#pragma push_macro("nv_weak")
#define nv_weak weak
#undef __CUDABE__
#undef __CUDA_LIBDEVICE__
#define __CUDACC__
#include "cuda_runtime.h"

#pragma pop_macro("nv_weak")
#undef __CUDACC__
#define __CUDABE__

// CLANGJIT: Address space query intrinsics needed by libcudacxx
// These must be declared before any CCCL headers that use them
// Device implementations use NVVM builtins, host stubs just return false
#ifdef __CUDA_ARCH__
__device__ inline bool __isGlobal(const void* __ptr) {
    return __nvvm_isspacep_global(__ptr);
}
__device__ inline bool __isShared(const void* __ptr) {
    return __nvvm_isspacep_shared(__ptr);
}
__device__ inline bool __isConstant(const void* __ptr) {
    return __nvvm_isspacep_const(__ptr);
}
__device__ inline bool __isLocal(const void* __ptr) {
    return __nvvm_isspacep_local(__ptr);
}
__device__ inline bool __isClusterShared(const void* __ptr) {
    (void)__ptr;
    return false;
}
__device__ inline unsigned __cvta_generic_to_shared(const void* __ptr) {
    unsigned __result;
    asm("cvta.to.shared.u32 %0, %1;" : "=r"(__result) : "l"(__ptr));
    return __result;
}
__device__ inline unsigned long long __cvta_generic_to_global(const void* __ptr) {
    unsigned long long __result;
    asm("cvta.to.global.u64 %0, %1;" : "=l"(__result) : "l"(__ptr));
    return __result;
}
__device__ inline void* __cvta_shared_to_generic(unsigned __ptr) {
    void* __result;
    asm("cvta.shared.u64 %0, %1;" : "=l"(__result) : "r"(__ptr));
    return __result;
}
__device__ inline void* __cvta_global_to_generic(unsigned long long __ptr) {
    void* __result;
    asm("cvta.global.u64 %0, %1;" : "=l"(__result) : "l"(__ptr));
    return __result;
}
#else
// Host stubs - these are never called but satisfy the parser
inline bool __isGlobal(const void*) { return false; }
inline bool __isShared(const void*) { return false; }
inline bool __isConstant(const void*) { return false; }
inline bool __isLocal(const void*) { return false; }
inline bool __isClusterShared(const void*) { return false; }
inline unsigned __cvta_generic_to_shared(const void*) { return 0; }
inline unsigned long long __cvta_generic_to_global(const void*) { return 0; }
inline void* __cvta_shared_to_generic(unsigned) { return nullptr; }
inline void* __cvta_global_to_generic(unsigned long long) { return nullptr; }
#endif

// CUDA headers use __nvvm_memcpy and __nvvm_memset which Clang does
// not have at the moment. Emulate them with a builtin memcpy/memset.
#define __nvvm_memcpy(s, d, n, a) __builtin_memcpy(s, d, n)
#define __nvvm_memset(d, c, n, a) __builtin_memset(d, c, n)

#if CUDA_VERSION < 9000
#include "crt/device_runtime.h"
#endif
#include "crt/host_runtime.h"
// device_runtime.h defines __cxa_* macros that will conflict with
// cxxabi.h.
// FIXME: redefine these as __device__ functions.
#undef __cxa_vec_ctor
#undef __cxa_vec_cctor
#undef __cxa_vec_dtor
#undef __cxa_vec_new
#undef __cxa_vec_new2
#undef __cxa_vec_new3
#undef __cxa_vec_delete2
#undef __cxa_vec_delete
#undef __cxa_vec_delete3
#undef __cxa_pure_virtual

// math_functions.hpp expects this host function be defined on MacOS, but it
// ends up not being there because of the games we play here.  Just define it
// ourselves; it's simple enough.
#ifdef __APPLE__
inline __host__ double __signbitd(double x) { return std::signbit(x); }
#endif

// CUDA 9.1 no longer provides declarations for libdevice functions, so we need
// to provide our own.
#include <__clang_cuda_libdevice_declares.h>

// Wrappers for many device-side standard library functions, incl. math
// functions, became compiler builtins in CUDA-9 and have been removed from the
// CUDA headers. Clang now provides its own implementation of the wrappers.
#if CUDA_VERSION >= 9000
// CLANGJIT: Include clang's device functions for all the CUDA intrinsics
// (atomics, shuffles, bit operations, math functions, etc.)
#include <__clang_cuda_device_functions.h>
// CLANGJIT: High-level atomic wrappers (atomicAdd, atomicCAS, etc.)
#include <cuda_device_atomic_functions.h>

// CLANGJIT: Additional intrinsics for CUB/CCCL freestanding mode
// These supplement clang's device functions with overloads CUB expects
#ifdef __CLANGJIT_DEVICE_COMPILATION__

// Warp synchronization
__device__ inline void __syncwarp(unsigned __mask = 0xFFFFFFFF) {
    __nvvm_bar_warp_sync(__mask);
}

__device__ inline unsigned __activemask() {
    unsigned mask;
    asm volatile("activemask.b32 %0;" : "=r"(mask));
    return mask;
}

// Voting intrinsics
__device__ inline unsigned __ballot_sync(unsigned __mask, int __pred) {
    return __nvvm_vote_ballot_sync(__mask, __pred);
}

__device__ inline int __all_sync(unsigned __mask, int __pred) {
    return __nvvm_vote_all_sync(__mask, __pred);
}

__device__ inline int __any_sync(unsigned __mask, int __pred) {
    return __nvvm_vote_any_sync(__mask, __pred);
}

// Warp shuffle intrinsics - CUB uses these exact signatures
__device__ inline int __shfl_sync(unsigned __mask, int __val, int __src, int __width = 32) {
    return __nvvm_shfl_sync_idx_i32(__mask, __val, __src, ((32 - __width) << 8) | 0x1f);
}
__device__ inline unsigned int __shfl_sync(unsigned __mask, unsigned int __val, int __src, int __width = 32) {
    return static_cast<unsigned int>(__nvvm_shfl_sync_idx_i32(__mask, static_cast<int>(__val), __src, ((32 - __width) << 8) | 0x1f));
}
__device__ inline long __shfl_sync(unsigned __mask, long __val, int __src, int __width = 32) {
    return static_cast<long>(__nvvm_shfl_sync_idx_i32(__mask, static_cast<int>(__val), __src, ((32 - __width) << 8) | 0x1f));
}
__device__ inline unsigned long __shfl_sync(unsigned __mask, unsigned long __val, int __src, int __width = 32) {
    return static_cast<unsigned long>(__nvvm_shfl_sync_idx_i32(__mask, static_cast<int>(__val), __src, ((32 - __width) << 8) | 0x1f));
}
__device__ inline long long __shfl_sync(unsigned __mask, long long __val, int __src, int __width = 32) {
    int lo = static_cast<int>(__val);
    int hi = static_cast<int>(__val >> 32);
    lo = __nvvm_shfl_sync_idx_i32(__mask, lo, __src, ((32 - __width) << 8) | 0x1f);
    hi = __nvvm_shfl_sync_idx_i32(__mask, hi, __src, ((32 - __width) << 8) | 0x1f);
    return (static_cast<long long>(hi) << 32) | static_cast<unsigned int>(lo);
}
__device__ inline unsigned long long __shfl_sync(unsigned __mask, unsigned long long __val, int __src, int __width = 32) {
    return static_cast<unsigned long long>(__shfl_sync(__mask, static_cast<long long>(__val), __src, __width));
}
__device__ inline float __shfl_sync(unsigned __mask, float __val, int __src, int __width = 32) {
    return __nvvm_shfl_sync_idx_f32(__mask, __val, __src, ((32 - __width) << 8) | 0x1f);
}
__device__ inline double __shfl_sync(unsigned __mask, double __val, int __src, int __width = 32) {
    long long tmp = __double_as_longlong(__val);
    tmp = __shfl_sync(__mask, tmp, __src, __width);
    return __longlong_as_double(tmp);
}

__device__ inline int __shfl_xor_sync(unsigned __mask, int __val, int __lane, int __width = 32) {
    return __nvvm_shfl_sync_bfly_i32(__mask, __val, __lane, ((32 - __width) << 8) | 0x1f);
}
__device__ inline unsigned int __shfl_xor_sync(unsigned __mask, unsigned int __val, int __lane, int __width = 32) {
    return static_cast<unsigned int>(__nvvm_shfl_sync_bfly_i32(__mask, static_cast<int>(__val), __lane, ((32 - __width) << 8) | 0x1f));
}
__device__ inline float __shfl_xor_sync(unsigned __mask, float __val, int __lane, int __width = 32) {
    return __nvvm_shfl_sync_bfly_f32(__mask, __val, __lane, ((32 - __width) << 8) | 0x1f);
}

__device__ inline int __shfl_up_sync(unsigned __mask, int __val, unsigned __delta, int __width = 32) {
    return __nvvm_shfl_sync_up_i32(__mask, __val, __delta, (32 - __width) << 8);
}
__device__ inline unsigned int __shfl_up_sync(unsigned __mask, unsigned int __val, unsigned __delta, int __width = 32) {
    return static_cast<unsigned int>(__nvvm_shfl_sync_up_i32(__mask, static_cast<int>(__val), __delta, (32 - __width) << 8));
}
__device__ inline float __shfl_up_sync(unsigned __mask, float __val, unsigned __delta, int __width = 32) {
    return __nvvm_shfl_sync_up_f32(__mask, __val, __delta, (32 - __width) << 8);
}

__device__ inline int __shfl_down_sync(unsigned __mask, int __val, unsigned __delta, int __width = 32) {
    return __nvvm_shfl_sync_down_i32(__mask, __val, __delta, ((32 - __width) << 8) | 0x1f);
}
__device__ inline unsigned int __shfl_down_sync(unsigned __mask, unsigned int __val, unsigned __delta, int __width = 32) {
    return static_cast<unsigned int>(__nvvm_shfl_sync_down_i32(__mask, static_cast<int>(__val), __delta, ((32 - __width) << 8) | 0x1f));
}
__device__ inline float __shfl_down_sync(unsigned __mask, float __val, unsigned __delta, int __width = 32) {
    return __nvvm_shfl_sync_down_f32(__mask, __val, __delta, ((32 - __width) << 8) | 0x1f);
}

// Warp-level reduction intrinsics (sm_80+)
__device__ inline int __reduce_add_sync(unsigned __mask, int __val) {
    return __nvvm_redux_sync_add(__val, __mask);
}
__device__ inline unsigned __reduce_add_sync(unsigned __mask, unsigned __val) {
    return static_cast<unsigned>(__nvvm_redux_sync_add(static_cast<int>(__val), __mask));
}
__device__ inline int __reduce_min_sync(unsigned __mask, int __val) {
    return __nvvm_redux_sync_min(__val, __mask);
}
__device__ inline unsigned __reduce_min_sync(unsigned __mask, unsigned __val) {
    return __nvvm_redux_sync_umin(__val, __mask);
}
__device__ inline int __reduce_max_sync(unsigned __mask, int __val) {
    return __nvvm_redux_sync_max(__val, __mask);
}
__device__ inline unsigned __reduce_max_sync(unsigned __mask, unsigned __val) {
    return __nvvm_redux_sync_umax(__val, __mask);
}
__device__ inline unsigned __reduce_and_sync(unsigned __mask, unsigned __val) {
    return __nvvm_redux_sync_and(static_cast<int>(__val), __mask);
}
__device__ inline unsigned __reduce_or_sync(unsigned __mask, unsigned __val) {
    return __nvvm_redux_sync_or(static_cast<int>(__val), __mask);
}
__device__ inline unsigned __reduce_xor_sync(unsigned __mask, unsigned __val) {
    return __nvvm_redux_sync_xor(static_cast<int>(__val), __mask);
}

// Match intrinsics (sm_70+)
__device__ inline unsigned __match_any_sync(unsigned __mask, int __val) {
    unsigned result;
    asm volatile("match.any.sync.b32 %0, %1, %2;" : "=r"(result) : "r"(__val), "r"(__mask));
    return result;
}
__device__ inline unsigned __match_any_sync(unsigned __mask, unsigned __val) {
    unsigned result;
    asm volatile("match.any.sync.b32 %0, %1, %2;" : "=r"(result) : "r"(__val), "r"(__mask));
    return result;
}
__device__ inline unsigned __match_any_sync(unsigned __mask, long long __val) {
    unsigned result;
    asm volatile("match.any.sync.b64 %0, %1, %2;" : "=r"(result) : "l"(__val), "r"(__mask));
    return result;
}
__device__ inline unsigned __match_any_sync(unsigned __mask, unsigned long long __val) {
    unsigned result;
    asm volatile("match.any.sync.b64 %0, %1, %2;" : "=r"(result) : "l"(__val), "r"(__mask));
    return result;
}
__device__ inline unsigned __match_all_sync(unsigned __mask, int __val, int* __pred) {
    unsigned result;
    int p;
    asm volatile("match.all.sync.b32 %0|%1, %2, %3;" : "=r"(result), "=r"(p) : "r"(__val), "r"(__mask));
    *__pred = p;
    return result;
}
__device__ inline unsigned __match_all_sync(unsigned __mask, unsigned __val, int* __pred) {
    unsigned result;
    int p;
    asm volatile("match.all.sync.b32 %0|%1, %2, %3;" : "=r"(result), "=r"(p) : "r"(__val), "r"(__mask));
    *__pred = p;
    return result;
}

// Nanosleep intrinsic (sm_70+)
__device__ inline void __nanosleep(unsigned __ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(__ns));
}

// PDL (Programmatic Dependent Launch) intrinsics (sm_90+)
__device__ inline void cudaTriggerProgrammaticLaunchCompletion() {
    asm volatile("griddepcontrol.launch_dependents;");
}
__device__ inline void cudaGridDependencySynchronize() {
    asm volatile("griddepcontrol.wait;");
}

#else // !__CLANGJIT_DEVICE_COMPILATION__
// Host stubs for warp intrinsics - satisfy the parser during host compilation
// Must be __host__ __device__ because device code templates are still parsed

// Warp synchronization
__host__ __device__ inline void __syncwarp(unsigned = 0xFFFFFFFF) {}
__host__ __device__ inline unsigned __activemask() { return 0; }

// Voting intrinsics
__host__ __device__ inline unsigned __ballot_sync(unsigned, int) { return 0; }
__host__ __device__ inline int __all_sync(unsigned, int) { return 0; }
__host__ __device__ inline int __any_sync(unsigned, int) { return 0; }

// Warp shuffle intrinsics
__host__ __device__ inline int __shfl_sync(unsigned, int __val, int, int = 32) { return __val; }
__host__ __device__ inline unsigned int __shfl_sync(unsigned, unsigned int __val, int, int = 32) { return __val; }
__host__ __device__ inline long __shfl_sync(unsigned, long __val, int, int = 32) { return __val; }
__host__ __device__ inline unsigned long __shfl_sync(unsigned, unsigned long __val, int, int = 32) { return __val; }
__host__ __device__ inline long long __shfl_sync(unsigned, long long __val, int, int = 32) { return __val; }
__host__ __device__ inline unsigned long long __shfl_sync(unsigned, unsigned long long __val, int, int = 32) { return __val; }
__host__ __device__ inline float __shfl_sync(unsigned, float __val, int, int = 32) { return __val; }
__host__ __device__ inline double __shfl_sync(unsigned, double __val, int, int = 32) { return __val; }

__host__ __device__ inline int __shfl_xor_sync(unsigned, int __val, int, int = 32) { return __val; }
__host__ __device__ inline unsigned int __shfl_xor_sync(unsigned, unsigned int __val, int, int = 32) { return __val; }
__host__ __device__ inline float __shfl_xor_sync(unsigned, float __val, int, int = 32) { return __val; }

__host__ __device__ inline int __shfl_up_sync(unsigned, int __val, unsigned, int = 32) { return __val; }
__host__ __device__ inline unsigned int __shfl_up_sync(unsigned, unsigned int __val, unsigned, int = 32) { return __val; }
__host__ __device__ inline float __shfl_up_sync(unsigned, float __val, unsigned, int = 32) { return __val; }

__host__ __device__ inline int __shfl_down_sync(unsigned, int __val, unsigned, int = 32) { return __val; }
__host__ __device__ inline unsigned int __shfl_down_sync(unsigned, unsigned int __val, unsigned, int = 32) { return __val; }
__host__ __device__ inline float __shfl_down_sync(unsigned, float __val, unsigned, int = 32) { return __val; }

// Warp-level reduction intrinsics
__host__ __device__ inline int __reduce_add_sync(unsigned, int __val) { return __val; }
__host__ __device__ inline unsigned __reduce_add_sync(unsigned, unsigned __val) { return __val; }
__host__ __device__ inline int __reduce_min_sync(unsigned, int __val) { return __val; }
__host__ __device__ inline unsigned __reduce_min_sync(unsigned, unsigned __val) { return __val; }
__host__ __device__ inline int __reduce_max_sync(unsigned, int __val) { return __val; }
__host__ __device__ inline unsigned __reduce_max_sync(unsigned, unsigned __val) { return __val; }
__host__ __device__ inline unsigned __reduce_and_sync(unsigned, unsigned __val) { return __val; }
__host__ __device__ inline unsigned __reduce_or_sync(unsigned, unsigned __val) { return __val; }
__host__ __device__ inline unsigned __reduce_xor_sync(unsigned, unsigned __val) { return __val; }

// Match intrinsics
__host__ __device__ inline unsigned __match_any_sync(unsigned, int) { return 0; }
__host__ __device__ inline unsigned __match_any_sync(unsigned, unsigned) { return 0; }
__host__ __device__ inline unsigned __match_any_sync(unsigned, long long) { return 0; }
__host__ __device__ inline unsigned __match_any_sync(unsigned, unsigned long long) { return 0; }
__host__ __device__ inline unsigned __match_all_sync(unsigned, int, int*) { return 0; }
__host__ __device__ inline unsigned __match_all_sync(unsigned, unsigned, int*) { return 0; }
__host__ __device__ inline unsigned __match_all_sync(unsigned, long long, int*) { return 0; }
__host__ __device__ inline unsigned __match_all_sync(unsigned, unsigned long long, int*) { return 0; }

// Nanosleep intrinsic
__host__ __device__ inline void __nanosleep(unsigned) {}

// Note: PDL intrinsics (cudaTriggerProgrammaticLaunchCompletion, cudaGridDependencySynchronize)
// are provided by cuda_device_runtime_api.h, so we don't need host stubs

#endif // __CLANGJIT_DEVICE_COMPILATION__ (additional intrinsics)
#endif

// __THROW is redefined to be empty by device_functions_decls.h in CUDA. Clang's
// counterpart does not do it, so we need to make it empty here to keep
// following CUDA includes happy.
#undef __THROW
#define __THROW

// CUDA 8.0.41 relies on __USE_FAST_MATH__ and __CUDA_PREC_DIV's values.
// Previous versions used to check whether they are defined or not.
// CU_DEVICE_INVALID macro is only defined in 8.0.41, so we use it
// here to detect the switch.

#if defined(CU_DEVICE_INVALID)
#if !defined(__USE_FAST_MATH__)
#define __USE_FAST_MATH__ 0
#endif

#if !defined(__CUDA_PREC_DIV)
#define __CUDA_PREC_DIV 0
#endif
#endif

// CLANGJIT: Skipping the __host__ poison and device_functions.hpp includes
// since we're not including the headers that would need checking.
// The original wrapper used this to catch incorrect __host__ usage in
// device_functions.hpp and math_functions.hpp, which we've skipped.

// device_functions.hpp and math_functions*.hpp use 'static
// __forceinline__' (with no __device__) for definitions of device
// functions. Temporarily redefine __forceinline__ to include
// __device__.
#pragma push_macro("__forceinline__")
#define __forceinline__ __device__ __inline__ __attribute__((always_inline))
// CLANGJIT: Skip device_functions.hpp
// #if CUDA_VERSION < 9000
// #include "device_functions.hpp"
// #endif

// math_function.hpp uses the __USE_FAST_MATH__ macro to determine whether we
// get the slow-but-accurate or fast-but-inaccurate versions of functions like
// sin and exp.  This is controlled in clang by -fgpu-approx-transcendentals.
//
// device_functions.hpp uses __USE_FAST_MATH__ for a different purpose (fast vs.
// slow divides), so we need to scope our define carefully here.
#pragma push_macro("__USE_FAST_MATH__")
#if defined(__CLANG_GPU_APPROX_TRANSCENDENTALS__)
#define __USE_FAST_MATH__ 1
#endif

// CLANGJIT: Skip NVIDIA math_functions.hpp - requires system math functions
// #if CUDA_VERSION >= 9000
// #include "crt/math_functions.hpp"
// #else
// #include "math_functions.hpp"
// #endif

#pragma pop_macro("__USE_FAST_MATH__")

// #if CUDA_VERSION < 9000
// #include "math_functions_dbl_ptx3.hpp"
// #endif
#pragma pop_macro("__forceinline__")

// CLANGJIT: Skip host-only math functions
// Pull in host-only functions that are only available when neither
// __CUDACC__ nor __CUDABE__ are defined.
#undef __MATH_FUNCTIONS_HPP__
#undef __CUDABE__
// #if CUDA_VERSION < 9000
// #include "math_functions.hpp"
// #endif

#if CUDA_VERSION < 9000
// For some reason single-argument variant is not always declared by
// CUDA headers. Alas, device_functions.hpp included below needs it.
static inline __device__ void __brkpt(int __c) { __brkpt(); }
#endif

// CLANGJIT: Skipping the device/math function .hpp includes that would
// need __host__ to be defined as empty. Since we're not including those
// files, we don't need to manipulate __host__ here.
#undef __CUDABE__
#define __CUDACC__
// CLANGJIT: Skip device functions and atomics hpp files - require system
// headers #if CUDA_VERSION >= 9000 #include "device_atomic_functions.h" #endif
// #undef __DEVICE_FUNCTIONS_HPP__
// #include "device_atomic_functions.hpp"
// #if CUDA_VERSION >= 9000
// #include "crt/device_functions.hpp"
// #include "crt/device_double_functions.hpp"
// #else
// #include "device_functions.hpp"
// #define __CUDABE__
// #include "device_double_functions.h"
// #undef __CUDABE__
// #endif
// #include "sm_20_atomic_functions.hpp"
// CLANGJIT: Skip sm_* intrinsics and atomic headers - require system headers
// All skipped: sm_20_intrinsics.hpp, sm_32_atomic_functions.hpp,
//              sm_60_atomic_functions.hpp, sm_61_intrinsics.hpp

#undef __MATH_FUNCTIONS_HPP__

// math_functions.hpp defines ::signbit as a __host__ __device__ function.  This
// conflicts with libstdc++'s constexpr ::signbit, so we have to rename
// math_function.hpp's ::signbit.  It's guarded by #undef signbit, but that's
// conditional on __GNUC__.  :)
#pragma push_macro("signbit")
#pragma push_macro("__GNUC__")
#undef __GNUC__
#define signbit __ignored_cuda_signbit

// CUDA-9 omits device-side definitions of some math functions if it sees
// include guard from math.h wrapper from libstdc++. We have to undo the header
// guard temporarily to get the definitions we need.
#pragma push_macro("_GLIBCXX_MATH_H")
#pragma push_macro("_LIBCPP_VERSION")
#if CUDA_VERSION >= 9000
#undef _GLIBCXX_MATH_H
// We also need to undo another guard that checks for libc++ 3.8+
#ifdef _LIBCPP_VERSION
#define _LIBCPP_VERSION 3700
#endif
#endif

// CLANGJIT: Skip math_functions.hpp
// #if CUDA_VERSION >= 9000
// #include "crt/math_functions.hpp"
// #else
// #include "math_functions.hpp"
// #endif
#pragma pop_macro("_GLIBCXX_MATH_H")
#pragma pop_macro("_LIBCPP_VERSION")
#pragma pop_macro("__GNUC__")
#pragma pop_macro("signbit")

// CLANGJIT: Removed orphaned pop_macro("__host__") - we're not poisoning it

// __clang_cuda_texture_intrinsics.h must be included first in order to provide
// implementation for __nv_tex_surf_handler that CUDA's headers depend on.
// The implementation requires c++11 and only works with CUDA-9 or newer.
#if __cplusplus >= 201103L && CUDA_VERSION >= 9000
// clang-format off
#include <__clang_cuda_texture_intrinsics.h>
// clang-format on
#else
#if CUDA_VERSION >= 9000
// Provide a hint that texture support needs C++11.
template <typename T> struct __nv_tex_needs_cxx11 {
  const static bool value = false;
};
template <class T>
__host__ __device__ void __nv_tex_surf_handler(const char *name, T *ptr,
                                               cudaTextureObject_t obj,
                                               float x) {
  _Static_assert(__nv_tex_needs_cxx11<T>::value,
                 "Texture support requires C++11");
}
#else
// Textures in CUDA-8 and older are not supported by clang.There's no
// convenient way to intercept texture use in these versions, so we can't
// produce a meaningful error. The source code that attempts to use textures
// will continue to fail as it does now.
#endif // CUDA_VERSION
#endif // __cplusplus >= 201103L && CUDA_VERSION >= 9000
// CLANGJIT: Skip texture headers - not needed for basic kernels
// #include "texture_fetch_functions.h"
// #include "texture_indirect_functions.h"

// Restore state of __CUDA_ARCH__ and __THROW we had on entry.
#pragma pop_macro("__CUDA_ARCH__")
#pragma pop_macro("__THROW")

// Set up compiler macros expected to be seen during compilation.
#undef __CUDABE__
#define __CUDACC__

extern "C" {
// Device-side CUDA system calls.
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
// We need these declarations and wrappers for device-side
// malloc/free/printf calls to work without relying on
// -fcuda-disable-target-call-checks option.
__device__ int vprintf(const char *, const char *);
__device__ void free(void *) __attribute((nothrow));
__device__ void *malloc(size_t) __attribute((nothrow)) __attribute__((malloc));

// __assertfail() used to have a `noreturn` attribute. Unfortunately that
// contributed to triggering the longstanding bug in ptxas when assert was used
// in sufficiently convoluted code. See
// https://bugs.llvm.org/show_bug.cgi?id=27738 for the details.
__device__ void __assertfail(const char *__message, const char *__file,
                             unsigned __line, const char *__function,
                             size_t __charSize);

// In order for standard assert() macro on linux to work we need to
// provide device-side __assert_fail()
__device__ static inline void __assert_fail(const char *__message,
                                            const char *__file, unsigned __line,
                                            const char *__function) {
  __assertfail(__message, __file, __line, __function, sizeof(char));
}

// Clang will convert printf into vprintf, but we still need
// device-side declaration for it.
__device__ int printf(const char *, ...);
} // extern "C"

// We also need device-side std::malloc and std::free.
namespace std {
__device__ static inline void free(void *__ptr) { ::free(__ptr); }
__device__ static inline void *malloc(size_t __size) {
  return ::malloc(__size);
}
} // namespace std

// Out-of-line implementations from __clang_cuda_builtin_vars.h.  These need to
// come after we've pulled in the definition of uint3 and dim3.

__device__ inline __cuda_builtin_threadIdx_t::operator dim3() const {
  return dim3(x, y, z);
}

__device__ inline __cuda_builtin_threadIdx_t::operator uint3() const {
  return {x, y, z};
}

__device__ inline __cuda_builtin_blockIdx_t::operator dim3() const {
  return dim3(x, y, z);
}

__device__ inline __cuda_builtin_blockIdx_t::operator uint3() const {
  return {x, y, z};
}

__device__ inline __cuda_builtin_blockDim_t::operator dim3() const {
  return dim3(x, y, z);
}

__device__ inline __cuda_builtin_blockDim_t::operator uint3() const {
  return {x, y, z};
}

__device__ inline __cuda_builtin_gridDim_t::operator dim3() const {
  return dim3(x, y, z);
}

__device__ inline __cuda_builtin_gridDim_t::operator uint3() const {
  return {x, y, z};
}

// CLANGJIT: Skip clang cmath/intrinsics/complex headers - require system
// headers #include <__clang_cuda_cmath.h> #include <__clang_cuda_intrinsics.h>
// #include <__clang_cuda_complex_builtins.h>

// CLANGJIT: Skip curand header - requires system headers
// curand_mtgp32_kernel helpfully redeclares blockDim and threadIdx in host
// mode, giving them their "proper" types of dim3 and uint3.  This is
// incompatible with the types we give in __clang_cuda_builtin_vars.h.
// #pragma push_macro("dim3")
// #pragma push_macro("uint3")
// #define dim3 __cuda_builtin_blockDim_t
// #define uint3 __cuda_builtin_threadIdx_t
// #include "curand_mtgp32_kernel.h"
// #pragma pop_macro("dim3")
// #pragma pop_macro("uint3")
// CLANGJIT: Removed orphaned pop_macro - the corresponding push/include was
// skipped #pragma pop_macro("__USE_FAST_MATH__")
#pragma pop_macro("__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__")

// CUDA runtime uses this undocumented function to access kernel launch
// configuration. The declaration is in crt/device_functions.h but that file
// includes a lot of other stuff we don't want. Instead, we'll provide our own
// declaration for it here.
#if CUDA_VERSION >= 9020
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem = 0,
                                                void *stream = 0);
#endif

// CLANGJIT: Occupancy API is provided by cuda_runtime.h which we include above.

// CLANGJIT: CUDA runtime API declarations for device compilation
// In NVRTC mode, these may not be declared but the device compiler still parses host code.
// These declarations satisfy the parser; actual implementations come from libcudart at link time.
#ifdef __CLANGJIT_DEVICE_COMPILATION__
// Template version of cudaMalloc to accept any pointer type
template <typename T>
__host__ cudaError_t cudaMalloc(T** devPtr, size_t size);
__host__ cudaError_t cudaFree(void* devPtr);
__host__ cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
__host__ cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
__host__ cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = 0);
__host__ cudaError_t cudaDeviceSynchronize(void);
__host__ cudaError_t cudaGetLastError(void);
__host__ cudaError_t cudaPeekAtLastError(void);
__host__ const char* cudaGetErrorString(cudaError_t error);
__host__ cudaError_t cudaStreamSynchronize(cudaStream_t stream);
__host__ cudaError_t cudaGetDevice(int* device);
__host__ cudaError_t cudaSetDevice(int device);
__host__ cudaError_t cudaGetDeviceCount(int* count);
__host__ cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
__host__ cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
__host__ cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);
__host__ cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int  value);
template <class T>
__host__ cudaError_t cudaLaunchKernel(const T *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
template <class T>
__host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize);
#endif

#endif // __CUDA__
#endif // __CLANG_CUDA_RUNTIME_WRAPPER_H__
