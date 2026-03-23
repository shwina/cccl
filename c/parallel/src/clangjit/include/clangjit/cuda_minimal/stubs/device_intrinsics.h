// Minimal CUDA device intrinsic wrappers for JIT compilation
// These map CUDA intrinsic names to libdevice function names
#ifndef _CLANGJIT_DEVICE_INTRINSICS_H
#define _CLANGJIT_DEVICE_INTRINSICS_H

// Ensure we have the libdevice declarations
#include <__clang_cuda_libdevice_declares.h>

#define __DEVICE__ static __device__ __forceinline__

// Float FMA intrinsics
__DEVICE__ float __fmaf_rn(float a, float b, float c) { return __nv_fmaf_rn(a, b, c); }
__DEVICE__ float __fmaf_rz(float a, float b, float c) { return __nv_fmaf_rz(a, b, c); }
__DEVICE__ float __fmaf_rd(float a, float b, float c) { return __nv_fmaf_rd(a, b, c); }
__DEVICE__ float __fmaf_ru(float a, float b, float c) { return __nv_fmaf_ru(a, b, c); }

// Double FMA intrinsics
__DEVICE__ double __fma_rn(double a, double b, double c) { return __nv_fma_rn(a, b, c); }
__DEVICE__ double __fma_rz(double a, double b, double c) { return __nv_fma_rz(a, b, c); }
__DEVICE__ double __fma_rd(double a, double b, double c) { return __nv_fma_rd(a, b, c); }
__DEVICE__ double __fma_ru(double a, double b, double c) { return __nv_fma_ru(a, b, c); }

// Float arithmetic with rounding modes
__DEVICE__ float __fadd_rn(float a, float b) { return __nv_fadd_rn(a, b); }
__DEVICE__ float __fadd_rz(float a, float b) { return __nv_fadd_rz(a, b); }
__DEVICE__ float __fadd_rd(float a, float b) { return __nv_fadd_rd(a, b); }
__DEVICE__ float __fadd_ru(float a, float b) { return __nv_fadd_ru(a, b); }

__DEVICE__ float __fsub_rn(float a, float b) { return __nv_fsub_rn(a, b); }
__DEVICE__ float __fsub_rz(float a, float b) { return __nv_fsub_rz(a, b); }
__DEVICE__ float __fsub_rd(float a, float b) { return __nv_fsub_rd(a, b); }
__DEVICE__ float __fsub_ru(float a, float b) { return __nv_fsub_ru(a, b); }

__DEVICE__ float __fmul_rn(float a, float b) { return __nv_fmul_rn(a, b); }
__DEVICE__ float __fmul_rz(float a, float b) { return __nv_fmul_rz(a, b); }
__DEVICE__ float __fmul_rd(float a, float b) { return __nv_fmul_rd(a, b); }
__DEVICE__ float __fmul_ru(float a, float b) { return __nv_fmul_ru(a, b); }

__DEVICE__ float __fdiv_rn(float a, float b) { return __nv_fdiv_rn(a, b); }
__DEVICE__ float __fdiv_rz(float a, float b) { return __nv_fdiv_rz(a, b); }
__DEVICE__ float __fdiv_rd(float a, float b) { return __nv_fdiv_rd(a, b); }
__DEVICE__ float __fdiv_ru(float a, float b) { return __nv_fdiv_ru(a, b); }

__DEVICE__ float __frcp_rn(float a) { return __nv_frcp_rn(a); }
__DEVICE__ float __frcp_rz(float a) { return __nv_frcp_rz(a); }
__DEVICE__ float __frcp_rd(float a) { return __nv_frcp_rd(a); }
__DEVICE__ float __frcp_ru(float a) { return __nv_frcp_ru(a); }

__DEVICE__ float __fsqrt_rn(float a) { return __nv_fsqrt_rn(a); }
__DEVICE__ float __fsqrt_rz(float a) { return __nv_fsqrt_rz(a); }
__DEVICE__ float __fsqrt_rd(float a) { return __nv_fsqrt_rd(a); }
__DEVICE__ float __fsqrt_ru(float a) { return __nv_fsqrt_ru(a); }

// Double arithmetic with rounding modes
__DEVICE__ double __dadd_rn(double a, double b) { return __nv_dadd_rn(a, b); }
__DEVICE__ double __dadd_rz(double a, double b) { return __nv_dadd_rz(a, b); }
__DEVICE__ double __dadd_rd(double a, double b) { return __nv_dadd_rd(a, b); }
__DEVICE__ double __dadd_ru(double a, double b) { return __nv_dadd_ru(a, b); }

__DEVICE__ double __dsub_rn(double a, double b) { return __nv_dsub_rn(a, b); }
__DEVICE__ double __dsub_rz(double a, double b) { return __nv_dsub_rz(a, b); }
__DEVICE__ double __dsub_rd(double a, double b) { return __nv_dsub_rd(a, b); }
__DEVICE__ double __dsub_ru(double a, double b) { return __nv_dsub_ru(a, b); }

__DEVICE__ double __dmul_rn(double a, double b) { return __nv_dmul_rn(a, b); }
__DEVICE__ double __dmul_rz(double a, double b) { return __nv_dmul_rz(a, b); }
__DEVICE__ double __dmul_rd(double a, double b) { return __nv_dmul_rd(a, b); }
__DEVICE__ double __dmul_ru(double a, double b) { return __nv_dmul_ru(a, b); }

__DEVICE__ double __ddiv_rn(double a, double b) { return __nv_ddiv_rn(a, b); }
__DEVICE__ double __ddiv_rz(double a, double b) { return __nv_ddiv_rz(a, b); }
__DEVICE__ double __ddiv_rd(double a, double b) { return __nv_ddiv_rd(a, b); }
__DEVICE__ double __ddiv_ru(double a, double b) { return __nv_ddiv_ru(a, b); }

__DEVICE__ double __drcp_rn(double a) { return __nv_drcp_rn(a); }
__DEVICE__ double __drcp_rz(double a) { return __nv_drcp_rz(a); }
__DEVICE__ double __drcp_rd(double a) { return __nv_drcp_rd(a); }
__DEVICE__ double __drcp_ru(double a) { return __nv_drcp_ru(a); }

__DEVICE__ double __dsqrt_rn(double a) { return __nv_dsqrt_rn(a); }
__DEVICE__ double __dsqrt_rz(double a) { return __nv_dsqrt_rz(a); }
__DEVICE__ double __dsqrt_rd(double a) { return __nv_dsqrt_rd(a); }
__DEVICE__ double __dsqrt_ru(double a) { return __nv_dsqrt_ru(a); }

// Conversion intrinsics
__DEVICE__ int __float2int_rn(float a) { return __nv_float2int_rn(a); }
__DEVICE__ int __float2int_rz(float a) { return __nv_float2int_rz(a); }
__DEVICE__ int __float2int_rd(float a) { return __nv_float2int_rd(a); }
__DEVICE__ int __float2int_ru(float a) { return __nv_float2int_ru(a); }

__DEVICE__ unsigned int __float2uint_rn(float a) { return __nv_float2uint_rn(a); }
__DEVICE__ unsigned int __float2uint_rz(float a) { return __nv_float2uint_rz(a); }
__DEVICE__ unsigned int __float2uint_rd(float a) { return __nv_float2uint_rd(a); }
__DEVICE__ unsigned int __float2uint_ru(float a) { return __nv_float2uint_ru(a); }

__DEVICE__ float __int2float_rn(int a) { return __nv_int2float_rn(a); }
__DEVICE__ float __int2float_rz(int a) { return __nv_int2float_rz(a); }
__DEVICE__ float __int2float_rd(int a) { return __nv_int2float_rd(a); }
__DEVICE__ float __int2float_ru(int a) { return __nv_int2float_ru(a); }

__DEVICE__ float __uint2float_rn(unsigned int a) { return __nv_uint2float_rn(a); }
__DEVICE__ float __uint2float_rz(unsigned int a) { return __nv_uint2float_rz(a); }
__DEVICE__ float __uint2float_rd(unsigned int a) { return __nv_uint2float_rd(a); }
__DEVICE__ float __uint2float_ru(unsigned int a) { return __nv_uint2float_ru(a); }

// Long long conversions
__DEVICE__ long long __float2ll_rn(float a) { return __nv_float2ll_rn(a); }
__DEVICE__ long long __float2ll_rz(float a) { return __nv_float2ll_rz(a); }
__DEVICE__ long long __float2ll_rd(float a) { return __nv_float2ll_rd(a); }
__DEVICE__ long long __float2ll_ru(float a) { return __nv_float2ll_ru(a); }

__DEVICE__ unsigned long long __float2ull_rn(float a) { return __nv_float2ull_rn(a); }
__DEVICE__ unsigned long long __float2ull_rz(float a) { return __nv_float2ull_rz(a); }
__DEVICE__ unsigned long long __float2ull_rd(float a) { return __nv_float2ull_rd(a); }
__DEVICE__ unsigned long long __float2ull_ru(float a) { return __nv_float2ull_ru(a); }

__DEVICE__ float __ll2float_rn(long long a) { return __nv_ll2float_rn(a); }
__DEVICE__ float __ll2float_rz(long long a) { return __nv_ll2float_rz(a); }
__DEVICE__ float __ll2float_rd(long long a) { return __nv_ll2float_rd(a); }
__DEVICE__ float __ll2float_ru(long long a) { return __nv_ll2float_ru(a); }

__DEVICE__ float __ull2float_rn(unsigned long long a) { return __nv_ull2float_rn(a); }
__DEVICE__ float __ull2float_rz(unsigned long long a) { return __nv_ull2float_rz(a); }
__DEVICE__ float __ull2float_rd(unsigned long long a) { return __nv_ull2float_rd(a); }
__DEVICE__ float __ull2float_ru(unsigned long long a) { return __nv_ull2float_ru(a); }

// Double conversions
__DEVICE__ int __double2int_rn(double a) { return __nv_double2int_rn(a); }
__DEVICE__ int __double2int_rz(double a) { return __nv_double2int_rz(a); }
__DEVICE__ int __double2int_rd(double a) { return __nv_double2int_rd(a); }
__DEVICE__ int __double2int_ru(double a) { return __nv_double2int_ru(a); }

__DEVICE__ unsigned int __double2uint_rn(double a) { return __nv_double2uint_rn(a); }
__DEVICE__ unsigned int __double2uint_rz(double a) { return __nv_double2uint_rz(a); }
__DEVICE__ unsigned int __double2uint_rd(double a) { return __nv_double2uint_rd(a); }
__DEVICE__ unsigned int __double2uint_ru(double a) { return __nv_double2uint_ru(a); }

__DEVICE__ long long __double2ll_rn(double a) { return __nv_double2ll_rn(a); }
__DEVICE__ long long __double2ll_rz(double a) { return __nv_double2ll_rz(a); }
__DEVICE__ long long __double2ll_rd(double a) { return __nv_double2ll_rd(a); }
__DEVICE__ long long __double2ll_ru(double a) { return __nv_double2ll_ru(a); }

__DEVICE__ unsigned long long __double2ull_rn(double a) { return __nv_double2ull_rn(a); }
__DEVICE__ unsigned long long __double2ull_rz(double a) { return __nv_double2ull_rz(a); }
__DEVICE__ unsigned long long __double2ull_rd(double a) { return __nv_double2ull_rd(a); }
__DEVICE__ unsigned long long __double2ull_ru(double a) { return __nv_double2ull_ru(a); }

__DEVICE__ double __int2double_rn(int a) { return __nv_int2double_rn(a); }
__DEVICE__ double __uint2double_rn(unsigned int a) { return __nv_uint2double_rn(a); }
__DEVICE__ double __ll2double_rn(long long a) { return __nv_ll2double_rn(a); }
__DEVICE__ double __ll2double_rz(long long a) { return __nv_ll2double_rz(a); }
__DEVICE__ double __ll2double_rd(long long a) { return __nv_ll2double_rd(a); }
__DEVICE__ double __ll2double_ru(long long a) { return __nv_ll2double_ru(a); }
__DEVICE__ double __ull2double_rn(unsigned long long a) { return __nv_ull2double_rn(a); }
__DEVICE__ double __ull2double_rz(unsigned long long a) { return __nv_ull2double_rz(a); }
__DEVICE__ double __ull2double_rd(unsigned long long a) { return __nv_ull2double_rd(a); }
__DEVICE__ double __ull2double_ru(unsigned long long a) { return __nv_ull2double_ru(a); }

// Float-double conversions
__DEVICE__ float __double2float_rn(double a) { return __nv_double2float_rn(a); }
__DEVICE__ float __double2float_rz(double a) { return __nv_double2float_rz(a); }
__DEVICE__ float __double2float_rd(double a) { return __nv_double2float_rd(a); }
__DEVICE__ float __double2float_ru(double a) { return __nv_double2float_ru(a); }

// Saturating conversions
__DEVICE__ int __float2int_rn_sat(float a) { return __nv_float2int_rn(a < -2147483648.0f ? -2147483648.0f : (a > 2147483647.0f ? 2147483647.0f : a)); }

// Note: Half precision conversions (__half2float, __float2half_rn) are
// provided by cuda_fp16.h with __host__ __device__ attributes, so we don't
// define them here to avoid conflicts.

// Bit casting intrinsics
__DEVICE__ unsigned int __float_as_uint(float a) { return __nv_float_as_uint(a); }
__DEVICE__ int __float_as_int(float a) { return __nv_float_as_int(a); }
__DEVICE__ float __uint_as_float(unsigned int a) { return __nv_uint_as_float(a); }
__DEVICE__ float __int_as_float(int a) { return __nv_int_as_float(a); }
__DEVICE__ unsigned long long __double_as_longlong(double a) { return __nv_double_as_longlong(a); }
__DEVICE__ double __longlong_as_double(unsigned long long a) { return __nv_longlong_as_double(a); }

// IEEE FMA
__DEVICE__ float __fmaf_ieee_rn(float a, float b, float c) { return __nv_fmaf_ieee_rn(a, b, c); }
__DEVICE__ float __fmaf_ieee_rz(float a, float b, float c) { return __nv_fmaf_ieee_rz(a, b, c); }
__DEVICE__ float __fmaf_ieee_rd(float a, float b, float c) { return __nv_fmaf_ieee_rd(a, b, c); }
__DEVICE__ float __fmaf_ieee_ru(float a, float b, float c) { return __nv_fmaf_ieee_ru(a, b, c); }

// Basic math functions (float)
__DEVICE__ float fabsf(float a) { return __nv_fabsf(a); }
__DEVICE__ float sqrtf(float a) { return __nv_sqrtf(a); }
__DEVICE__ float rsqrtf(float a) { return __nv_rsqrtf(a); }
__DEVICE__ float sinf(float a) { return __nv_sinf(a); }
__DEVICE__ float cosf(float a) { return __nv_cosf(a); }
__DEVICE__ float tanf(float a) { return __nv_tanf(a); }
__DEVICE__ float tanhf(float a) { return __nv_tanhf(a); }
__DEVICE__ float sinhf(float a) { return __nv_sinhf(a); }
__DEVICE__ float coshf(float a) { return __nv_coshf(a); }
__DEVICE__ float expf(float a) { return __nv_expf(a); }
__DEVICE__ float exp2f(float a) { return __nv_exp2f(a); }
__DEVICE__ float exp10f(float a) { return __nv_exp10f(a); }
__DEVICE__ float logf(float a) { return __nv_logf(a); }
__DEVICE__ float log2f(float a) { return __nv_log2f(a); }
__DEVICE__ float log10f(float a) { return __nv_log10f(a); }
__DEVICE__ float powf(float a, float b) { return __nv_powf(a, b); }
__DEVICE__ float floorf(float a) { return __nv_floorf(a); }
__DEVICE__ float ceilf(float a) { return __nv_ceilf(a); }
__DEVICE__ float truncf(float a) { return __nv_truncf(a); }
__DEVICE__ float roundf(float a) { return __nv_roundf(a); }
__DEVICE__ float fmodf(float a, float b) { return __nv_fmodf(a, b); }
__DEVICE__ float fmaxf(float a, float b) { return __nv_fmaxf(a, b); }
__DEVICE__ float fminf(float a, float b) { return __nv_fminf(a, b); }
__DEVICE__ float copysignf(float a, float b) { return __nv_copysignf(a, b); }
__DEVICE__ float atanf(float a) { return __nv_atanf(a); }
__DEVICE__ float atan2f(float a, float b) { return __nv_atan2f(a, b); }
__DEVICE__ float asinf(float a) { return __nv_asinf(a); }
__DEVICE__ float acosf(float a) { return __nv_acosf(a); }

// Basic math functions (double)
__DEVICE__ double fabs(double a) { return __nv_fabs(a); }
__DEVICE__ double sqrt(double a) { return __nv_sqrt(a); }
__DEVICE__ double rsqrt(double a) { return __nv_rsqrt(a); }
__DEVICE__ double sin(double a) { return __nv_sin(a); }
__DEVICE__ double cos(double a) { return __nv_cos(a); }
__DEVICE__ double tan(double a) { return __nv_tan(a); }
__DEVICE__ double tanh(double a) { return __nv_tanh(a); }
__DEVICE__ double sinh(double a) { return __nv_sinh(a); }
__DEVICE__ double cosh(double a) { return __nv_cosh(a); }
__DEVICE__ double exp(double a) { return __nv_exp(a); }
__DEVICE__ double exp2(double a) { return __nv_exp2(a); }
__DEVICE__ double exp10(double a) { return __nv_exp10(a); }
__DEVICE__ double log(double a) { return __nv_log(a); }
__DEVICE__ double log2(double a) { return __nv_log2(a); }
__DEVICE__ double log10(double a) { return __nv_log10(a); }
__DEVICE__ double pow(double a, double b) { return __nv_pow(a, b); }
__DEVICE__ double floor(double a) { return __nv_floor(a); }
__DEVICE__ double ceil(double a) { return __nv_ceil(a); }
__DEVICE__ double trunc(double a) { return __nv_trunc(a); }
__DEVICE__ double round(double a) { return __nv_round(a); }
__DEVICE__ double fmod(double a, double b) { return __nv_fmod(a, b); }
__DEVICE__ double fmax(double a, double b) { return __nv_fmax(a, b); }
__DEVICE__ double fmin(double a, double b) { return __nv_fmin(a, b); }
__DEVICE__ double copysign(double a, double b) { return __nv_copysign(a, b); }
__DEVICE__ double atan(double a) { return __nv_atan(a); }
__DEVICE__ double atan2(double a, double b) { return __nv_atan2(a, b); }
__DEVICE__ double asin(double a) { return __nv_asin(a); }
__DEVICE__ double acos(double a) { return __nv_acos(a); }

// Address space query intrinsics (for libcudacxx)
// These return whether a pointer is in a specific address space
__DEVICE__ bool __isGlobal(const void* ptr) {
    return __nvvm_isspacep_global(ptr);
}
__DEVICE__ bool __isShared(const void* ptr) {
    return __nvvm_isspacep_shared(ptr);
}
__DEVICE__ bool __isConstant(const void* ptr) {
    return __nvvm_isspacep_const(ptr);
}
__DEVICE__ bool __isLocal(const void* ptr) {
    return __nvvm_isspacep_local(ptr);
}
#if __CUDA_ARCH__ >= 900
__DEVICE__ bool __isClusterShared(const void* ptr) {
    // Cluster shared memory query - requires sm_90+
    return false; // Stub for now
}
#endif

// Warp shuffle intrinsics with sync
// These are templates/overloads that libcudacxx expects
__DEVICE__ int __shfl_sync(unsigned mask, int val, int src, int width = 32) {
    return __nvvm_shfl_sync_idx_i32(mask, val, src, ((32 - width) << 8) | 0x1f);
}
__DEVICE__ unsigned __shfl_sync(unsigned mask, unsigned val, int src, int width = 32) {
    return static_cast<unsigned>(__nvvm_shfl_sync_idx_i32(mask, static_cast<int>(val), src, ((32 - width) << 8) | 0x1f));
}
__DEVICE__ long long __shfl_sync(unsigned mask, long long val, int src, int width = 32) {
    int lo = static_cast<int>(val);
    int hi = static_cast<int>(val >> 32);
    lo = __nvvm_shfl_sync_idx_i32(mask, lo, src, ((32 - width) << 8) | 0x1f);
    hi = __nvvm_shfl_sync_idx_i32(mask, hi, src, ((32 - width) << 8) | 0x1f);
    return (static_cast<long long>(hi) << 32) | static_cast<unsigned>(lo);
}
__DEVICE__ unsigned long long __shfl_sync(unsigned mask, unsigned long long val, int src, int width = 32) {
    return static_cast<unsigned long long>(__shfl_sync(mask, static_cast<long long>(val), src, width));
}
__DEVICE__ float __shfl_sync(unsigned mask, float val, int src, int width = 32) {
    return __nvvm_shfl_sync_idx_f32(mask, val, src, ((32 - width) << 8) | 0x1f);
}
__DEVICE__ double __shfl_sync(unsigned mask, double val, int src, int width = 32) {
    long long tmp = __double_as_longlong(val);
    tmp = __shfl_sync(mask, tmp, src, width);
    return __longlong_as_double(tmp);
}

__DEVICE__ int __shfl_up_sync(unsigned mask, int val, unsigned delta, int width = 32) {
    return __nvvm_shfl_sync_up_i32(mask, val, delta, (32 - width) << 8);
}
__DEVICE__ unsigned __shfl_up_sync(unsigned mask, unsigned val, unsigned delta, int width = 32) {
    return static_cast<unsigned>(__nvvm_shfl_sync_up_i32(mask, static_cast<int>(val), delta, (32 - width) << 8));
}
__DEVICE__ long long __shfl_up_sync(unsigned mask, long long val, unsigned delta, int width = 32) {
    int lo = static_cast<int>(val);
    int hi = static_cast<int>(val >> 32);
    lo = __nvvm_shfl_sync_up_i32(mask, lo, delta, (32 - width) << 8);
    hi = __nvvm_shfl_sync_up_i32(mask, hi, delta, (32 - width) << 8);
    return (static_cast<long long>(hi) << 32) | static_cast<unsigned>(lo);
}
__DEVICE__ unsigned long long __shfl_up_sync(unsigned mask, unsigned long long val, unsigned delta, int width = 32) {
    return static_cast<unsigned long long>(__shfl_up_sync(mask, static_cast<long long>(val), delta, width));
}
__DEVICE__ float __shfl_up_sync(unsigned mask, float val, unsigned delta, int width = 32) {
    return __nvvm_shfl_sync_up_f32(mask, val, delta, (32 - width) << 8);
}
__DEVICE__ double __shfl_up_sync(unsigned mask, double val, unsigned delta, int width = 32) {
    long long tmp = __double_as_longlong(val);
    tmp = __shfl_up_sync(mask, tmp, delta, width);
    return __longlong_as_double(tmp);
}

__DEVICE__ int __shfl_down_sync(unsigned mask, int val, unsigned delta, int width = 32) {
    return __nvvm_shfl_sync_down_i32(mask, val, delta, ((32 - width) << 8) | 0x1f);
}
__DEVICE__ unsigned __shfl_down_sync(unsigned mask, unsigned val, unsigned delta, int width = 32) {
    return static_cast<unsigned>(__nvvm_shfl_sync_down_i32(mask, static_cast<int>(val), delta, ((32 - width) << 8) | 0x1f));
}
__DEVICE__ long long __shfl_down_sync(unsigned mask, long long val, unsigned delta, int width = 32) {
    int lo = static_cast<int>(val);
    int hi = static_cast<int>(val >> 32);
    lo = __nvvm_shfl_sync_down_i32(mask, lo, delta, ((32 - width) << 8) | 0x1f);
    hi = __nvvm_shfl_sync_down_i32(mask, hi, delta, ((32 - width) << 8) | 0x1f);
    return (static_cast<long long>(hi) << 32) | static_cast<unsigned>(lo);
}
__DEVICE__ unsigned long long __shfl_down_sync(unsigned mask, unsigned long long val, unsigned delta, int width = 32) {
    return static_cast<unsigned long long>(__shfl_down_sync(mask, static_cast<long long>(val), delta, width));
}
__DEVICE__ float __shfl_down_sync(unsigned mask, float val, unsigned delta, int width = 32) {
    return __nvvm_shfl_sync_down_f32(mask, val, delta, ((32 - width) << 8) | 0x1f);
}
__DEVICE__ double __shfl_down_sync(unsigned mask, double val, unsigned delta, int width = 32) {
    long long tmp = __double_as_longlong(val);
    tmp = __shfl_down_sync(mask, tmp, delta, width);
    return __longlong_as_double(tmp);
}

__DEVICE__ int __shfl_xor_sync(unsigned mask, int val, int lane_mask, int width = 32) {
    return __nvvm_shfl_sync_bfly_i32(mask, val, lane_mask, ((32 - width) << 8) | 0x1f);
}
__DEVICE__ unsigned __shfl_xor_sync(unsigned mask, unsigned val, int lane_mask, int width = 32) {
    return static_cast<unsigned>(__nvvm_shfl_sync_bfly_i32(mask, static_cast<int>(val), lane_mask, ((32 - width) << 8) | 0x1f));
}
__DEVICE__ long long __shfl_xor_sync(unsigned mask, long long val, int lane_mask, int width = 32) {
    int lo = static_cast<int>(val);
    int hi = static_cast<int>(val >> 32);
    lo = __nvvm_shfl_sync_bfly_i32(mask, lo, lane_mask, ((32 - width) << 8) | 0x1f);
    hi = __nvvm_shfl_sync_bfly_i32(mask, hi, lane_mask, ((32 - width) << 8) | 0x1f);
    return (static_cast<long long>(hi) << 32) | static_cast<unsigned>(lo);
}
__DEVICE__ unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long val, int lane_mask, int width = 32) {
    return static_cast<unsigned long long>(__shfl_xor_sync(mask, static_cast<long long>(val), lane_mask, width));
}
__DEVICE__ float __shfl_xor_sync(unsigned mask, float val, int lane_mask, int width = 32) {
    return __nvvm_shfl_sync_bfly_f32(mask, val, lane_mask, ((32 - width) << 8) | 0x1f);
}
__DEVICE__ double __shfl_xor_sync(unsigned mask, double val, int lane_mask, int width = 32) {
    long long tmp = __double_as_longlong(val);
    tmp = __shfl_xor_sync(mask, tmp, lane_mask, width);
    return __longlong_as_double(tmp);
}

// Note: The following are provided by __clang_cuda_device_functions.h:
// - Warp synchronization (__syncwarp)
// - Trap (__trap)
// - Address space conversion intrinsics (__cvta_*)
// - Atomic operations (atomicAdd, atomicCAS, etc.)
// - Bit manipulation (__clz, __ffs, __popc, __brev)

#undef __DEVICE__

#endif // _CLANGJIT_DEVICE_INTRINSICS_H
