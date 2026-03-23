// Minimal math.h stub for CUDA JIT compilation
#ifndef _CLANGJIT_MATH_H
#define _CLANGJIT_MATH_H

// Define math constants needed by CUDA
#define HUGE_VAL __builtin_huge_val()
#define HUGE_VALF __builtin_huge_valf()
#define HUGE_VALL __builtin_huge_vall()
#define INFINITY __builtin_inf()
#define NAN __builtin_nanf("")

// Classification macros
#define FP_NAN       0
#define FP_INFINITE  1
#define FP_ZERO      2
#define FP_SUBNORMAL 3
#define FP_NORMAL    4

// Math error constants
#define MATH_ERRNO        1
#define MATH_ERREXCEPT    2
#define math_errhandling  (MATH_ERRNO | MATH_ERREXCEPT)

// Floating-point types (typically same as float/double)
typedef float float_t;
typedef double double_t;

// Define CUDA attributes if not already defined - must be before libdevice declares
#if defined(__CUDACC__) || defined(__CUDA__)
#  ifndef __host__
#    define __host__ __attribute__((host))
#  endif
#  ifndef __device__
#    define __device__ __attribute__((device))
#  endif
#  define _CLANGJIT_HD __host__ __device__
#else
#  define _CLANGJIT_HD
#endif

// Ensure we have the libdevice declarations (needs __device__ to be defined)
#include <__clang_cuda_libdevice_declares.h>

#define __CLANGJIT_MATH_FN static _CLANGJIT_HD __inline__ __attribute__((always_inline))

// Float math functions
__CLANGJIT_MATH_FN float fabsf(float x) { return __nv_fabsf(x); }
__CLANGJIT_MATH_FN float sqrtf(float x) { return __nv_sqrtf(x); }
__CLANGJIT_MATH_FN float rsqrtf(float x) { return __nv_rsqrtf(x); }
__CLANGJIT_MATH_FN float cbrtf(float x) { return __nv_cbrtf(x); }
__CLANGJIT_MATH_FN float sinf(float x) { return __nv_sinf(x); }
__CLANGJIT_MATH_FN float cosf(float x) { return __nv_cosf(x); }
__CLANGJIT_MATH_FN float tanf(float x) { return __nv_tanf(x); }
__CLANGJIT_MATH_FN float asinf(float x) { return __nv_asinf(x); }
__CLANGJIT_MATH_FN float acosf(float x) { return __nv_acosf(x); }
__CLANGJIT_MATH_FN float atanf(float x) { return __nv_atanf(x); }
__CLANGJIT_MATH_FN float atan2f(float y, float x) { return __nv_atan2f(y, x); }
__CLANGJIT_MATH_FN float sinhf(float x) { return __nv_sinhf(x); }
__CLANGJIT_MATH_FN float coshf(float x) { return __nv_coshf(x); }
__CLANGJIT_MATH_FN float tanhf(float x) { return __nv_tanhf(x); }
__CLANGJIT_MATH_FN float asinhf(float x) { return __nv_asinhf(x); }
__CLANGJIT_MATH_FN float acoshf(float x) { return __nv_acoshf(x); }
__CLANGJIT_MATH_FN float atanhf(float x) { return __nv_atanhf(x); }
__CLANGJIT_MATH_FN float expf(float x) { return __nv_expf(x); }
__CLANGJIT_MATH_FN float exp2f(float x) { return __nv_exp2f(x); }
__CLANGJIT_MATH_FN float exp10f(float x) { return __nv_exp10f(x); }
__CLANGJIT_MATH_FN float expm1f(float x) { return __nv_expm1f(x); }
__CLANGJIT_MATH_FN float logf(float x) { return __nv_logf(x); }
__CLANGJIT_MATH_FN float log2f(float x) { return __nv_log2f(x); }
__CLANGJIT_MATH_FN float log10f(float x) { return __nv_log10f(x); }
__CLANGJIT_MATH_FN float log1pf(float x) { return __nv_log1pf(x); }
__CLANGJIT_MATH_FN float logbf(float x) { return __nv_logbf(x); }
__CLANGJIT_MATH_FN int ilogbf(float x) { return __nv_ilogbf(x); }
__CLANGJIT_MATH_FN float powf(float x, float y) { return __nv_powf(x, y); }
__CLANGJIT_MATH_FN float floorf(float x) { return __nv_floorf(x); }
__CLANGJIT_MATH_FN float ceilf(float x) { return __nv_ceilf(x); }
__CLANGJIT_MATH_FN float truncf(float x) { return __nv_truncf(x); }
__CLANGJIT_MATH_FN float roundf(float x) { return __nv_roundf(x); }
__CLANGJIT_MATH_FN long lroundf(float x) { return __nv_llroundf(x); }
__CLANGJIT_MATH_FN long long llroundf(float x) { return __nv_llroundf(x); }
__CLANGJIT_MATH_FN float rintf(float x) { return __nv_rintf(x); }
__CLANGJIT_MATH_FN long lrintf(float x) { return __nv_llrintf(x); }
__CLANGJIT_MATH_FN long long llrintf(float x) { return __nv_llrintf(x); }
__CLANGJIT_MATH_FN float nearbyintf(float x) { return __nv_nearbyintf(x); }
__CLANGJIT_MATH_FN float fmodf(float x, float y) { return __nv_fmodf(x, y); }
__CLANGJIT_MATH_FN float remainderf(float x, float y) { return __nv_remainderf(x, y); }
__CLANGJIT_MATH_FN float fmaxf(float x, float y) { return __nv_fmaxf(x, y); }
__CLANGJIT_MATH_FN float fminf(float x, float y) { return __nv_fminf(x, y); }
__CLANGJIT_MATH_FN float fdimf(float x, float y) { return __nv_fdimf(x, y); }
__CLANGJIT_MATH_FN float fmaf(float x, float y, float z) { return __nv_fmaf(x, y, z); }
__CLANGJIT_MATH_FN float copysignf(float x, float y) { return __nv_copysignf(x, y); }
__CLANGJIT_MATH_FN float scalbnf(float x, int n) { return __nv_scalbnf(x, n); }
__CLANGJIT_MATH_FN float scalblnf(float x, long n) { return __nv_scalbnf(x, static_cast<int>(n)); }
__CLANGJIT_MATH_FN float frexpf(float x, int* exp) { return __nv_frexpf(x, exp); }
__CLANGJIT_MATH_FN float ldexpf(float x, int exp) { return __nv_ldexpf(x, exp); }
__CLANGJIT_MATH_FN float modff(float x, float* iptr) { return __nv_modff(x, iptr); }
__CLANGJIT_MATH_FN float hypotf(float x, float y) { return __nv_hypotf(x, y); }
__CLANGJIT_MATH_FN float erff(float x) { return __nv_erff(x); }
__CLANGJIT_MATH_FN float erfcf(float x) { return __nv_erfcf(x); }
__CLANGJIT_MATH_FN float tgammaf(float x) { return __nv_tgammaf(x); }
__CLANGJIT_MATH_FN float lgammaf(float x) { return __nv_lgammaf(x); }
__CLANGJIT_MATH_FN float nextafterf(float x, float y) { return __nv_nextafterf(x, y); }

// Double math functions
__CLANGJIT_MATH_FN double fabs(double x) { return __nv_fabs(x); }
__CLANGJIT_MATH_FN double sqrt(double x) { return __nv_sqrt(x); }
__CLANGJIT_MATH_FN double rsqrt(double x) { return __nv_rsqrt(x); }
__CLANGJIT_MATH_FN double cbrt(double x) { return __nv_cbrt(x); }
__CLANGJIT_MATH_FN double sin(double x) { return __nv_sin(x); }
__CLANGJIT_MATH_FN double cos(double x) { return __nv_cos(x); }
__CLANGJIT_MATH_FN double tan(double x) { return __nv_tan(x); }
__CLANGJIT_MATH_FN double asin(double x) { return __nv_asin(x); }
__CLANGJIT_MATH_FN double acos(double x) { return __nv_acos(x); }
__CLANGJIT_MATH_FN double atan(double x) { return __nv_atan(x); }
__CLANGJIT_MATH_FN double atan2(double y, double x) { return __nv_atan2(y, x); }
__CLANGJIT_MATH_FN double sinh(double x) { return __nv_sinh(x); }
__CLANGJIT_MATH_FN double cosh(double x) { return __nv_cosh(x); }
__CLANGJIT_MATH_FN double tanh(double x) { return __nv_tanh(x); }
__CLANGJIT_MATH_FN double asinh(double x) { return __nv_asinh(x); }
__CLANGJIT_MATH_FN double acosh(double x) { return __nv_acosh(x); }
__CLANGJIT_MATH_FN double atanh(double x) { return __nv_atanh(x); }
__CLANGJIT_MATH_FN double exp(double x) { return __nv_exp(x); }
__CLANGJIT_MATH_FN double exp2(double x) { return __nv_exp2(x); }
__CLANGJIT_MATH_FN double exp10(double x) { return __nv_exp10(x); }
__CLANGJIT_MATH_FN double expm1(double x) { return __nv_expm1(x); }
__CLANGJIT_MATH_FN double log(double x) { return __nv_log(x); }
__CLANGJIT_MATH_FN double log2(double x) { return __nv_log2(x); }
__CLANGJIT_MATH_FN double log10(double x) { return __nv_log10(x); }
__CLANGJIT_MATH_FN double log1p(double x) { return __nv_log1p(x); }
__CLANGJIT_MATH_FN double logb(double x) { return __nv_logb(x); }
__CLANGJIT_MATH_FN int ilogb(double x) { return __nv_ilogb(x); }
__CLANGJIT_MATH_FN double pow(double x, double y) { return __nv_pow(x, y); }
__CLANGJIT_MATH_FN double floor(double x) { return __nv_floor(x); }
__CLANGJIT_MATH_FN double ceil(double x) { return __nv_ceil(x); }
__CLANGJIT_MATH_FN double trunc(double x) { return __nv_trunc(x); }
__CLANGJIT_MATH_FN double round(double x) { return __nv_round(x); }
__CLANGJIT_MATH_FN long lround(double x) { return __nv_llround(x); }
__CLANGJIT_MATH_FN long long llround(double x) { return __nv_llround(x); }
__CLANGJIT_MATH_FN double rint(double x) { return __nv_rint(x); }
__CLANGJIT_MATH_FN long lrint(double x) { return __nv_llrint(x); }
__CLANGJIT_MATH_FN long long llrint(double x) { return __nv_llrint(x); }
__CLANGJIT_MATH_FN double nearbyint(double x) { return __nv_nearbyint(x); }
__CLANGJIT_MATH_FN double fmod(double x, double y) { return __nv_fmod(x, y); }
__CLANGJIT_MATH_FN double remainder(double x, double y) { return __nv_remainder(x, y); }
__CLANGJIT_MATH_FN double fmax(double x, double y) { return __nv_fmax(x, y); }
__CLANGJIT_MATH_FN double fmin(double x, double y) { return __nv_fmin(x, y); }
__CLANGJIT_MATH_FN double fdim(double x, double y) { return __nv_fdim(x, y); }
__CLANGJIT_MATH_FN double fma(double x, double y, double z) { return __nv_fma(x, y, z); }
__CLANGJIT_MATH_FN double copysign(double x, double y) { return __nv_copysign(x, y); }
__CLANGJIT_MATH_FN double scalbn(double x, int n) { return __nv_scalbn(x, n); }
__CLANGJIT_MATH_FN double scalbln(double x, long n) { return __nv_scalbn(x, static_cast<int>(n)); }
__CLANGJIT_MATH_FN double frexp(double x, int* exp) { return __nv_frexp(x, exp); }
__CLANGJIT_MATH_FN double ldexp(double x, int exp) { return __nv_ldexp(x, exp); }
__CLANGJIT_MATH_FN double modf(double x, double* iptr) { return __nv_modf(x, iptr); }
__CLANGJIT_MATH_FN double hypot(double x, double y) { return __nv_hypot(x, y); }
__CLANGJIT_MATH_FN double erf(double x) { return __nv_erf(x); }
__CLANGJIT_MATH_FN double erfc(double x) { return __nv_erfc(x); }
__CLANGJIT_MATH_FN double tgamma(double x) { return __nv_tgamma(x); }
__CLANGJIT_MATH_FN double lgamma(double x) { return __nv_lgamma(x); }
__CLANGJIT_MATH_FN double nextafter(double x, double y) { return __nv_nextafter(x, y); }

// Note: labs and llabs are provided by cstdlib
// Note: Classification functions (__isnan*, __isinf*, __finite*, __signbit*)
// are provided by __clang_cuda_device_functions.h

// NaN generation functions
__CLANGJIT_MATH_FN float nanf(const char*) { return __builtin_nanf(""); }
__CLANGJIT_MATH_FN double nan(const char*) { return __builtin_nan(""); }
__CLANGJIT_MATH_FN long double nanl(const char*) { return __builtin_nanl(""); }

// sincos (combined sine and cosine)
__CLANGJIT_MATH_FN void sincosf(float x, float* s, float* c) { __nv_sincosf(x, s, c); }
__CLANGJIT_MATH_FN void sincos(double x, double* s, double* c) { __nv_sincos(x, s, c); }

// nexttoward functions
__CLANGJIT_MATH_FN float nexttowardf(float x, long double y) { return __nv_nextafterf(x, static_cast<float>(y)); }
__CLANGJIT_MATH_FN double nexttoward(double x, long double y) { return __nv_nextafter(x, static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double nexttowardl(long double x, long double y) { return __nv_nextafter(static_cast<double>(x), static_cast<double>(y)); }

// remquo functions
__CLANGJIT_MATH_FN float remquof(float x, float y, int* quo) { return __nv_remquof(x, y, quo); }
__CLANGJIT_MATH_FN double remquo(double x, double y, int* quo) { return __nv_remquo(x, y, quo); }
__CLANGJIT_MATH_FN long double remquol(long double x, long double y, int* quo) { return __nv_remquo(static_cast<double>(x), static_cast<double>(y), quo); }

// Long double versions (forwarding to double since CUDA doesn't have native long double)
__CLANGJIT_MATH_FN long double fabsl(long double x) { return __nv_fabs(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double sqrtl(long double x) { return __nv_sqrt(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double cbrtl(long double x) { return __nv_cbrt(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double sinl(long double x) { return __nv_sin(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double cosl(long double x) { return __nv_cos(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double tanl(long double x) { return __nv_tan(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double asinl(long double x) { return __nv_asin(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double acosl(long double x) { return __nv_acos(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double atanl(long double x) { return __nv_atan(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double atan2l(long double y, long double x) { return __nv_atan2(static_cast<double>(y), static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double sinhl(long double x) { return __nv_sinh(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double coshl(long double x) { return __nv_cosh(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double tanhl(long double x) { return __nv_tanh(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double asinhl(long double x) { return __nv_asinh(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double acoshl(long double x) { return __nv_acosh(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double atanhl(long double x) { return __nv_atanh(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double expl(long double x) { return __nv_exp(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double exp2l(long double x) { return __nv_exp2(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double expm1l(long double x) { return __nv_expm1(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double logl(long double x) { return __nv_log(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double log2l(long double x) { return __nv_log2(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double log10l(long double x) { return __nv_log10(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double log1pl(long double x) { return __nv_log1p(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double logbl(long double x) { return __nv_logb(static_cast<double>(x)); }
__CLANGJIT_MATH_FN int ilogbl(long double x) { return __nv_ilogb(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double powl(long double x, long double y) { return __nv_pow(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double floorl(long double x) { return __nv_floor(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double ceill(long double x) { return __nv_ceil(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double truncl(long double x) { return __nv_trunc(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double roundl(long double x) { return __nv_round(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long lroundl(long double x) { return __nv_llround(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long long llroundl(long double x) { return __nv_llround(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double rintl(long double x) { return __nv_rint(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long lrintl(long double x) { return __nv_llrint(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long long llrintl(long double x) { return __nv_llrint(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double nearbyintl(long double x) { return __nv_nearbyint(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double fmodl(long double x, long double y) { return __nv_fmod(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double remainderl(long double x, long double y) { return __nv_remainder(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double fmaxl(long double x, long double y) { return __nv_fmax(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double fminl(long double x, long double y) { return __nv_fmin(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double fdiml(long double x, long double y) { return __nv_fdim(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double fmal(long double x, long double y, long double z) { return __nv_fma(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)); }
__CLANGJIT_MATH_FN long double copysignl(long double x, long double y) { return __nv_copysign(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double scalbnl(long double x, int n) { return __nv_scalbn(static_cast<double>(x), n); }
__CLANGJIT_MATH_FN long double scalblnl(long double x, long n) { return __nv_scalbn(static_cast<double>(x), static_cast<int>(n)); }
__CLANGJIT_MATH_FN long double frexpl(long double x, int* exp) { return __nv_frexp(static_cast<double>(x), exp); }
__CLANGJIT_MATH_FN long double ldexpl(long double x, int exp) { return __nv_ldexp(static_cast<double>(x), exp); }
__CLANGJIT_MATH_FN long double modfl(long double x, long double* iptr) { double i; double f = __nv_modf(static_cast<double>(x), &i); *iptr = i; return f; }
__CLANGJIT_MATH_FN long double hypotl(long double x, long double y) { return __nv_hypot(static_cast<double>(x), static_cast<double>(y)); }
__CLANGJIT_MATH_FN long double erfl(long double x) { return __nv_erf(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double erfcl(long double x) { return __nv_erfc(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double tgammal(long double x) { return __nv_tgamma(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double lgammal(long double x) { return __nv_lgamma(static_cast<double>(x)); }
__CLANGJIT_MATH_FN long double nextafterl(long double x, long double y) { return __nv_nextafter(static_cast<double>(x), static_cast<double>(y)); }

#undef __CLANGJIT_MATH_FN

#endif // _CLANGJIT_MATH_H
