#ifndef _CLANGJIT_MATH_H
#define _CLANGJIT_MATH_H
#define HUGE_VAL __builtin_huge_val()
#define HUGE_VALF __builtin_huge_valf()
#define HUGE_VALL __builtin_huge_vall()
#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")
#define MATH_ERRNO 1
#define MATH_ERREXCEPT 2
#define math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
#define FP_NAN       0
#define FP_INFINITE  1
#define FP_ZERO      2
#define FP_SUBNORMAL 3
#define FP_NORMAL    4
#define __signbit(x)  __builtin_signbit(x)
#define __signbitl(x) __builtin_signbitl(x)

// Standard math function declarations (host implementations link to libm)
#ifdef __cplusplus
extern "C" {
#endif
double sqrt(double);   float sqrtf(float);
double sin(double);    float sinf(float);
double cos(double);    float cosf(float);
double exp(double);    float expf(float);
double log(double);    float logf(float);
double log2(double);   float log2f(float);
double log1p(double);  float log1pf(float);
double fabs(double);   float fabsf(float);
double floor(double);  float floorf(float);
double round(double);  float roundf(float);
double copysign(double, double); float copysignf(float, float);
double exp2(double); float exp2f(float);
double erfc(double);   float erfcf(float);
// CUDA-specific math functions (defined in math_functions.hpp, which we skip)
float rsqrtf(float); float rcbrtf(float);
float sinpif(float); float cospif(float);
void sincospif(float, float*, float*);
float erfcinvf(float); float normcdfinvf(float);
float normcdff(float); float erfcxf(float);
#ifdef __cplusplus
}
#endif
#endif
