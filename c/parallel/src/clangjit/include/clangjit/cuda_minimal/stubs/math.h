#ifndef _CLANGJIT_MATH_H
#define _CLANGJIT_MATH_H
#define HUGE_VAL __builtin_huge_val()
#define HUGE_VALF __builtin_huge_valf()
#define HUGE_VALL __builtin_huge_vall()
#define INFINITY __builtin_inff()
#define NAN __builtin_nanf("")
#define FP_INFINITE 1
#define FP_NAN 0
#define FP_NORMAL 4
#define FP_SUBNORMAL 3
#define FP_ZERO 2
// glibc-specific — only declare ones NOT in __clang_cuda_device_functions.h
#ifdef __cplusplus
extern "C" {
#endif
int __signbit(double);
int __signbitl(long double);
int __finitel(long double);
int __isnanl(long double);
int __isinfl(long double);
#ifdef __cplusplus
}
#endif
#endif
