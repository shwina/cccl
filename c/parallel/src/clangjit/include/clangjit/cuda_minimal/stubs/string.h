#ifndef _CLANGJIT_STRING_H
#define _CLANGJIT_STRING_H
#include "stddef.h"
#ifdef __cplusplus
extern "C" {
#endif
// memcpy/memset are already declared as __host__ __device__ in
// __clang_cuda_device_functions.h; re-declaring them as __host__-only (extern
// "C") causes an overload conflict.  Skip them when that header is present.
#ifndef __CLANG_CUDA_DEVICE_FUNCTIONS_H__
void* memcpy(void*, const void*, size_t);
void* memset(void*, int, size_t);
#endif
int memcmp(const void*, const void*, size_t);
void* memmove(void*, const void*, size_t);
size_t strlen(const char*);
#ifdef __cplusplus
}
#endif
#endif
