// Minimal stddef.h stub for CUDA JIT compilation
// Provides only the types needed by CUDA headers

#ifndef _CLANGJIT_STDDEF_H
#define _CLANGJIT_STDDEF_H

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

#ifndef NULL
#ifdef __cplusplus
#define NULL nullptr
#else
#define NULL ((void*)0)
#endif
#endif

#ifndef offsetof
#define offsetof(type, member) __builtin_offsetof(type, member)
#endif

#endif // _CLANGJIT_STDDEF_H
