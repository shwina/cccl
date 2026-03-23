// Minimal assert.h stub for CUDA JIT compilation
#ifndef _CLANGJIT_ASSERT_H
#define _CLANGJIT_ASSERT_H

#ifdef NDEBUG
#define assert(expr) ((void)0)
#else
#define assert(expr) ((void)((expr) || (__assert_fail(#expr, __FILE__, __LINE__, __func__), 0)))
#endif

extern "C" void __assert_fail(const char*, const char*, unsigned int, const char*);

#endif // _CLANGJIT_ASSERT_H
