// Minimal string.h stub for CUDA JIT compilation
#ifndef _CLANGJIT_STRING_H
#define _CLANGJIT_STRING_H

#include "stddef.h"

// Provide __host__ __device__ versions using builtins for CUDA compatibility
// These work in both host and device contexts

// Define CUDA attributes if not already defined
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

// Memory functions
// Note: memcpy and memset are provided by __clang_cuda_device_functions.h for CUDA
// We provide the rest
_CLANGJIT_HD inline void* memmove(void* dest, const void* src, size_t n) {
    return __builtin_memmove(dest, src, n);
}

_CLANGJIT_HD inline int memcmp(const void* s1, const void* s2, size_t n) {
    return __builtin_memcmp(s1, s2, n);
}

_CLANGJIT_HD inline void* memchr(const void* s, int c, size_t n) {
    return __builtin_memchr(s, c, n);
}

// String length/comparison (have builtin support)
_CLANGJIT_HD inline size_t strlen(const char* s) {
    return __builtin_strlen(s);
}

_CLANGJIT_HD inline int strcmp(const char* s1, const char* s2) {
    return __builtin_strcmp(s1, s2);
}

_CLANGJIT_HD inline int strncmp(const char* s1, const char* s2, size_t n) {
    return __builtin_strncmp(s1, s2, n);
}

_CLANGJIT_HD inline char* strchr(const char* s, int c) {
    return __builtin_strchr(s, c);
}

_CLANGJIT_HD inline char* strrchr(const char* s, int c) {
    return __builtin_strrchr(s, c);
}

_CLANGJIT_HD inline char* strstr(const char* haystack, const char* needle) {
    return __builtin_strstr(haystack, needle);
}

// String copy functions (have builtin support)
_CLANGJIT_HD inline char* strcpy(char* dest, const char* src) {
    return __builtin_strcpy(dest, src);
}

_CLANGJIT_HD inline char* strncpy(char* dest, const char* src, size_t n) {
    return __builtin_strncpy(dest, src, n);
}

_CLANGJIT_HD inline char* strcat(char* dest, const char* src) {
    return __builtin_strcat(dest, src);
}

_CLANGJIT_HD inline char* strncat(char* dest, const char* src, size_t n) {
    return __builtin_strncat(dest, src, n);
}

// String span functions - implemented manually
_CLANGJIT_HD inline size_t strspn(const char* s, const char* accept) {
    size_t count = 0;
    for (; *s; ++s) {
        const char* a = accept;
        bool found = false;
        for (; *a; ++a) {
            if (*s == *a) { found = true; break; }
        }
        if (!found) break;
        ++count;
    }
    return count;
}

_CLANGJIT_HD inline size_t strcspn(const char* s, const char* reject) {
    size_t count = 0;
    for (; *s; ++s) {
        const char* r = reject;
        for (; *r; ++r) {
            if (*s == *r) return count;
        }
        ++count;
    }
    return count;
}

_CLANGJIT_HD inline char* strpbrk(const char* s, const char* accept) {
    for (; *s; ++s) {
        const char* a = accept;
        for (; *a; ++a) {
            if (*s == *a) return const_cast<char*>(s);
        }
    }
    return nullptr;
}

// Functions that are host-only (need runtime support) - just declarations
// These won't be called on device
extern "C" {
int strcoll(const char* s1, const char* s2);
size_t strxfrm(char* dest, const char* src, size_t n);
char* strtok(char* s, const char* delim);
char* strerror(int errnum);
}

#undef _CLANGJIT_HD

#endif // _CLANGJIT_STRING_H
