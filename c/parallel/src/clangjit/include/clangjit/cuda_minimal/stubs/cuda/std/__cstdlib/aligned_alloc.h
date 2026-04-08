// ClangJIT override of cuda/std/__cstdlib/aligned_alloc.h
//
// Problem: clangjit compiles with _CCCL_ENABLE_FREESTANDING=1 in both device
// and host passes (the whole environment is freestanding). This causes
// _CCCL_HOSTED() = 0, so __aligned_alloc_host is never defined. But
// kernel_transform.cuh's include chain (copy.h -> cstdlib -> aligned_alloc.h)
// still references ::cuda::std::__aligned_alloc_host in the NV_IS_HOST branch
// of aligned_alloc(), causing a host compilation error.
//
// Fix: always define __aligned_alloc_host unconditionally (not gated on
// _CCCL_HOSTED). The NV_IF_ELSE_TARGET macro uses Clang's "if target"
// extension which discards the inactive branch, so this is never actually
// called in device code, and the host-side CUB dispatch never calls
// aligned_alloc either — it just needs to compile.

#ifndef _CUDA_STD___CSTDLIB_ALIGNED_ALLOC_H
#define _CUDA_STD___CSTDLIB_ALIGNED_ALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()
extern "C" _CCCL_DEVICE void* __cuda_syscall_aligned_malloc(size_t, size_t);
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Define __aligned_alloc_host unconditionally (not gated on _CCCL_HOSTED()).
// In the clangjit freestanding environment the host stdlib is unavailable as
// system headers, so we use __builtin_malloc (a compiler intrinsic requiring
// no headers). This path is never actually reached at runtime.
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST void*
__aligned_alloc_host([[maybe_unused]] size_t __nbytes, [[maybe_unused]] size_t) noexcept
{
  return __builtin_malloc(__nbytes);
}

[[nodiscard]] _CCCL_API inline void* aligned_alloc(size_t __nbytes, size_t __align) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::cuda::std::__aligned_alloc_host(__nbytes, __align);),
                    (return ::__cuda_syscall_aligned_malloc(__nbytes, __align);))
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CSTDLIB_ALIGNED_ALLOC_H
