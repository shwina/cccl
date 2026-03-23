// CLANGJIT: Device atomic function wrappers
// These wrap the low-level intrinsics from __clang_cuda_device_functions.h

#ifndef __CLANGJIT_CUDA_DEVICE_ATOMIC_FUNCTIONS_H__
#define __CLANGJIT_CUDA_DEVICE_ATOMIC_FUNCTIONS_H__

// The low-level intrinsics (__iAtomicAdd, __uAtomicAdd, etc.) are provided
// by clang's __clang_cuda_device_functions.h. We just provide the high-level
// wrappers here.

#define __ATOMIC_FUNCTIONS_DECL__ static __inline__ __device__

// atomicAdd
__ATOMIC_FUNCTIONS_DECL__ int atomicAdd(int *address, int val) {
  return __iAtomicAdd(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicAdd(unsigned int *address, unsigned int val) {
  return __uAtomicAdd(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {
  return __ullAtomicAdd(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ float atomicAdd(float *address, float val) {
  return __fAtomicAdd(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ double atomicAdd(double *address, double val) {
  return __dAtomicAdd(address, val);
}

// atomicSub
__ATOMIC_FUNCTIONS_DECL__ int atomicSub(int *address, int val) {
  return __iAtomicAdd(address, -val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicSub(unsigned int *address, unsigned int val) {
  return __uAtomicAdd(address, (unsigned int)-(int)val);
}

// atomicExch
__ATOMIC_FUNCTIONS_DECL__ int atomicExch(int *address, int val) {
  return __iAtomicExch(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicExch(unsigned int *address, unsigned int val) {
  return __uAtomicExch(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {
  return __ullAtomicExch(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ float atomicExch(float *address, float val) {
  return __fAtomicExch(address, val);
}

// atomicMin
__ATOMIC_FUNCTIONS_DECL__ int atomicMin(int *address, int val) {
  return __iAtomicMin(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicMin(unsigned int *address, unsigned int val) {
  return __uAtomicMin(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {
  return __ullAtomicMin(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ long long atomicMin(long long *address, long long val) {
  return __illAtomicMin(address, val);
}

// atomicMax
__ATOMIC_FUNCTIONS_DECL__ int atomicMax(int *address, int val) {
  return __iAtomicMax(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicMax(unsigned int *address, unsigned int val) {
  return __uAtomicMax(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {
  return __ullAtomicMax(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ long long atomicMax(long long *address, long long val) {
  return __illAtomicMax(address, val);
}

// atomicInc / atomicDec
__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicInc(unsigned int *address, unsigned int val) {
  return __uAtomicInc(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicDec(unsigned int *address, unsigned int val) {
  return __uAtomicDec(address, val);
}

// atomicCAS
__ATOMIC_FUNCTIONS_DECL__ int atomicCAS(int *address, int compare, int val) {
  return __iAtomicCAS(address, compare, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicCAS(unsigned int *address, unsigned int compare, unsigned int val) {
  return __uAtomicCAS(address, compare, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {
  return __ullAtomicCAS(address, compare, val);
}

// atomicAnd
__ATOMIC_FUNCTIONS_DECL__ int atomicAnd(int *address, int val) {
  return __iAtomicAnd(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicAnd(unsigned int *address, unsigned int val) {
  return __uAtomicAnd(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {
  return __ullAtomicAnd(address, val);
}

// atomicOr
__ATOMIC_FUNCTIONS_DECL__ int atomicOr(int *address, int val) {
  return __iAtomicOr(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicOr(unsigned int *address, unsigned int val) {
  return __uAtomicOr(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {
  return __ullAtomicOr(address, val);
}

// atomicXor
__ATOMIC_FUNCTIONS_DECL__ int atomicXor(int *address, int val) {
  return __iAtomicXor(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned int atomicXor(unsigned int *address, unsigned int val) {
  return __uAtomicXor(address, val);
}

__ATOMIC_FUNCTIONS_DECL__ unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {
  return __ullAtomicXor(address, val);
}

#undef __ATOMIC_FUNCTIONS_DECL__

#endif // __CLANGJIT_CUDA_DEVICE_ATOMIC_FUNCTIONS_H__
