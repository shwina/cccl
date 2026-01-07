# Raw LTOIR Layer

The Raw LTOIR layer provides a compiler-agnostic interface for using pre-compiled
LTOIR (Link-Time Optimization Intermediate Representation) with cuda.compute algorithms.

## Overview

The Raw layer sits directly above the C bindings and enables:

1. **Bring Your Own Compiler (BYOC)**: Use LTOIR from any source (nvcc, MLIR, Triton, etc.)
2. **Numba Independence**: Use cuda.compute without requiring Numba for JIT compilation
3. **Performance**: Skip JIT compilation overhead for pre-compiled operations

## Architecture

```
┌─────────────────────────────────────┐
│         User API Layer             │
│  reduce_into(), TransformIterator  │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│       Compiler Layer (Numba)       │
│   compile_op(), compile_iterator() │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│      Raw LTOIR Layer (this)        │  ◀── You can inject here!
│       RawOp, RawIterator           │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│        C Bindings Layer            │
│          Op, Iterator              │
└─────────────────────────────────────┘
```

## Usage

### Type Helpers

The Raw layer provides TypeInfo constants for common types:

```python
from cuda.compute import (
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64,
    bool_, storage
)

# For custom structs, use storage()
my_struct_type = storage(size=16, alignment=8)
```

### RawOp: Pre-compiled Operations

```python
from cuda.compute import RawOp, int32, reduce_into

# Assume you have LTOIR from your compiler
ltoir = my_compiler.compile_add_function()

# Create a RawOp
add_op = RawOp(
    ltoir=ltoir,
    name="my_add",  # Must match the extern "C" function name
    arg_types=(int32, int32),
    return_type=int32,
)

# Use with any algorithm
reduce_into(d_in, d_out, add_op, num_items, h_init)
```

### RawIterator: Pre-compiled Iterators

```python
from cuda.compute import RawIterator, int32

# Create a RawIterator with pre-compiled methods
my_iter = RawIterator(
    state=state_bytes,          # Iterator state as bytes
    state_alignment=8,          # Alignment requirement
    value_type=int32,           # Type of dereferenced values
    advance=("my_advance", advance_ltoir),
    input_dereference=("my_deref", deref_ltoir),
)
```

## LTOIR ABI Contract

All LTOIR functions **must** use `extern "C"` linkage with `void*` parameters.

### Binary Operator (reduce, scan, etc.)

```c
extern "C" void my_add(void* arg0, void* arg1, void* result) {
    int32_t a = *(int32_t*)arg0;
    int32_t b = *(int32_t*)arg1;
    *(int32_t*)result = a + b;
}
```

### Unary Operator (transform)

```c
extern "C" void my_transform(void* input, void* result) {
    int32_t x = *(int32_t*)input;
    *(int32_t*)result = x * 2;
}
```

### Iterator Advance

```c
extern "C" void my_advance(void* state_ptr, uint64_t offset) {
    int32_t** ptr = (int32_t**)state_ptr;
    *ptr = *ptr + offset;
}
```

### Iterator Input Dereference

```c
extern "C" void my_input_deref(void* state_ptr, void* result_ptr) {
    int32_t** ptr = (int32_t**)state_ptr;
    *(int32_t*)result_ptr = **ptr;
}
```

### Iterator Output Dereference

```c
extern "C" void my_output_deref(void* state_ptr, void* value_ptr) {
    int32_t** ptr = (int32_t**)state_ptr;
    **ptr = *(int32_t*)value_ptr;
}
```

## Compiling LTOIR

### Using nvcc

```bash
nvcc -dlto -dc -o my_ops.ltoir my_ops.cu
```

### Using Numba (for testing)

```python
from numba import cuda

@cuda.jit(device=True)
def my_add(a, b):
    return a + b

# Compile to LTOIR
ltoir, _ = cuda.compile(my_add, sig="int32(int32, int32)", output="ltoir")
```

## Integration with Custom Compilers

To integrate a custom compiler:

1. Generate LTOIR with the correct ABI (extern "C", void* parameters)
2. Create `RawOp` or `RawIterator` instances with the LTOIR
3. Pass them to cuda.compute algorithms

The Numba layer in `cuda.compute._numba` serves as a reference implementation.
