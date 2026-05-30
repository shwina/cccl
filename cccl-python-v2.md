# cuda.compute on the v2 (HostJIT) Backend

Follow-up to the `cccl-c-v2` PR. Wires Python (`cuda.compute`) to the v2 C
library and switches the operator pipeline from NVRTC-emitted LTO-IR to LLVM
bitcode end-to-end.

## Why

LTO-IR user ops, even after `nvJitLink -lto`, do **not** get inlined into the
CUB PTX kernel that references them. Every loop iteration pays a real `CALL`.
Routing the same op through LLVM bitcode lets the IR linker merge it into the
CUB module *before* PTX generation, so the optimizer inlines it through.

On a `DeviceReduceKernel` SASS comparison with an identical user reduce op:

| Metric          | LTO-IR | LLVM bitcode |
|-----------------|-------:|-------------:|
| `CALL` insns    |    170 |            0 |
| Spills (LDL+STL)|    376 |            4 |
| Registers/thread|    122 |           35 |
| Cubin size      | 180 KB |        54 KB |

## Architecture

### C ABI (`c/parallel.v2/include/cccl/c/types.h`)

- `cccl_op_code_type` gains `CCCL_OP_LLVM_IR = 2` alongside the existing
  `LTOIR` and `CPP_SOURCE`. The numeric default stays `LTOIR = 0` so zero-init
  C callers keep working unchanged.
- `cccl_op_t` gets a new `cccl_op_code_type* extra_code_types` field, parallel
  to `extra_ltoirs`. `NULL` means "treat every extra as LTO-IR" — preserves
  prior behavior. Callers that mix formats (e.g. an iterator extras blob that
  is `CPP_SOURCE` while siblings are `LLVM_IR`) declare the type per entry
  instead of relying on magic-byte sniffing.

### HostJIT bitcode collector (`c/parallel.v2/src/hostjit/codegen/bitcode.cpp`)

- `is_bitcode_op` accepts both `CCCL_OP_LTOIR` and `CCCL_OP_LLVM_IR`.
- Extras are dispatched per-entry on the caller-declared `extra_code_types[i]`
  (CPP source → compile via Clang, anything else → raw bitcode path).
- Both `add_raw_bitcode` and `compile_and_add` content-hash-dedup their input
  via `std::hash<std::string_view>`. Without this, a `ZipIterator(d_a, d_b)`
  where both children are `PointerIterator<int>` would push two identical
  source blobs into the linker and trip "symbol multiply defined".

### Numba → LLVM bitcode (`python/cuda_cccl/cuda/compute/_jit.py`)

`_compile_op_to_llvm_bitcode(wrapped_op, sig)` goes one layer below Numba's
public `cuda.compile` (PTX/LTO-IR only) into `_compile_pyfunc_with_fixup(...,
abi="c", lto=False)`, pulls the LLVM text off the code library, splits on
`; ModuleID` markers (the library is multi-module), parses with `llvmlite`,
links the parts together, and returns `as_bitcode()` bytes.

One subtlety worth flagging: **strip Numba's `target datalayout` line before
parsing**. llvmlite ships an old NVVM layout (`e-p:64:64:64-...`) which
disagrees with the modern CUDA layout (`e-p6:32:32-...`) that the hostjit's
Clang uses. Linking modules with mismatched layouts emits an LLVM warning and
can miscompile pointer-heavy code. Removing the line falls back to the target
triple's canonical layout, which agrees with Clang.

### Python op surface

- `DeviceCode(bytes_, kind)` (`_device_code.py`) is a small frozen dataclass
  that pairs an op's bytes with its format tag (`"ltoir" | "llvm_ir" |
  "cpp_source"`). The bindings duck-type on `(bytes_, kind)` attributes; the
  class is never imported in Cython, which keeps cycle-prone imports out of
  the build.
- `Op` (`_bindings_impl.pyx`) accepts either raw `bytes` (treated as LTO-IR
  for backward compat) or a `DeviceCode`. The Op stores a parallel
  `extra_code_types_arr` that's plumbed into the C struct.
- `RawOp` (`op.py`) accepts the same — bytes for legacy, `DeviceCode` for
  anything else. A new `llvm_stateless.py` example documents the recommended
  path.
- Iterators (`_pointer`, `_zip`, `_transform`, …) compose extras as
  `[child.code, *child.extra_code]`, where `code` / `extra_code` are
  `DeviceCode` lists. No iterator handles bytes directly anymore — the format
  tag travels with the blob through arbitrary nesting depths.
- `_cpp_compile.compile_cpp_op_code(source)` returns a `DeviceCode`. On v2 it
  hands the raw source through with `kind="cpp_source"` (hostjit's Clang
  compiles it inline); on v1 it pre-compiles to LTO-IR via NVRTC.

## Build wiring

- `CCCL_PYTHON_USE_V2=ON` (off by default) makes `python/cuda_cccl/CMakeLists`
  link against `cccl.c.parallel.v2` instead of v1.
- CMake writes `_build_info.py` into the wheel with `USING_V2 = True/False`.
  Python reads it at import time to pick the bitcode-vs-LTO-IR branch in
  `_jit.py` and `_cpp_compile.py`.
- Three `.pxi` pairs (`_bindings_op_code_type_v{1,2}`,
  `_bindings_segmented_reduce_backend_v{1,2}`,
  `_bindings_binary_search_backend_v{1,2}`) are `configure_file`'d into the
  build dir by stem. They bridge backend ABI differences in: the enum (v1
  lacks `LLVM_IR`), `cccl_device_segmented_reduce`'s signature (v1 takes
  `max_segment_size`, v2 doesn't), and `cccl_device_binary_search_build_result_t`'s
  layout (v1 nests a transform result, v2 flattens).
- `__init__.py` calls `_configure_hostjit_paths()` early to point hostjit at
  the wheel-bundled Clang and `cuda_minimal` headers via env vars — only
  when bundled headers are present (skip on editable installs).

## CI

- `cccl_c_parallel_v2` matrix project (already present from #8985) covers the
  C library.
- New `python_v2` matrix project runs `test_py_par` against a wheel built with
  `CCCL_PYTHON_USE_V2=ON`. Linux/CUDA 13 only for now — `libnvfatbin` isn't on
  Windows containers, and v2 needs CUDA ≥ 12.4.
- `ci/build_cuda_cccl_wheel.sh` installs `libnvjitlink-devel` /
  `libnvfatbin-devel` (matching the in-container CTK) and pins
  `cmake>=3.27` for `FindCUDAToolkit`'s `CUDA::nvfatbin` target when the v2
  flag is set.
- `ci/test_cuda_compute_python.sh` runs pytest with `-x` on v2 — the suite is
  still stabilizing and one early failure beats scrolling through hundreds of
  passes.

## Out of scope (deferred)

- Removing `CCCL_OP_LTOIR` as the default. Keeping zero-init compat for C
  callers is more valuable than switching the default after a bake period.
- Removing the LTO-IR escape hatch in the linker. Callers with pre-built
  `nvcc -dlto` artifacts still flow through `nvJitLinkAddData(LTOIR, ...)`.
- Renaming `RawOp.ltoir=` to something format-neutral. Kept for compatibility.
