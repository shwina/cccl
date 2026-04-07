# Build & Test: CCCL C Parallel Library

## CPM source cache

CPM (used to fetch LLVM) caches sources in the build dir by default. To persist
the cache across clean builds, set `CPM_SOURCE_CACHE`:

```bash
export CPM_SOURCE_CACHE=~/.cache/CPM
```

Or pass `-DCPM_SOURCE_CACHE=~/.cache/CPM` to cmake. Without this, deleting the
build directory forces a full LLVM re-clone.

## Configure (standalone, from repo root)

```bash
cmake -S c/parallel -B build_cccl_c_parallel \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86-real
```

Add `-DCCCL_C_Parallel_ENABLE_TESTING=ON` to also build tests.

**Notes:**
- Use pip-installed cmake (not system cmake). The system cmake (3.28) does not know
  about `CUDA::nvfatbin_static` which clangjit requires.
- `c/parallel/CMakeLists.txt` contains a standalone bootstrap block that auto-includes
  the CCCL cmake helpers (CCCLConfigureTarget, CCCLGetDependencies, etc.) and creates
  a stub `cccl.compiler_interface` interface target.
- The build dir is `build_cccl_c_parallel/` (relative to repo root).

## Build

```bash
# Library only
cmake --build build_cccl_c_parallel --target cccl.c.parallel -j$(nproc)

# Everything (library + examples + tests if enabled)
cmake --build build_cccl_c_parallel -j$(nproc)
```

Output:
- Library: `build_cccl_c_parallel/lib/libcccl.c.parallel.so`
- Executables: `build_cccl_c_parallel/bin/`

## Run Examples (clangjit-based, no GPU data needed at build time)

Examples land in `build_cccl_c_parallel/src/clangjit/` (not `bin/`) because the
clangjit CMakeLists.txt doesn't route them through `cccl_configure_target`.

```bash
./build_cccl_c_parallel/src/clangjit/example_cub_reduce
./build_cccl_c_parallel/src/clangjit/example_cub_reduce_deterministic
./build_cccl_c_parallel/src/clangjit/example_cub_reduce_custom_op
./build_cccl_c_parallel/src/clangjit/example_cub_adjacent_difference

# Run all examples
for f in build_cccl_c_parallel/src/clangjit/example_*; do echo "=== $f ==="; $f; done
```

## Run Tests

```bash
# All tests via ctest
ctest --test-dir build_cccl_c_parallel

# Verbose, filtered to one algorithm
ctest --test-dir build_cccl_c_parallel -V -R reduce

# Run a specific test binary directly
./build_cccl_c_parallel/bin/cccl.c.parallel.test.reduce
```

Test binary naming: `cccl.c.parallel.test.<algo>` (e.g. `test.reduce`, `test.scan`, `test.for`).

**Migration status (freestanding branch):**
- `test_reduce` / `example_cub_*` → exercise new **clangjit** path
- All other tests → still on old NVRTC path (pending migration)

## Architecture: clangjit (freestanding branch)

The `clangjit` subsystem (`c/parallel/src/clangjit/`) embeds Clang/LLVM to JIT-compile
host+device CUDA code, replacing NVRTC (device-only). This lets CUB's host dispatch
logic run unchanged.

**Key new API — `CubCall` DSL** (`src/clangjit/codegen/cub_call.cpp`):
```cpp
#include <clangjit/codegen/cub_call.hpp>
using namespace clangjit::codegen;

auto result = CubCall::from("cub/device/device_reduce.cuh")
                .run("cub::DeviceReduce::Reduce")
                .name("cccl_jit_reduce")
                .with(temp_storage, temp_bytes, in(input_it), out(output_it),
                      num_items, op, init)
                .compile(cc_major, cc_minor, clang_path, build_config,
                         ctk_root, cccl_include_path);

// result.compiler  — clangjit::JITCompiler* (caller owns, delete to unload .so)
// result.fn_ptr    — void* function pointer, cast to int(*)(void*, size_t*, ...)
// result.cubin     — std::vector<char> for SASS inspection
```

**`.with()` argument tags:**
| Tag | Generated param | CUB arg |
|-----|----------------|---------|
| `temp_storage` | `void* d_temp_storage` | `d_temp_storage` |
| `temp_bytes` | `size_t* temp_storage_bytes` | `*temp_storage_bytes` |
| `num_items` | `unsigned long long num_items` | `(unsigned long long)num_items` |
| `in(it)` | `void* d_in_N` | iterator struct initialized from state bytes |
| `out(it)` | `void* d_out_N` | iterator struct initialized from state bytes |
| `cccl_op_t` | `void* op_N_state` | functor struct |
| `cccl_value_t` | `void* val_N_ptr` | value via `__builtin_memcpy` |

**Build result struct pattern (migrated algorithms):**
```c
// Old: CUlibrary library; CUkernel kernel; void* runtime_policy;
// New:
void* jit_compiler;   // clangjit::JITCompiler* — owns the loaded .so
void* algo_fn;        // int(*)(void*, size_t*, ...) — the compiled function
void* cubin;          // raw cubin bytes (for inspection/serialization)
size_t cubin_size;
```

**Cleanup pattern:**
```cpp
delete static_cast<clangjit::JITCompiler*>(build->jit_compiler);
delete[] static_cast<char*>(build->cubin);
```

**Compilation pipeline:** Clang (nvptx64 triple) → PTX → LLD → `.so` → `dlopen` → fn ptr.
Custom ops/iterators in LTOIR/LLVM IR are linked via `libdevice.10.bc`.

## Migration plan (freestanding branch)

Algorithms to migrate from NVRTC → clangjit, in order:

| Tier | Files | Notes |
|------|-------|-------|
| 1 (simple) | `segmented_reduce`, `for`, `binary_search`, `transform` | Single CUB call |
| 2 (variants) | `scan`, `unique_by_key` | Multiple fn pointers in build result |
| 3 (complex) | `merge_sort`, `radix_sort`, `histogram`, `segmented_sort`, `three_way_partition` | Multi-type or complex dispatch |

Reference implementation: `c/parallel/src/reduce.cu` (already migrated).
