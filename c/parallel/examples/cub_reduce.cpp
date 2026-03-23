#include "clangjit/config.hpp"
#include "clangjit/jit_compiler.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << "\n";                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  // Define CUDA code that uses CUB DeviceReduce::Sum
  const char *cuda_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#ifdef _WIN32
#define CLANGJIT_EXPORT __declspec(dllexport)
#else
#define CLANGJIT_EXPORT
#endif

extern "C" CLANGJIT_EXPORT int computeSum(int* d_input, int num_items, int* result) {
    int* d_output = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate output
    cudaError_t err = cudaMalloc(&d_output, sizeof(int));
    if (err != cudaSuccess) return -1;

    // Run sum reduction
    err = cub::DeviceReduce::Sum(d_input, d_output, num_items);
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -2;
    }

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -3;
    }

    // Copy result back
    err = cudaMemcpy(result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_output);
        return -4;
    }

    // Cleanup
    cudaFree(d_output);

    return 0;
}
)";

  std::cout << "ClangJIT Example: CUB DeviceReduce::Sum\n\n";

  // Create JIT compiler with default configuration
  clangjit::CompilerConfig config = clangjit::detectDefaultConfig();
  config.verbose = true;
  config.trace_includes = true;

  std::cout << "CUDA toolkit: " << config.cuda_toolkit_path << "\n";
  std::cout << "SM version: " << config.sm_version << "\n\n";

  clangjit::JITCompiler compiler(config);

  // Compile the CUDA source
  std::cout << "Compiling CUDA code with CUB...\n";
  if (!compiler.compile(cuda_source)) {
    std::cerr << "Compilation failed:\n" << compiler.getLastError() << "\n";
    return 1;
  }
  std::cout << "Compilation successful!\n\n";

  // Get the function pointer
  auto computeSum =
      compiler.getFunction<int (*)(int *, int, int *)>("computeSum");
  if (!computeSum) {
    std::cerr << "Failed to get function: " << compiler.getLastError() << "\n";
    return 1;
  }

  // Prepare test data
  const int N = 1024;
  std::vector<int> h_input(N);

  // Initialize with values 1 to N
  std::iota(h_input.begin(), h_input.end(), 1);

  // Expected sum: N*(N+1)/2
  int expected_sum = N * (N + 1) / 2;

  // Allocate device memory
  int *d_input;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Call the JIT-compiled function
  std::cout << "Computing sum of " << N << " integers using CUB...\n";
  int result = 0;
  int status = computeSum(d_input, N, &result);

  if (status != 0) {
    std::cerr << "computeSum failed with status: " << status << "\n";
    CUDA_CHECK(cudaFree(d_input));
    return 1;
  }

  // Verify result
  std::cout << "Result: " << result << " (expected: " << expected_sum << ")\n";

  bool success = (result == expected_sum);
  if (success) {
    std::cout << "Results verified successfully!\n";
  } else {
    std::cerr << "Mismatch! Got " << result << " but expected " << expected_sum
              << "\n";
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_input));

  return success ? 0 : 1;
}
