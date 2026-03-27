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
  const char *cuda_source = R"(
#include <cuda_runtime.h>
#include <cub/device/device_adjacent_difference.cuh>
#include <cuda/std/functional>

#ifdef _WIN32
#define CLANGJIT_EXPORT __declspec(dllexport)
#else
#define CLANGJIT_EXPORT
#endif

extern "C" CLANGJIT_EXPORT int adjacentDifference(
    const int* d_input, int* d_output, int num_items) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Query temp storage
    cudaError_t err = cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes, d_input, d_output, num_items,
        cuda::std::minus<int>{});
    if (err != cudaSuccess) return -1;

    // Allocate temp storage
    err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess) return -2;

    // Run adjacent difference
    err = cub::DeviceAdjacentDifference::SubtractLeftCopy(
        d_temp_storage, temp_storage_bytes, d_input, d_output, num_items,
        cuda::std::minus<int>{});
    if (err != cudaSuccess) {
        cudaFree(d_temp_storage);
        return -3;
    }

    err = cudaDeviceSynchronize();
    cudaFree(d_temp_storage);
    return (err == cudaSuccess) ? 0 : -4;
}
)";

  std::cout << "ClangJIT Example: CUB DeviceAdjacentDifference::SubtractLeft\n\n";

  clangjit::CompilerConfig config = clangjit::detectDefaultConfig();
  config.verbose = true;
  config.trace_includes = true;

  std::cout << "CUDA toolkit: " << config.cuda_toolkit_path << "\n";
  std::cout << "SM version: " << config.sm_version << "\n\n";

  clangjit::JITCompiler compiler(config);

  std::cout << "Compiling CUDA code with CUB...\n";
  if (!compiler.compile(cuda_source)) {
    std::cerr << "Compilation failed:\n" << compiler.getLastError() << "\n";
    return 1;
  }
  std::cout << "Compilation successful!\n\n";

  auto adjacentDiff =
      compiler.getFunction<int (*)(const int *, int *, int)>("adjacentDifference");
  if (!adjacentDiff) {
    std::cerr << "Failed to get function: " << compiler.getLastError() << "\n";
    return 1;
  }

  // Prepare test data: [1, 2, 3, 4, 5, ...]
  const int N = 8;
  std::vector<int> h_input(N);
  std::iota(h_input.begin(), h_input.end(), 1);

  // Expected: SubtractLeft produces [d[0], d[1]-d[0], d[2]-d[1], ...]
  // For input [1,2,3,4,5,6,7,8]: output [1,1,1,1,1,1,1,1]
  std::vector<int> expected(N);
  expected[0] = h_input[0];
  for (int i = 1; i < N; i++) {
    expected[i] = h_input[i] - h_input[i - 1];
  }

  int *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(int),
                        cudaMemcpyHostToDevice));

  std::cout << "Computing adjacent differences of " << N << " integers...\n";
  int status = adjacentDiff(d_input, d_output, N);
  if (status != 0) {
    std::cerr << "adjacentDifference failed with status: " << status << "\n";
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 1;
  }

  std::vector<int> h_output(N);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(int),
                        cudaMemcpyDeviceToHost));

  std::cout << "Input:    [";
  for (int i = 0; i < N; i++) std::cout << (i ? ", " : "") << h_input[i];
  std::cout << "]\n";

  std::cout << "Output:   [";
  for (int i = 0; i < N; i++) std::cout << (i ? ", " : "") << h_output[i];
  std::cout << "]\n";

  std::cout << "Expected: [";
  for (int i = 0; i < N; i++) std::cout << (i ? ", " : "") << expected[i];
  std::cout << "]\n";

  bool success = (h_output == expected);
  if (success) {
    std::cout << "Results verified successfully!\n";
  } else {
    std::cerr << "Mismatch!\n";
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  return success ? 0 : 1;
}
