#define CCCL_C_EXPERIMENTAL 1

#include <chrono>
#include <cccl/c/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << "\n";                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* str = nullptr;                                               \
      cuGetErrorString(err, &str);                                             \
      std::cerr << "CU error at " << __FILE__ << ":" << __LINE__ << ": "       \
                << (str ? str : "unknown") << "\n";                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  // Initialize CUDA driver API (needed by cccl_device_reduce_build)
  CU_CHECK(cuInit(0));

  // Get device compute capability
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::cout << "CCCL C API Example: DeviceReduce Sum\n";
  std::cout << "SM version: " << prop.major << "." << prop.minor << "\n\n";

  // Prepare test data
  const uint64_t N = 1024;
  std::vector<int> h_input(N);
  std::iota(h_input.begin(), h_input.end(), 1);
  const int expected_sum = N * (N + 1) / 2;

  // Allocate device memory
  int* d_input = nullptr;
  int* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Set up input iterator (pointer)
  cccl_type_info int_type{};
  int_type.size = sizeof(int);
  int_type.alignment = alignof(int);
  int_type.type = CCCL_INT32;

  cccl_iterator_t input_it{};
  input_it.size = sizeof(int);
  input_it.alignment = alignof(int);
  input_it.type = CCCL_POINTER;
  input_it.value_type = int_type;
  input_it.state = d_input;

  // Set up output iterator (pointer)
  cccl_iterator_t output_it{};
  output_it.size = sizeof(int);
  output_it.alignment = alignof(int);
  output_it.type = CCCL_POINTER;
  output_it.value_type = int_type;
  output_it.state = d_output;

  // Set up the reduction operator (well-known plus)
  cccl_op_t op{};
  op.type = CCCL_PLUS;

  // Set up initial value
  int init_value = 0;
  cccl_value_t init{};
  init.type = int_type;
  init.state = &init_value;

  // Build the reduction algorithm
  std::cout << "Building reduce algorithm...\n";

  cccl_device_reduce_build_result_t build{};
  std::chrono::high_resolution_clock::time_point build_start =
      std::chrono::high_resolution_clock::now();
  CU_CHECK(cccl_device_reduce_build(&build, input_it, output_it, op, init,
                                    CCCL_RUN_TO_RUN, prop.major, prop.minor,
                                    nullptr, nullptr, nullptr,
                                    nullptr, nullptr));
  std::chrono::high_resolution_clock::time_point build_end =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> build_time =
      build_end - build_start;
  std::cout << "Build time (ms): " << build_time.count() << "\n";

  std::cout << "Build successful!\n\n";

  // Query temp storage size
  size_t temp_storage_bytes = 0;
  CU_CHECK(cccl_device_reduce(build, nullptr, &temp_storage_bytes, input_it,
                              output_it, N, op, init, nullptr));

  // Allocate temp storage
  void* d_temp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_temp, temp_storage_bytes));

  // Run reduction
  std::cout << "Computing sum of " << N << " integers...\n";
  CU_CHECK(cccl_device_reduce(build, d_temp, &temp_storage_bytes, input_it,
                              output_it, N, op, init, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back
  int result = 0;
  CUDA_CHECK(
      cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));

  // Verify
  std::cout << "Result: " << result << " (expected: " << expected_sum << ")\n";
  bool success = (result == expected_sum);
  if (success) {
    std::cout << "Results verified successfully!\n";
  } else {
    std::cerr << "Mismatch!\n";
  }

  // Cleanup
  CU_CHECK(cccl_device_reduce_cleanup(&build));
  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));

  return success ? 0 : 1;
}
