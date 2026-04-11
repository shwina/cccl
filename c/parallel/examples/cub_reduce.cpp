#define CCCL_C_EXPERIMENTAL 1

#include <chrono>
#include <cccl/c/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>

#include <algorithm>
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
  CU_CHECK(cuInit(0));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  std::cout << "CCCL C API Example: DeviceReduce (Sum + Max)\n";
  std::cout << "SM version: " << prop.major << "." << prop.minor << "\n\n";

  // PCH cache status
  int sm = prop.major * 10 + prop.minor;
  auto pch_dir    = std::filesystem::temp_directory_path() / "clangjit_pch";
  auto device_pch = pch_dir / ("device_sm" + std::to_string(sm) + ".pch");
  auto host_pch   = pch_dir / ("host_sm" + std::to_string(sm) + ".pch");
  bool pch_cached = std::filesystem::exists(device_pch) && std::filesystem::exists(host_pch);
  std::cout << "PCH cache: " << pch_dir.string()
            << (pch_cached ? " (hit)" : " (miss, will generate)") << "\n\n";

  // Prepare test data
  const uint64_t N = 1024;
  std::vector<int> h_input(N);
  std::iota(h_input.begin(), h_input.end(), 1);

  // Allocate device memory
  int* d_input = nullptr;
  int* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Common type / iterator setup
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

  cccl_iterator_t output_it{};
  output_it.size = sizeof(int);
  output_it.alignment = alignof(int);
  output_it.type = CCCL_POINTER;
  output_it.value_type = int_type;
  output_it.state = d_output;

  cccl_build_config build_config{};
  build_config.enable_pch = 1;

  bool all_passed = true;

  // ── Sum reduction ────────────────────────────────────────────────
  {
    std::cout << "=== Sum reduction ===\n";

    cccl_op_t op{};
    op.type = CCCL_PLUS;

    int init_value = 0;
    cccl_value_t init{};
    init.type = int_type;
    init.state = &init_value;

    cccl_device_reduce_build_result_t build{};
    auto t0 = std::chrono::high_resolution_clock::now();
    CU_CHECK(cccl_device_reduce_build_ex(&build, input_it, output_it, op, init,
                                         CCCL_RUN_TO_RUN, prop.major, prop.minor,
                                         nullptr, nullptr, nullptr,
                                         nullptr, nullptr, &build_config));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Build time (ms): "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << "\n";

    size_t temp_bytes = 0;
    CU_CHECK(cccl_device_reduce(build, nullptr, &temp_bytes, input_it,
                                output_it, N, op, init, nullptr));
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    CU_CHECK(cccl_device_reduce(build, d_temp, &temp_bytes, input_it,
                                output_it, N, op, init, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int result = 0;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    int expected = N * (N + 1) / 2;
    std::cout << "Result: " << result << " (expected: " << expected << ")\n";
    if (result != expected) { std::cerr << "MISMATCH!\n"; all_passed = false; }

    CU_CHECK(cccl_device_reduce_cleanup(&build));
    CUDA_CHECK(cudaFree(d_temp));
  }

  std::cout << "\n";

  // ── Max reduction ────────────────────────────────────────────────
  {
    std::cout << "=== Max reduction ===\n";

    cccl_op_t op{};
    op.type = CCCL_MAXIMUM;

    int init_value = 0;
    cccl_value_t init{};
    init.type = int_type;
    init.state = &init_value;

    cccl_device_reduce_build_result_t build{};
    auto t0 = std::chrono::high_resolution_clock::now();
    CU_CHECK(cccl_device_reduce_build_ex(&build, input_it, output_it, op, init,
                                         CCCL_RUN_TO_RUN, prop.major, prop.minor,
                                         nullptr, nullptr, nullptr,
                                         nullptr, nullptr, &build_config));
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Build time (ms): "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << "\n";

    size_t temp_bytes = 0;
    CU_CHECK(cccl_device_reduce(build, nullptr, &temp_bytes, input_it,
                                output_it, N, op, init, nullptr));
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    CU_CHECK(cccl_device_reduce(build, d_temp, &temp_bytes, input_it,
                                output_it, N, op, init, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int result = 0;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost));
    int expected = *std::max_element(h_input.begin(), h_input.end());
    std::cout << "Result: " << result << " (expected: " << expected << ")\n";
    if (result != expected) { std::cerr << "MISMATCH!\n"; all_passed = false; }

    CU_CHECK(cccl_device_reduce_cleanup(&build));
    CUDA_CHECK(cudaFree(d_temp));
  }

  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_input));

  std::cout << "\n" << (all_passed ? "All tests passed!" : "FAILURES detected!") << "\n";
  return all_passed ? 0 : 1;
}
