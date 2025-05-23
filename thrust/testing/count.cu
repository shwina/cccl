#include <thrust/count.h>
#include <thrust/iterator/retag.h>

#include <unittest/unittest.h>

template <class Vector>
void TestCountSimple()
{
  Vector data{1, 1, 0, 0, 1};

  ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 0), 2);
  ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 1), 3);
  ASSERT_EQUAL(thrust::count(data.begin(), data.end(), 2), 0);
}
DECLARE_VECTOR_UNITTEST(TestCountSimple);

template <typename T>
void TestCount(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  size_t cpu_result = thrust::count(h_data.begin(), h_data.end(), T(5));
  size_t gpu_result = thrust::count(d_data.begin(), d_data.end(), T(5));

  ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestCount);

template <typename T>
struct greater_than_five
{
  _CCCL_HOST_DEVICE bool operator()(const T& x) const
  {
    return x > 5;
  }
};

template <class Vector>
void TestCountIfSimple()
{
  using T = typename Vector::value_type;

  Vector data{1, 6, 1, 9, 2};

  ASSERT_EQUAL(thrust::count_if(data.begin(), data.end(), greater_than_five<T>()), 2);
}
DECLARE_VECTOR_UNITTEST(TestCountIfSimple);

template <typename T>
void TestCountIf(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data = h_data;

  size_t cpu_result = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
  size_t gpu_result = thrust::count_if(d_data.begin(), d_data.end(), greater_than_five<T>());

  ASSERT_EQUAL(cpu_result, gpu_result);
}
DECLARE_VARIABLE_UNITTEST(TestCountIf);

template <typename Vector>
void TestCountFromConstIteratorSimple()
{
  Vector data{1, 1, 0, 0, 1};

  ASSERT_EQUAL(thrust::count(data.cbegin(), data.cend(), 0), 2);
  ASSERT_EQUAL(thrust::count(data.cbegin(), data.cend(), 1), 3);
  ASSERT_EQUAL(thrust::count(data.cbegin(), data.cend(), 2), 0);
}
DECLARE_VECTOR_UNITTEST(TestCountFromConstIteratorSimple);

template <typename InputIterator, typename EqualityComparable>
int count(my_system& system, InputIterator, InputIterator, EqualityComparable x)
{
  system.validate_dispatch();
  return x;
}

void TestCountDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::count(sys, vec.begin(), vec.end(), 13);

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestCountDispatchExplicit);

template <typename InputIterator, typename EqualityComparable>
int count(my_tag, InputIterator /*first*/, InputIterator, EqualityComparable x)
{
  return x;
}

void TestCountDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  auto result = thrust::count(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 13);

  ASSERT_EQUAL(13, result);
}
DECLARE_UNITTEST(TestCountDispatchImplicit);

void TestCountWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(1);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  long long result = thrust::count(thrust::device, begin, end, (1ll << magnitude) - 17);

  ASSERT_EQUAL(result, 1);
}

void TestCountWithBigIndexes()
{
  TestCountWithBigIndexesHelper(30);
#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
  TestCountWithBigIndexesHelper(31);
  TestCountWithBigIndexesHelper(32);
  TestCountWithBigIndexesHelper(33);
#endif
}
DECLARE_UNITTEST(TestCountWithBigIndexes);
