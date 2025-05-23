/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename RandomAccessIterator,
          typename BinaryPredicate = ::cuda::std::equal_to<it_value_t<RandomAccessIterator>>,
          typename ValueType       = bool,
          typename IndexType       = it_difference_t<RandomAccessIterator>>
class head_flags
{
public:
  struct head_flag_functor
  {
    BinaryPredicate binary_pred; // this must be the first member for performance reasons
    RandomAccessIterator iter;

    using result_type = ValueType;

    _CCCL_HOST_DEVICE head_flag_functor(RandomAccessIterator iter, BinaryPredicate binary_pred = {})
        : binary_pred(binary_pred)
        , iter(iter)
    {}

    _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result_type operator()(const IndexType i)
    {
      // note that we do not dereference the iterators i <= 0
      // and therefore do not dereference a bad location at the boundary
      if (i <= 0)
      {
        return true;
      }

      return !binary_pred(thrust::raw_reference_cast(iter[i]), thrust::raw_reference_cast(iter[i - 1]));
    }
  };

public:
  using iterator = thrust::transform_iterator<head_flag_functor, thrust::counting_iterator<IndexType>>;

  _CCCL_HOST_DEVICE head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred = {})
      : m_begin(thrust::make_transform_iterator(
          thrust::counting_iterator<IndexType>(0), head_flag_functor(first, binary_pred)))
      , m_count(last - first)
  {}

  _CCCL_HOST_DEVICE iterator begin() const
  {
    return m_begin;
  }

  _CCCL_HOST_DEVICE iterator end() const
  {
    return m_begin + m_count;
  }

  template <typename OtherIndex>
  _CCCL_HOST_DEVICE typename iterator::reference operator[](OtherIndex i)
  {
    return *(begin() + i);
  }

  _CCCL_SYNTHESIZE_SEQUENCE_ACCESS(head_flags, iterator)

private:
  iterator m_begin;
  IndexType m_count;
};

template <typename RandomAccessIterator, typename BinaryPredicate>
_CCCL_HOST_DEVICE head_flags<RandomAccessIterator, BinaryPredicate>
make_head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
  return head_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}

template <typename RandomAccessIterator>
_CCCL_HOST_DEVICE head_flags<RandomAccessIterator> make_head_flags(RandomAccessIterator first, RandomAccessIterator last)
{
  return head_flags<RandomAccessIterator>(first, last);
}

} // namespace detail
THRUST_NAMESPACE_END
