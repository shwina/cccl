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
#include <thrust/iterator/iterator_traits.h>
#include <thrust/scan.h>
#include <thrust/system/detail/adl/scan.h>
#include <thrust/system/detail/adl/scan_by_key.h>
#include <thrust/system/detail/generic/scan.h>
#include <thrust/system/detail/generic/scan_by_key.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan");
  using thrust::system::detail::generic::inclusive_scan;
  return inclusive_scan(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result);
} // end inclusive_scan()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan");
  using thrust::system::detail::generic::inclusive_scan;
  return inclusive_scan(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, binary_op);
} // end inclusive_scan()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T, typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  T init,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan");
  using thrust::system::detail::generic::inclusive_scan;
  return inclusive_scan(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, init, binary_op);
} // end inclusive_scan()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan");
  using thrust::system::detail::generic::exclusive_scan;
  return exclusive_scan(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result);
} // end exclusive_scan()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  T init)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan");
  using thrust::system::detail::generic::exclusive_scan;
  return exclusive_scan(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, init);
} // end exclusive_scan()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T, typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  T init,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan");
  using thrust::system::detail::generic::exclusive_scan;
  return exclusive_scan(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, result, init, binary_op);
} // end exclusive_scan()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan_by_key");
  using thrust::system::detail::generic::inclusive_scan_by_key;
  return inclusive_scan_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, result);
} // end inclusive_scan_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan_by_key");
  using thrust::system::detail::generic::inclusive_scan_by_key;
  return inclusive_scan_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, result, binary_pred);
} // end inclusive_scan_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator inclusive_scan_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan_by_key");
  using thrust::system::detail::generic::inclusive_scan_by_key;
  return inclusive_scan_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    first1,
    last1,
    first2,
    result,
    binary_pred,
    binary_op);
} // end inclusive_scan_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, result);
} // end exclusive_scan_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, result, init);
} // end exclusive_scan_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, result, init, binary_pred);
} // end exclusive_scan_by_key()

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate,
          typename AssociativeOperator>
_CCCL_HOST_DEVICE OutputIterator exclusive_scan_by_key(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::exclusive_scan_by_key;
  return exclusive_scan_by_key(
    thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
    first1,
    last1,
    first2,
    result,
    init,
    binary_pred,
    binary_op);
} // end exclusive_scan_by_key()

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::inclusive_scan(select_system(system1, system2), first, last, result);
} // end inclusive_scan()

template <typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::inclusive_scan(select_system(system1, system2), first, last, result, binary_op);
} // end inclusive_scan()

template <typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
OutputIterator
inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init, BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::inclusive_scan(select_system(system1, system2), first, last, result, init, binary_op);
} // end inclusive_scan()

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(InputIterator first, InputIterator last, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::exclusive_scan(select_system(system1, system2), first, last, result);
} // end exclusive_scan()

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::exclusive_scan(select_system(system1, system2), first, last, result, init);
} // end exclusive_scan()

template <typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
OutputIterator
exclusive_scan(InputIterator first, InputIterator last, OutputIterator result, T init, BinaryFunction binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator>::type;
  using System2 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;

  return thrust::exclusive_scan(select_system(system1, system2), first, last, result, init, binary_op);
} // end exclusive_scan()

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
inclusive_scan_by_key(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::inclusive_scan_by_key(select_system(system1, system2, system3), first1, last1, first2, result);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryPredicate>
OutputIterator inclusive_scan_by_key(
  InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::inclusive_scan_by_key(
    select_system(system1, system2, system3), first1, last1, first2, result, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename BinaryPredicate,
          typename AssociativeOperator>
OutputIterator inclusive_scan_by_key(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::inclusive_scan_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::inclusive_scan_by_key(
    select_system(system1, system2, system3), first1, last1, first2, result, binary_pred, binary_op);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
OutputIterator
exclusive_scan_by_key(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::exclusive_scan_by_key(select_system(system1, system2, system3), first1, last1, first2, result);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
OutputIterator
exclusive_scan_by_key(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::exclusive_scan_by_key(select_system(system1, system2, system3), first1, last1, first2, result, init);
}

template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T, typename BinaryPredicate>
OutputIterator exclusive_scan_by_key(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::exclusive_scan_by_key(
    select_system(system1, system2, system3), first1, last1, first2, result, init, binary_pred);
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename T,
          typename BinaryPredicate,
          typename AssociativeOperator>
OutputIterator exclusive_scan_by_key(
  InputIterator1 first1,
  InputIterator1 last1,
  InputIterator2 first2,
  OutputIterator result,
  T init,
  BinaryPredicate binary_pred,
  AssociativeOperator binary_op)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::exclusive_scan_by_key");
  using thrust::system::detail::generic::select_system;

  using System1 = typename thrust::iterator_system<InputIterator1>::type;
  using System2 = typename thrust::iterator_system<InputIterator2>::type;
  using System3 = typename thrust::iterator_system<OutputIterator>::type;

  System1 system1;
  System2 system2;
  System3 system3;

  return thrust::exclusive_scan_by_key(
    select_system(system1, system2, system3), first1, last1, first2, result, init, binary_pred, binary_op);
}

THRUST_NAMESPACE_END
