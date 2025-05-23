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

/*! \file
 *  \brief A pointer to a variable which resides in memory associated with a
 *  system.
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
#include <thrust/detail/reference_forward_declaration.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/adl/assign_value.h>
#include <thrust/system/detail/adl/get_value.h>
#include <thrust/system/detail/adl/iter_swap.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cuda/std/type_traits>

#include <ostream>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename>
struct is_wrapped_reference;
}

/*! \p reference acts as a reference-like wrapper for an object residing in
 *  memory that a \p pointer refers to.
 */
template <typename Element, typename Pointer, typename Derived>
class reference
{
private:
  using derived_type = typename std::conditional<std::is_same<Derived, use_default>::value, reference, Derived>::type;

public:
  using pointer    = Pointer;
  using value_type = ::cuda::std::remove_cvref_t<Element>;

  reference(reference const&) = default;

  reference(reference&&) = default;

  /*! Construct a \p reference from another \p reference whose pointer type is
   *  convertible to \p pointer. After this \p reference is constructed, it
   *  shall refer to the same object as \p other.
   *
   *  \tparam OtherElement The element type of the other \p reference.
   *  \tparam OtherPointer The pointer type of the other \p reference.
   *  \tparam OtherDerived The derived type of the other \p reference.
   *  \param  other        A \p reference to copy from.
   */
  template <typename OtherElement, typename OtherPointer, typename OtherDerived>
  _CCCL_HOST_DEVICE reference(
    reference<OtherElement, OtherPointer, OtherDerived> const& other
    /*! \cond
     */
    ,
    typename std::enable_if<std::is_convertible<typename reference<OtherElement, OtherPointer, OtherDerived>::pointer,
                                                pointer>::value>::type* = nullptr
    /*! \endcond
     */
    )
      : ptr(other.ptr)
  {}

  /*! Construct a \p reference that refers to an object pointed to by the given
   *  \p pointer. After this \p reference is constructed, it shall refer to the
   *  object pointed to by \p ptr.
   *
   *  \param ptr A \p pointer to construct from.
   */
  _CCCL_HOST_DEVICE explicit reference(pointer const& p)
      : ptr(p)
  {}

  /*! Assign the object referred to \p other to the object referred to by
   *  this \p reference.
   *
   *  \param other The other \p reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  _CCCL_HOST_DEVICE const derived_type& operator=(reference const& other) const
  {
    assign_from(&other);
    return derived();
  }

  /*! Assign the object referred to by this \p reference with the object
   *  referred to by another \p reference whose pointer type is convertible to
   *  \p pointer.
   *
   *  \tparam OtherElement The element type of the other \p reference.
   *  \tparam OtherPointer The pointer type of the other \p reference.
   *  \tparam OtherDerived The derived type of the other \p reference.
   *  \param  other        The other \p reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  template <
    typename OtherElement,
    typename OtherPointer,
    typename OtherDerived,
    ::cuda::std::enable_if_t<
      ::cuda::std::is_convertible_v<typename reference<OtherElement, OtherPointer, OtherDerived>::pointer, pointer>,
      int> = 0>
  _CCCL_HOST_DEVICE const derived_type& operator=(reference<OtherElement, OtherPointer, OtherDerived> const& other) const
  {
    assign_from(&other);
    return derived();
  }

  /*! Assign \p rhs to the object referred to by this \p tagged_reference.
   *
   *  \param rhs The \p value_type to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  _CCCL_HOST_DEVICE const derived_type& operator=(value_type const& rhs) const
  {
    assign_from(&rhs);
    return derived();
  }

  /*! Exchanges the value of the object referred to by this \p tagged_reference
   *  with the object referred to by \p other.
   *
   *  \param other The \p tagged_reference to swap with.
   */
  _CCCL_HOST_DEVICE void swap(derived_type& other)
  {
    // Avoid default-constructing a system; instead, just use a null pointer
    // for dispatch. This assumes that `get_value` will not access any system
    // state.
    typename thrust::iterator_system<pointer>::type* system = nullptr;
    swap(system, other);
  }

  _CCCL_HOST_DEVICE pointer operator&() const
  {
    return ptr;
  }

  // This is inherently hazardous, as it discards the strong type information
  // about what system the object is on.
  _CCCL_HOST_DEVICE operator value_type() const
  {
    // Avoid default-constructing a system; instead, just use a null pointer
    // for dispatch. This assumes that `get_value` will not access any system
    // state.
    typename thrust::iterator_system<pointer>::type* system = nullptr;
    return convert_to_value_type(system);
  }

  _CCCL_HOST_DEVICE derived_type& operator++()
  {
    // Sadly, this has to make a copy. The only mechanism we have for
    // modifying the value, which may be in memory inaccessible to this
    // system, is to get a copy of it, modify the copy, and then update it.
    value_type tmp = *this;
    ++tmp;
    *this = tmp;
    return derived();
  }

  _CCCL_HOST_DEVICE value_type operator++(int)
  {
    value_type tmp    = *this;
    value_type result = tmp++;
    *this             = ::cuda::std::move(tmp);
    return result;
  }

  derived_type& operator--()
  {
    // Sadly, this has to make a copy. The only mechanism we have for
    // modifying the value, which may be in memory inaccessible to this
    // system, is to get a copy of it, modify the copy, and then update it.
    value_type tmp = *this;
    --tmp;
    *this = ::cuda::std::move(tmp);
    return derived();
  }

  value_type operator--(int)
  {
    value_type tmp    = *this;
    value_type result = tmp--;
    *this             = ::cuda::std::move(tmp);
    return derived();
  }

  _CCCL_HOST_DEVICE derived_type& operator+=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp += rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator-=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp -= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator*=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp *= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator/=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp /= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator%=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp %= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator<<=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp <<= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator>>=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp >>= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator&=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp &= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator|=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp |= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator^=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp ^= rhs;
    *this = tmp;
    return derived();
  }

private:
  pointer const ptr;

  // `thrust::detail::is_wrapped_reference` is a trait that indicates whether
  // a type is a fancy reference. It detects such types by looking for a
  // nested `wrapped_reference_hint` type.
  struct wrapped_reference_hint
  {};
  template <typename>
  friend struct thrust::detail::is_wrapped_reference;

  template <typename OtherElement, typename OtherPointer, typename OtherDerived>
  friend class reference;

  _CCCL_HOST_DEVICE derived_type& derived()
  {
    return static_cast<derived_type&>(*this);
  }

  _CCCL_HOST_DEVICE const derived_type& derived() const
  {
    return static_cast<const derived_type&>(*this);
  }

  template <typename System>
  _CCCL_HOST_DEVICE value_type convert_to_value_type(System* system) const
  {
    using thrust::system::detail::generic::select_system;
    return strip_const_get_value(select_system(*system));
  }

  template <typename System>
  _CCCL_HOST_DEVICE value_type strip_const_get_value(System const& system) const
  {
    System& non_const_system = const_cast<System&>(system);

    using thrust::system::detail::generic::get_value;
    return get_value(thrust::detail::derived_cast(non_const_system), ptr);
  }

  template <typename System0, typename System1, typename OtherPointer>
  _CCCL_HOST_DEVICE void assign_from(System0* system0, System1* system1, OtherPointer src) const
  {
    using thrust::system::detail::generic::select_system;
    strip_const_assign_value(select_system(*system0, *system1), src);
  }

  template <typename OtherPointer>
  _CCCL_HOST_DEVICE void assign_from(OtherPointer src) const
  {
    // Avoid default-constructing systems; instead, just use a null pointer
    // for dispatch. This assumes that `get_value` will not access any system
    // state.
    typename thrust::iterator_system<pointer>::type* system0      = nullptr;
    typename thrust::iterator_system<OtherPointer>::type* system1 = nullptr;
    assign_from(system0, system1, src);
  }

  template <typename System, typename OtherPointer>
  _CCCL_HOST_DEVICE void strip_const_assign_value(System const& system, OtherPointer src) const
  {
    System& non_const_system = const_cast<System&>(system);

    using thrust::system::detail::generic::assign_value;
    assign_value(thrust::detail::derived_cast(non_const_system), ptr, src);
  }

  template <typename System>
  _CCCL_HOST_DEVICE void swap(System* system, derived_type& other)
  {
    using thrust::system::detail::generic::iter_swap;
    using thrust::system::detail::generic::select_system;

    iter_swap(select_system(*system, *system), ptr, other.ptr);
  }
};

template <typename Pointer, typename Derived>
class reference<void, Pointer, Derived>
{};

template <typename Pointer, typename Derived>
class reference<void const, Pointer, Derived>
{};

template <typename Element, typename Pointer, typename Derived, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, reference<Element, Pointer, Derived> const& r)
{
  using value_type = typename reference<Element, Pointer, Derived>::value_type;
  return os << static_cast<value_type>(r);
}

template <typename Element, typename Tag>
class tagged_reference;

/*! \p tagged_reference acts as a reference-like wrapper for an object residing
 *  in memory associated with system \p Tag that a \p pointer refers to.
 */
template <typename Element, typename Tag>
class tagged_reference
    : public thrust::
        reference<Element, thrust::pointer<Element, Tag, tagged_reference<Element, Tag>>, tagged_reference<Element, Tag>>
{
private:
  using base_type = thrust::
    reference<Element, thrust::pointer<Element, Tag, tagged_reference<Element, Tag>>, tagged_reference<Element, Tag>>;

public:
  using value_type = typename base_type::value_type;
  using pointer    = typename base_type::pointer;

  tagged_reference(tagged_reference const&) = default;

  tagged_reference(tagged_reference&&) = default;

  /*! Construct a \p tagged_reference from another \p tagged_reference whose
   *  pointer type is convertible to \p pointer. After this \p tagged_reference
   *  is constructed, it shall refer to the same object as \p other.
   *
   *  \tparam OtherElement The element type of the other \p tagged_reference.
   *  \tparam OtherTag     The tag type of the other \p tagged_reference.
   *  \param  other        A \p tagged_reference to copy from.
   */
  template <typename OtherElement, typename OtherTag>
  _CCCL_HOST_DEVICE tagged_reference(tagged_reference<OtherElement, OtherTag> const& other)
      : base_type(other)
  {}

  /*! Construct a \p tagged_reference that refers to an object pointed to by
   *  the given \p pointer. After this \p tagged_reference is constructed, it
   *  shall refer to the object pointed to by \p ptr.
   *
   *  \param ptr A \p pointer to construct from.
   */
  _CCCL_HOST_DEVICE explicit tagged_reference(pointer const& p)
      : base_type(p)
  {}

  /*! Assign the object referred to \p other to the object referred to by
   *  this \p tagged_reference.
   *
   *  \param other The other \p tagged_reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  _CCCL_HOST_DEVICE const tagged_reference& operator=(tagged_reference const& other) const
  {
    return base_type::operator=(other);
  }

  /*! Assign the object referred to by this \p tagged_reference with the object
   *  referred to by another \p tagged_reference whose pointer type is
   *  convertible to \p pointer.
   *
   *  \tparam OtherElement The element type of the other \p tagged_reference.
   *  \tparam OtherTag     The tag type of the other \p tagged_reference.
   *  \param  other        The other \p tagged_reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  template <typename OtherElement, typename OtherTag>
  _CCCL_HOST_DEVICE const tagged_reference& operator=(tagged_reference<OtherElement, OtherTag> const& other) const
  {
    return base_type::operator=(other);
  }

  /*! Assign \p rhs to the object referred to by this \p tagged_reference.
   *
   *  \param rhs The \p value_type to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  _CCCL_HOST_DEVICE const tagged_reference& operator=(value_type const& rhs) const
  {
    return base_type::operator=(rhs);
  }
};

template <typename Tag>
class tagged_reference<void, Tag>
{};

template <typename Tag>
class tagged_reference<void const, Tag>
{};

/*! Exchanges the values of two objects referred to by \p tagged_reference.
 *
 *  \param x The first \p tagged_reference of interest.
 *  \param y The second \p tagged_reference of interest.
 */
// note: this is not a hidden friend, because we have template specializations of tagged_reference
template <typename Element, typename Tag>
_CCCL_HOST_DEVICE void
swap(tagged_reference<Element, Tag>& x, tagged_reference<Element, Tag>& y) noexcept(noexcept(x.swap(y)))
{
  x.swap(y);
}

THRUST_NAMESPACE_END
