//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday;

// constexpr bool operator==(const year_month_weekday& x, const year_month_weekday& y) noexcept;
//   Returns: x.year() == y.year() && x.month() == y.month() && x.weekday_indexed() == y.weekday_indexed()
//

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using year               = cuda::std::chrono::year;
  using month              = cuda::std::chrono::month;
  using weekday_indexed    = cuda::std::chrono::weekday_indexed;
  using weekday            = cuda::std::chrono::weekday;
  using year_month_weekday = cuda::std::chrono::year_month_weekday;

  AssertEqualityAreNoexcept<year_month_weekday>();
  AssertEqualityReturnBool<year_month_weekday>();

  constexpr month January   = cuda::std::chrono::January;
  constexpr month February  = cuda::std::chrono::February;
  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  static_assert(testEquality(year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
                             year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
                             true),
                "");

  //  different day
  static_assert(testEquality(year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
                             year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 2}},
                             false),
                "");

  //  different month
  static_assert(testEquality(year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
                             year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 1}},
                             false),
                "");

  //  different year
  static_assert(testEquality(year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
                             year_month_weekday{year{1235}, January, weekday_indexed{Tuesday, 1}},
                             false),
                "");

  //  different month and day
  static_assert(testEquality(year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 1}},
                             year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 2}},
                             false),
                "");

  //  different year and month
  static_assert(testEquality(year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 1}},
                             year_month_weekday{year{1235}, January, weekday_indexed{Tuesday, 1}},
                             false),
                "");

  //  different year and day
  static_assert(testEquality(year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, 2}},
                             year_month_weekday{year{1235}, January, weekday_indexed{Tuesday, 1}},
                             false),
                "");

  //  different year, month and day
  static_assert(testEquality(year_month_weekday{year{1234}, February, weekday_indexed{Tuesday, 2}},
                             year_month_weekday{year{1235}, January, weekday_indexed{Tuesday, 1}},
                             false),
                "");

  //  same year, different days
  for (unsigned i = 1; i < 28; ++i)
  {
    for (unsigned j = 1; j < 28; ++j)
    {
      assert((testEquality(year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, i}},
                           year_month_weekday{year{1234}, January, weekday_indexed{Tuesday, j}},
                           i == j)));
    }
  }

  //  same year, different months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testEquality(year_month_weekday{year{1234}, month{i}, weekday_indexed{Tuesday, 1}},
                           year_month_weekday{year{1234}, month{j}, weekday_indexed{Tuesday, 1}},
                           i == j)));
    }
  }

  //  same month, different years
  for (int i = 1000; i < 20; ++i)
  {
    for (int j = 1000; j < 20; ++j)
    {
      assert((testEquality(year_month_weekday{year{i}, January, weekday_indexed{Tuesday, 1}},
                           year_month_weekday{year{j}, January, weekday_indexed{Tuesday, 1}},
                           i == j)));
    }
  }

  return 0;
}
