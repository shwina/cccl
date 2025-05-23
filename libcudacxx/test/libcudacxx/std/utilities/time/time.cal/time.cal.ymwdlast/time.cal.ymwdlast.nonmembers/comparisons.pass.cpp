//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday_last;

// constexpr bool operator==(const year_month_weekday_last& x, const year_month_weekday_last& y) noexcept;
//   Returns: x.year() == y.year() && x.month() == y.month() && x.weekday_last() == y.weekday_last()
//

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday                 = cuda::std::chrono::weekday;
  using weekday_last            = cuda::std::chrono::weekday_last;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;

  AssertEqualityAreNoexcept<year_month_weekday_last>();
  AssertEqualityReturnBool<year_month_weekday_last>();

  constexpr month January     = cuda::std::chrono::January;
  constexpr month February    = cuda::std::chrono::February;
  constexpr weekday Tuesday   = cuda::std::chrono::Tuesday;
  constexpr weekday Wednesday = cuda::std::chrono::Wednesday;

  static_assert(testEquality(year_month_weekday_last{year{1234}, January, weekday_last{Tuesday}},
                             year_month_weekday_last{year{1234}, January, weekday_last{Tuesday}},
                             true),
                "");

  //  different day
  static_assert(testEquality(year_month_weekday_last{year{1234}, January, weekday_last{Tuesday}},
                             year_month_weekday_last{year{1234}, January, weekday_last{Wednesday}},
                             false),
                "");

  //  different month
  static_assert(testEquality(year_month_weekday_last{year{1234}, January, weekday_last{Tuesday}},
                             year_month_weekday_last{year{1234}, February, weekday_last{Tuesday}},
                             false),
                "");

  //  different year
  static_assert(testEquality(year_month_weekday_last{year{1234}, January, weekday_last{Tuesday}},
                             year_month_weekday_last{year{1235}, January, weekday_last{Tuesday}},
                             false),
                "");

  //  different month and day
  static_assert(testEquality(year_month_weekday_last{year{1234}, January, weekday_last{Tuesday}},
                             year_month_weekday_last{year{1234}, February, weekday_last{Wednesday}},
                             false),
                "");

  //  different year and month
  static_assert(testEquality(year_month_weekday_last{year{1234}, February, weekday_last{Tuesday}},
                             year_month_weekday_last{year{1235}, January, weekday_last{Tuesday}},
                             false),
                "");

  //  different year and day
  static_assert(testEquality(year_month_weekday_last{year{1234}, January, weekday_last{Wednesday}},
                             year_month_weekday_last{year{1235}, January, weekday_last{Tuesday}},
                             false),
                "");

  //  different year, month and day
  static_assert(testEquality(year_month_weekday_last{year{1234}, February, weekday_last{Wednesday}},
                             year_month_weekday_last{year{1235}, January, weekday_last{Tuesday}},
                             false),
                "");

  //  same year, different days
  for (unsigned i = 1; i < 28; ++i)
  {
    for (unsigned j = 1; j < 28; ++j)
    {
      assert((testEquality(year_month_weekday_last{year{1234}, January, weekday_last{weekday{i}}},
                           year_month_weekday_last{year{1234}, January, weekday_last{weekday{j}}},
                           i == j)));
    }
  }

  //  same year, different months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testEquality(year_month_weekday_last{year{1234}, month{i}, weekday_last{Tuesday}},
                           year_month_weekday_last{year{1234}, month{j}, weekday_last{Tuesday}},
                           i == j)));
    }
  }

  //  same month, different years
  for (int i = 1000; i < 20; ++i)
  {
    for (int j = 1000; j < 20; ++j)
    {
      assert((testEquality(year_month_weekday_last{year{i}, January, weekday_last{Tuesday}},
                           year_month_weekday_last{year{j}, January, weekday_last{Tuesday}},
                           i == j)));
    }
  }

  return 0;
}
