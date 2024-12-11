/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/\>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#include "firestarter/Config/CpuBind.hpp"

#include <gtest/gtest.h>

TEST(CpuBindTest, ParseValidInput) {
  const auto StringToParsed = std::map<std::string, std::set<uint64_t>>({
      {
          "0",
          {0},
      },
      {
          "7",
          {7},
      },
      {
          "1,3,5",
          {1, 3, 5},
      },
      {
          "4-7",
          {4, 5, 6, 7},
      },
      {
          "4-8/1",
          {4, 5, 6, 7, 8},
      },
      {
          "4-8/2",
          {4, 6, 8},
      },
      {
          "4-10/3",
          {4, 7, 10},
      },
  });

  // Check if the individual ones work.
  for (const auto& [Input, Output] : StringToParsed) {
    EXPECT_EQ(firestarter::CpuBind::fromString(Input), Output);
  }

  // Check combinations with two comma seperated values.
  for (const auto& [Input1, Output1] : StringToParsed) {
    for (const auto& [Input2, Output2] : StringToParsed) {
      auto Input = Input1;
      Input += ",";
      Input += Input2;

      auto Output = Output1;
      Output.merge(std::set<uint64_t>(Output2));

      EXPECT_EQ(firestarter::CpuBind::fromString(Input), Output);
    }
  }
}

TEST(CpuBindTest, ParseInvalidInput) {
  std::vector<std::string> ThrowStrings = {"-1", "1,-1", "-1-4", "-4-1", "1-3/-1", "1-3/0", "A", "1-3/A", "A-B"};

  // Check if the individual ones fail.
  for (const auto& Input : ThrowStrings) {
    EXPECT_ANY_THROW(firestarter::CpuBind::fromString(Input));
  }

  // Check if the combinations with two comma seperated values fail.
  for (const auto& Input1 : ThrowStrings) {
    for (const auto& Input2 : ThrowStrings) {
      auto Input = Input1;
      Input += ",";
      Input += Input2;

      EXPECT_ANY_THROW(firestarter::CpuBind::fromString(Input));
    }
  }
}