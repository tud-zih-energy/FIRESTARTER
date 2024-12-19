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

#include "firestarter/Config/InstructionGroups.hpp"

#include <gtest/gtest.h>

TEST(InstructionGroupsTest, ParseValidInput) {
  const auto StringToParsed = std::map<std::string, std::vector<std::pair<std::string, unsigned>>>({
      {"Inst:1", {{"Inst", 1}}},
      {"Inst_2:2", {{"Inst_2", 2}}},
  });

  // Check if the individual ones work.
  for (const auto& [Input, Output] : StringToParsed) {
    const auto Groups = firestarter::InstructionGroups::fromString(Input);
    const auto ConvertedGroups = static_cast<std::vector<std::pair<std::string, unsigned>>>(Groups);
    EXPECT_EQ(ConvertedGroups, Output);

    {
      std::stringstream Ss;
      Ss << Groups;
      EXPECT_EQ(Ss.str(), Input);
    }
  }

  // Check combinations with two comma seperated values.
  for (const auto& [Input1, Output1] : StringToParsed) {
    for (const auto& [Input2, Output2] : StringToParsed) {
      auto Input = Input1;
      Input += ",";
      Input += Input2;

      auto Output = Output1;
      Output.insert(Output.end(), Output2.cbegin(), Output2.cend());

      const auto Groups = firestarter::InstructionGroups::fromString(Input);
      const auto ConvertedGroups = static_cast<std::vector<std::pair<std::string, unsigned>>>(Groups);
      EXPECT_EQ(ConvertedGroups, Output);

      {
        std::stringstream Ss;
        Ss << Groups;
        EXPECT_EQ(Ss.str(), Input);
      }
    }
  }
}

TEST(InstructionGroupsTest, ParseInvalidInput) {
  const auto ThrowStrings = std::vector<std::string>({"Inst", "$:1", "Inst:0", "Inst:-1", "Inst:$", "Inst:A"});

  // Check if the individual ones fail.
  for (const auto& Input : ThrowStrings) {
    EXPECT_ANY_THROW((void)firestarter::InstructionGroups::fromString(Input));
  }

  // Check if the combinations with two comma seperated values fail.
  for (const auto& Input1 : ThrowStrings) {
    for (const auto& Input2 : ThrowStrings) {
      auto Input = Input1;
      Input += ",";
      Input += Input2;

      EXPECT_ANY_THROW((void)firestarter::InstructionGroups::fromString(Input));
    }
  }
}