/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2025 TU Dresden, Center for Information Services and High
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

#include "firestarter/Config/MetricName.hpp"

#include <gtest/gtest.h>

TEST(MetricNameTest, ParseValidInput) {

  const auto StringToParsed = std::map<std::string, firestarter::MetricName>({
      {"sysfs-powercap-rapl", firestarter::MetricName(/*Inverted*/ false, "sysfs-powercap-rapl")},
      {"-sysfs-powercap-rapl", firestarter::MetricName(/*Inverted*/ true, "sysfs-powercap-rapl")},
      {"sysfs-powercap-rapl/sub-metric1",
       firestarter::MetricName(/*Inverted*/ false, "sysfs-powercap-rapl", "sub-metric1")},
      {"-sysfs-powercap-rapl/sub-metric2",
       firestarter::MetricName(/*Inverted*/ true, "sysfs-powercap-rapl", "sub-metric2")},
  });

  // Check if the individual ones work.
  for (const auto& [Input, Output] : StringToParsed) {
    const auto Name = firestarter::MetricName::fromString(Input);
    EXPECT_EQ(Name, Output);

    EXPECT_EQ(Name.toString(), Input);
  }
}