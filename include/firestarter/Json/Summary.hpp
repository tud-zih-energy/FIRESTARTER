/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

#pragma once

#include "firestarter/Measurement/Summary.hpp"

/// Json serializer and deserializer for the firestarter::measurement::Summary struct
namespace nlohmann {
template <> struct adl_serializer<firestarter::measurement::Summary> {
  // functions for nlohmann json do not follow LLVM code style
  // NOLINTBEGIN(readability-identifier-naming)
  static auto from_json(const json& J) -> firestarter::measurement::Summary {
    return {J["num_timepoints"].get<size_t>(),
            std::chrono::milliseconds(J["duration"].get<std::chrono::milliseconds::rep>()), J["average"].get<double>(),
            J["stddev"].get<double>()};
  }

  static void to_json(json& J, firestarter::measurement::Summary S) {
    J = json::object();

    J["num_timepoints"] = S.NumTimepoints;
    J["duration"] = S.Duration.count();
    J["average"] = S.Average;
    J["stddev"] = S.Stddev;
  }
  // NOLINTEND(readability-identifier-naming)
};
} // namespace nlohmann
