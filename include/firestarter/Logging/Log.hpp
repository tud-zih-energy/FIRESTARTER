/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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

#include "firestarter/Logging/FirstWorkerThreadFilter.hpp"
#include "firestarter/SafeExit.hpp"

#include <cstdlib>
#include <iostream>
#include <nitro/log/attribute/message.hpp>
#include <nitro/log/attribute/severity.hpp>
#include <nitro/log/attribute/std_thread_id.hpp>
#include <nitro/log/attribute/timestamp.hpp>
#include <nitro/log/filter/and_filter.hpp>
#include <nitro/log/filter/severity_filter.hpp>
#include <nitro/log/log.hpp>
#include <nitro/log/severity.hpp>
#include <sstream>
#include <string>

namespace firestarter {

namespace logging {

/// Formatter to log Records with severity warn, error and fatal to stderr and all other Records to stdout. If a record
/// has severity error or fatal we abort the program.
class StdOut {
public:
  static void sink(nitro::log::severity_level Severity, const std::string& FormattedRecord) {
    switch (Severity) {
    case nitro::log::severity_level::warn:
    case nitro::log::severity_level::error:
    case nitro::log::severity_level::fatal:
      std::cerr << FormattedRecord << '\n' << std::flush;
      break;
    default:
      std::cout << FormattedRecord << '\n' << std::flush;
      break;
    }

    // Exit on error or fatal
    if (Severity == nitro::log::severity_level::error || Severity == nitro::log::severity_level::fatal) {
      safeExit(EXIT_FAILURE);
    }
  }
};

// NOLINTBEGIN(readability-identifier-naming)
// The class may not be named Record since this is used as a template argument name in nitro which will cause errors
// when compiling with MSC.
using record = nitro::log::record<nitro::log::severity_attribute, nitro::log::message_attribute,
                                  nitro::log::timestamp_attribute, nitro::log::std_thread_id_attribute>;
// NOLINTEND(readability-identifier-naming)

template <typename Record>
// NOLINTBEGIN(readability-identifier-naming)
// The class may not be named Formater since this is used as a template argument name in nitro which will cause errors
// when compiling with MSC. We will also write it with lower case and the correct spelling in case it gets renamed
// correctly there.
/// Format Record and add a string representing the severity in front.
class formatter {
  // NOLINTEND(readability-identifier-naming)
public:
  auto format(Record& R) -> std::string {
    std::stringstream S;

    switch (R.severity()) {
    case nitro::log::severity_level::warn:
      S << "Warning: ";
      break;
    case nitro::log::severity_level::error:
      S << "Error: ";
      break;
    case nitro::log::severity_level::fatal:
      S << "Fatal: ";
      break;
    case nitro::log::severity_level::trace:
      S << "Debug: ";
      break;
    default:
      break;
    }

    S << R.message();

    return S.str();
  }
};

template <typename Record> using Filter = nitro::log::filter::severity_filter<Record>;

template <typename Record>
using WorkerFilter = nitro::log::filter::and_filter<Filter<Record>, FirstWorkerThreadFilter<Record>>;

} // namespace logging

using log = nitro::log::logger<logging::record, logging::formatter, firestarter::logging::StdOut, logging::Filter>;

using workerLog =
    nitro::log::logger<logging::record, logging::formatter, firestarter::logging::StdOut, logging::WorkerFilter>;

} // namespace firestarter
