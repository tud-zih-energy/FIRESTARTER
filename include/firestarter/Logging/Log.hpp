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

#include "FirstWorkerThreadFilter.hpp"
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
  }
};

using Record = nitro::log::record<nitro::log::severity_attribute, nitro::log::message_attribute,
                                  nitro::log::timestamp_attribute, nitro::log::std_thread_id_attribute>;

template <typename Record> class Formater {
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

using log = nitro::log::logger<logging::Record, logging::Formater, firestarter::logging::StdOut, logging::Filter>;

using workerLog =
    nitro::log::logger<logging::Record, logging::Formater, firestarter::logging::StdOut, logging::WorkerFilter>;

} // namespace firestarter
