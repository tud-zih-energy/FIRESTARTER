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

#include <nitro/log/log.hpp>
#include <nitro/log/severity.hpp>

#include <nitro/log/attribute/message.hpp>
#include <nitro/log/attribute/severity.hpp>
#include <nitro/log/attribute/timestamp.hpp>

#include <nitro/log/filter/severity_filter.hpp>

#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>

namespace firestarter {

namespace logging {

class StdOut {
public:
  void sink(nitro::log::severity_level severity,
            const std::string &formatted_record) {
    switch (severity) {
    case nitro::log::severity_level::warn:
    case nitro::log::severity_level::error:
    case nitro::log::severity_level::fatal:
      std::cerr << formatted_record << std::endl << std::flush;
      break;
    default:
      std::cout << formatted_record << std::endl;
      break;
    }
  }
};

using record = nitro::log::record<nitro::log::severity_attribute,
                                  nitro::log::message_attribute,
                                  nitro::log::timestamp_attribute>;

template <typename Record> class formater {
public:
  std::string format(Record &r) {
    std::stringstream s;

    switch (r.severity()) {
    case nitro::log::severity_level::warn:
      s << "Warning: ";
      break;
    case nitro::log::severity_level::error:
      s << "Error: ";
      break;
    case nitro::log::severity_level::fatal:
      s << "Fatal: ";
      break;
    case nitro::log::severity_level::trace:
      s << "Debug: ";
      break;
    default:
      break;
    }

    s << r.message();

    return s.str();
  }
};

template <typename Record>
using filter = nitro::log::filter::severity_filter<Record>;

} // namespace logging

using log = nitro::log::logger<logging::record, logging::formater,
                               firestarter::logging::StdOut, logging::filter>;

} // namespace firestarter
