#ifndef INCLUDE_FIRESTARTER_LOG_HPP
#define INCLUDE_FIRESTARTER_LOG_HPP

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
      std::cout << formatted_record << std::endl << std::flush;
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

#endif
