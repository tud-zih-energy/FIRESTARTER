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
  void sink(nitro::log::severity_level, const std::string &formatted_record) {
    std::cout << formatted_record << std::endl << std::flush;
  }
};

using record = nitro::log::record<nitro::log::severity_attribute,
                                  nitro::log::message_attribute,
                                  nitro::log::timestamp_attribute>;

template <typename Record> class formater {
public:
  std::string format(Record &r) {
    std::stringstream s;
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
