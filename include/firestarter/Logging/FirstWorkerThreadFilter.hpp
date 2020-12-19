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

namespace firestarter {

namespace logging {

template <typename Record> class FirstWorkerThreadFilter {
public:
  typedef Record record_type;

  static void setFirstThread(std::uint64_t newFirstThread) {
    firstThread = newFirstThread;
  }

  bool filter(Record &r) const {
    return r.pthread_id() == firstThread ||
           r.severity() >= nitro::log::severity_level::error;
  }

private:
  static std::uint64_t firstThread;
};

template <typename Record>
std::uint64_t FirstWorkerThreadFilter<Record>::firstThread = 0;
} // namespace logging

} // namespace firestarter
