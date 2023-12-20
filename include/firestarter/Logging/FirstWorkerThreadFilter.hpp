/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2021 TU Dresden, Center for Information Services and High
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

#include <thread>

namespace firestarter {

namespace logging {

template <typename Record> class FirstWorkerThreadFilter {
public:
  typedef Record record_type;

  static void setFirstThread(std::thread::id newFirstThread) {
    firstThread = newFirstThread;
  }

  bool filter(Record &r) const {
    return r.std_thread_id() == firstThread ||
           r.severity() >= nitro::log::severity_level::error;
  }

private:
  inline static std::thread::id firstThread{};
};
} // namespace logging

} // namespace firestarter
