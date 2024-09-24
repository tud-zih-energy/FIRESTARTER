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

#include <nitro/log/log.hpp>
#include <nitro/log/severity.hpp>
#include <thread>

namespace firestarter::logging {

template <typename Record> class FirstWorkerThreadFilter {
public:
  using record_type = Record;

  static void setFirstThread(std::thread::id NewFirstThread) { FirstThread = NewFirstThread; }

  auto filter(Record& r) const -> bool {
    return r.std_thread_id() == FirstThread || r.severity() >= nitro::log::severity_level::error;
  }

private:
  inline static std::thread::id FirstThread{};
};
} // namespace firestarter::logging
