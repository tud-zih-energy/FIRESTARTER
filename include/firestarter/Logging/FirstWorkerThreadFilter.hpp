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

#include <nitro/log/severity.hpp>
#include <thread>

namespace firestarter::logging {

/// Logging filter for nitro to discard values that do not match a specific thread id.
template <typename Record> class FirstWorkerThreadFilter {
public:
  using record_type = Record;

  /// Set the thread id from which records should not be discarded.
  /// \arg NewFirstThread The specified thread.
  static void setFirstThread(std::thread::id NewFirstThread) { FirstThread = NewFirstThread; }

  /// Filter records. We keep record if they are from the specified thread or if the severity is at least error.
  /// \arg R The record to filter.
  /// \returns true if the record should be kept.
  auto filter(Record& R) const -> bool {
    return R.std_thread_id() == FirstThread || R.severity() >= nitro::log::severity_level::error;
  }

private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  inline static std::thread::id FirstThread{};
};
} // namespace firestarter::logging
