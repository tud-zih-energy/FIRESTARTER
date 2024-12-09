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

#include "firestarter/CPUTopology.hpp"

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

namespace firestarter {

/// This class handles selecting the correct number of threads. They are parsed as user input to FIRESTARTER, but still
/// need checking with runtime information.
struct ThreadAffinity {
  /// Parse the user input for the cpu affinity and the number of requested threads. If a CpuBind is provided we
  /// evaluate it and set the number of threads and their affinity accordingly. This function will save the correct
  /// number of threads based on the user input in RequestedNumThreads. It must be called for FIRESTARTER to function
  /// properly.
  /// \arg ThreadsInfo The information about hardware threads on the current platform.
  /// \arg RequestedNumThreads The optional number of threads that are requested by a user. If this is std::nullopt the
  /// number will be automatically determined.
  /// \arg CpuBinding The optional vector of cpus which are used for FIRESTARTER. It overrides the requested number of
  /// threads.
  [[nodiscard]] static auto fromCommandLine(const HardwareThreadsInfo& ThreadsInfo,
                                            const std::optional<unsigned>& RequestedNumThreads,
                                            const std::optional<std::vector<uint64_t>>& CpuBinding) -> ThreadAffinity;

  /// Print the summary of the used thread for the workers. If thread affinity is supported (linux and windows), print
  /// which thread is pinned to which CPU.
  /// \arg Topology The topological information about the processor.
  void printThreadSummary(const CPUTopology& Topology) const;

  /// The number of threads FIRESTARTER is requested to run with.
  uint64_t RequestedNumThreads = 0;

  /// The list of physical CPU ids that are requested to be used. The length of this list matches the number of
  /// requested threads.
  std::vector<unsigned> CpuBind;
};

} // namespace firestarter
