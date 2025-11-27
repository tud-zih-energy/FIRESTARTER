/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2025 TU Dresden, Center for Information Services and High
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

#include <cstdint>

namespace firestarter::payload {

/// This class describes the type of control flow that is used for the generation of the high load assembler kernels.
enum class HighLoadControlFlowDescription : std::uint8_t {
  // The control flow is generated such that the LoadThreadWorkType variable is used to interrupt the hot loop.
  kStopOnLoadThreadWorkType = 0,
  // The control flow is generated such that the LoadThreadWorkType variable is used to interrupt the hot loop. In
  // addition the hot loop will repeat a specified maximum number of iterations.
  kMaxIterationCount = 1
};

} // namespace firestarter::payload
