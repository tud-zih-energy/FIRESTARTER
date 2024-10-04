/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2021 TU Dresden, Center for Information Services and High
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

using EightBytesType = uint64_t;

// We want enum to have the size of 8B. Disable the warnings for bigger enum size than needed.
// NOLINTBEGIN(performance-enum-size)

enum class LoadThreadState : EightBytesType { ThreadWait = 1, ThreadWork = 2, ThreadInit = 3, ThreadSwitch = 4 };

enum class LoadThreadWorkType : EightBytesType {
  /* DO NOT CHANGE! the asm load-loop tests if load-variable is == 0 */
  LoadLow = 0,
  /* DO NOT CHANGE! the asm load-loop continues until the load-variable is != 1 */
  LoadHigh = 1,
  LoadStop = 2,
  LoadSwitch = 4
};
// NOLINTEND(performance-enum-size)