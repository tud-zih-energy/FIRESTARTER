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

namespace firestarter {

struct ErrorDetectionStruct {
  // we have two cache lines (64B) containing each two 16B local variable and
  // one ptr (8B)

  // the pointer to 16B of communication
  volatile unsigned long long *communicationLeft;
  volatile unsigned long long localsLeft[4];
  // if this variable is not 0, an error occured in the comparison with the left
  // thread.
  volatile unsigned long long errorLeft;
  volatile unsigned long long paddingLeft[2];

  volatile unsigned long long *communicationRight;
  volatile unsigned long long localsRight[4];
  // if this variable is not 0, an error occured in the comparison with the
  // right thread.
  volatile unsigned long long errorRight;
  volatile unsigned long long paddingRight[2];
};

} // namespace firestarter
