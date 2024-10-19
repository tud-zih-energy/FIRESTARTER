/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2023 TU Dresden, Center for Information Services and High
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
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#pragma once

#include "firestarter/Constants.hpp"
#include <condition_variable>
#include <thread>

namespace firestarter::oneapi {

class OneAPI {
private:
  std::thread InitThread;

  static void initGpus(std::condition_variable& WaitForInitCv, const volatile firestarter::LoadThreadWorkType& LoadVar,
                       bool UseFloat, bool UseDouble, unsigned MatrixSize, int Gpus);

public:
  OneAPI(const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble, unsigned MatrixSize,
         int Gpus)
#if defined(FIRESTARTER_BUILD_ONEAPI)
      ;
#else
  {
    (void)&LoadVar;
    (void)UseFloat;
    (void)UseDouble;
    (void)MatrixSize;
    (void)Gpus;
  }
#endif

  ~OneAPI() {
    if (InitThread.joinable()) {
      InitThread.join();
    }
  }
};

} // namespace firestarter::oneapi