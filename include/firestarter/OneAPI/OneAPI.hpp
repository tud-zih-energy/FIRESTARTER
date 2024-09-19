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

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace firestarter::oneapi {

class OneAPI {
private:
  std::thread _initThread;
  std::condition_variable _waitForInitCv;
  std::mutex _waitForInitCvMutex;

  static void initGpus(std::condition_variable& cv, volatile unsigned long long* loadVar, bool useFloat, bool useDouble,
                       unsigned matrixSize, int gpus);

public:
  OneAPI(volatile unsigned long long* loadVar, bool useFloat, bool useDouble, unsigned matrixSize, int gpus);

  ~OneAPI() {
    if (_initThread.joinable()) {
      _initThread.join();
    }
  }
};

} // namespace firestarter::oneapi
