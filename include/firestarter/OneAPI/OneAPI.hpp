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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <thread>

namespace firestarter::oneapi {

/// This struct contains the number of flop estimated that have been executed. It will be incremented by the flop when
/// on execution of a kernel is complete.
struct GpuFlop {
  /// The number of executed single precision flop
  std::atomic<std::size_t> SingleFlop = 0;
  /// The number of executed double precision flop
  std::atomic<std::size_t> DoubleFlop = 0;
};

/// This class handles the workload on OneAPI compatible GPUs. A gemm routine is used to stress them with a
/// constant high load. This header does not include any OneAPI specific headers to allow us to not guard the
/// include of this header in other parts of the programm.
class OneAPI {
private:
  /// The thread that is used to initilize the gpus. This thread will wait until each thread that runs the gemm routine
  /// joins.
  std::thread InitThread;

  /// The estimation on the number of flops that have been executed. It will be incremented by the flops when on
  /// execution of a kernel is complete.
  GpuFlop ExecutedFlop;

  /// Spawns a thread for each of the selected gpus, initilizes them and starts the execution of the gemm in parallel.
  /// \arg ExecutedFlop The variable that contains the number of flop estimated that have been executed. It will be
  /// incremented by the flops when on execution of a kernel is complete.
  /// \arg WaitForInitCv The condition variables used to signal that all gpus are initialized.
  /// \arg LoadVar A reference to the variable that controlls the current load of Firestarter.
  /// \arg UseFloat Set to true if we want to stress using single precision floating points.
  /// \arg UseDouble Set to true if we want to stress using double precision floating points. If neither UseFloat or
  /// UseDouble is set the precision will be choosen automatically.
  /// \arg MatrixSize Set to a specific matrix size which will be choosen for the gemm operation or set to 0 for
  /// automatic selection.
  /// \arg Gpus Select the number of gpus to stress or -1 for all.
  static void initGpus(GpuFlop& ExecutedFlop, std::condition_variable& WaitForInitCv,
                       const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble,
                       uint64_t MatrixSize, int Gpus);

public:
  /// Initilize the OneAPI class. This will start a thread running the OneAPI::initGpus function and wait until all gpus
  /// are inititialized.
  /// \arg LoadVar A reference to the variable that controlls the current load of Firestarter.
  /// \arg UseFloat Set to true if we want to stress using single precision floating points.
  /// \arg UseDouble Set to true if we want to stress using double precision floating points. If neither UseFloat or
  /// UseDouble is set the precision will be choosen automatically.
  /// \arg MatrixSize Set to a specific matrix size which will be choosen for the gemm operation or set to 0 for
  /// automatic selection.
  /// \arg Gpus Select the number of gpus to stress or -1 for all.
  OneAPI(const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble, uint64_t MatrixSize,
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

  /// Get the estimation on the executed flops of the cuda thread.
  /// \returns The number of estimated executed flop.
  auto executedFlop() -> const GpuFlop& { return ExecutedFlop; };
};

} // namespace firestarter::oneapi