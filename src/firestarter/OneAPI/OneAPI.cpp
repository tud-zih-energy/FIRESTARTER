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

/* OneAPI for GPUs, based on CUDA component
 *****************************************************************************/

#include "firestarter/OneAPI/OneAPI.hpp"
#include "firestarter/Logging/Log.hpp"

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <type_traits>

namespace firestarter::oneapi {

namespace {

/// Helper function to generate random floating point values between 0 and 1 in an array.
/// \tparam FloatingPointType The type of floating point value of the array. Either float or double.
/// \arg NumberOfElems The number of elements of the array.
/// \arg Array The array of floating point values which should be initilized with random data between 0 and 1.
template <typename FloatingPointType> void fillArrayWithRandomFloats(size_t NumberOfElems, FloatingPointType* Array) {
  static_assert(std::is_same_v<FloatingPointType, float> || std::is_same_v<FloatingPointType, double>,
                "fillArrayWithRandomFloats<FloatingPointType>: Template argument must be either float or double");

  for (size_t i = 0; i < NumberOfElems; i++) {
    Array[i] = static_cast<FloatingPointType>(std::rand()) / static_cast<FloatingPointType>(RAND_MAX);
  }
}

template <typename FloatingPointType>
void replicateData(sycl::queue& Q, FloatingPointType* Dst, size_t DstElems, const FloatingPointType* Src,
                   size_t SrcElems) {
  static_assert(std::is_same_v<FloatingPointType, float> || std::is_same_v<FloatingPointType, double>,
                "fillArrayWithRandomFloats<FloatingPointType>: Template argument must be either float or double");

  firestarter::log::trace() << "replicateData<FloatingPointType> " << DstElems << " elements from " << Src << " to "
                            << Dst;
  while (DstElems > 0) {
    auto copy_elems = std::min(DstElems, SrcElems);
    Q.copy(Src, Dst, copy_elems);
    Dst += copy_elems;
    DstElems -= copy_elems;
  }
  Q.wait();
}

int getPrecision(int DeviceIndex, int UseDouble) {
  firestarter::log::trace() << "Checking UseDouble " << UseDouble;

  if (!UseDouble) {
    return 0;
  }

  int SupportsDouble = 0;

  auto Platforms = sycl::platform::get_platforms();

  if (Platforms.empty()) {
    firestarter::log::warn() << "No SYCL platforms found.";
    return -1;
  }
  // Choose a platform based on specific criteria (e.g., device type)
  // TODO(Issue #75): We may select the incorrect platform with gpu devices of the wrong vendor/type.
  sycl::platform ChosenPlatform;
  auto NbGpus = 0;
  for (const auto& Platform : Platforms) {
    firestarter::log::trace() << "Checking SYCL platform " << Platform.get_info<sycl::info::platform::name>();
    auto Devices = Platform.get_devices();
    NbGpus = 0;
    for (const auto& Device : Devices) {
      firestarter::log::trace() << "Checking SYCL device " << Device.get_info<sycl::info::device::name>();
      if (Device.is_gpu()) { // Choose GPU, you can use other criteria
        firestarter::log::trace() << " ... is GPU";
        ChosenPlatform = Platform;
        NbGpus++;
      }
    }
  }

  if (!NbGpus) {
    firestarter::log::warn() << "No suitable platform with GPU found.";
    return -1;
  }
  // Get a list of devices for the chosen platform

  firestarter::log::trace() << "Get support for double" << " on device nr. " << DeviceIndex;
  auto Devices = ChosenPlatform.get_devices();
  if (Devices[DeviceIndex].has(sycl::aspect::fp64))
    SupportsDouble = 1;

  return SupportsDouble;
}

template <std::size_t Multiple> auto roundUp(int NumToRound) -> int {
  static_assert(Multiple != 0, "Multiple may not be zero.");

  const int Remainder = NumToRound % Multiple;
  if (Remainder == 0) {
    return NumToRound;
  }

  return NumToRound + Multiple - Remainder;
}

// GPU index. Used to pin this thread to the GPU.
// The main difference to the CUDA/HIP version is that we do not run multiple iterations of C=A*B, just one single
// iteration.
template <typename FloatingPointType>
void createLoad(GpuFlop& ExecutedFlop, std::condition_variable& WaitForInitCv, std::mutex& WaitForInitCvMutex,
                int DeviceIndex, std::atomic<int>& InitCount, const volatile firestarter::LoadThreadWorkType& LoadVar,
                unsigned MatrixSize) {
  static_assert(std::is_same<FloatingPointType, float>::value || std::is_same<FloatingPointType, double>::value,
                "createLoad<T>: Template argument T must be either float or double");

  firestarter::log::trace() << "Starting OneAPI with given matrix size " << MatrixSize;

  // reserving the GPU and initializing

  firestarter::log::trace() << "Getting device nr. " << DeviceIndex;

  auto Platforms = sycl::platform::get_platforms();

  if (Platforms.empty()) {
    firestarter::log::warn() << "No SYCL platforms found.";
    return;
  }

  // Choose a platform based on specific criteria (e.g., device type)
  sycl::platform ChosenPlatform;
  auto NbGpus = 0;
  for (const auto& Platform : Platforms) {
    auto Devices = Platform.get_devices();
    NbGpus = 0;
    for (const auto& Device : Devices) {
      if (Device.is_gpu()) { // Choose GPU, you can use other criteria
        ChosenPlatform = Platform;
        NbGpus++;
      }
    }
  }

  if (!NbGpus) {
    firestarter::log::warn() << "No suitable platform with GPU found.";
    return;
  }

  // Get a list of devices for the chosen platform
  auto Devices = ChosenPlatform.get_devices();

  firestarter::log::trace() << "Creating SYCL queue for computation on device nr. " << DeviceIndex;
  auto ChosenDevice = Devices[DeviceIndex];
  auto DeviceQueue = sycl::queue(ChosenDevice);

  firestarter::log::trace() << "Get memory size on device nr. " << DeviceIndex;

  // getting information about the GPU memory
  size_t MemoryTotal = Devices[DeviceIndex].get_info<sycl::info::device::global_mem_size>();

  firestarter::log::trace() << "Get Memory info on device nr. " << DeviceIndex << ": has " << MemoryTotal
                            << " B global memory";

  // If the matrix size is not set or three square matricies with dim size of SizeUse do not fit into the available
  // memory, select the size so that 3 square matricies will fit into the available device memory where the dim size
  // is a multiple of 1024.
  std::size_t MemorySize = sizeof(FloatingPointType) * MatrixSize * MatrixSize;
  if (!MatrixSize || (MemorySize * 3 > MemoryTotal)) {
    // a multiple of 1024 works always well
    MatrixSize = roundUp<1024>(0.8 * std::sqrt(MemoryTotal / sizeof(FloatingPointType) / 3));
    MemorySize = sizeof(FloatingPointType) * MatrixSize * MatrixSize;
  }

  firestarter::log::trace() << "Set OneAPI matrix size in B: " << MatrixSize;

  /* Allocate A/B/C matrices */

  firestarter::log::trace() << "Allocating memory on device nr. " << DeviceIndex;
  auto* A = sycl::malloc_device<FloatingPointType>(MatrixSize * MatrixSize, DeviceQueue);
  auto* B = sycl::malloc_device<FloatingPointType>(MatrixSize * MatrixSize, DeviceQueue);
  auto* C = sycl::malloc_device<FloatingPointType>(MatrixSize * MatrixSize, DeviceQueue);

  /* Create 64 MB random data on Host */
  constexpr int RandomSize = 1024 * 1024 * 64;
  auto* RandomData = sycl::malloc_host<FloatingPointType>(RandomSize, DeviceQueue);
  fillArrayWithRandomFloats<FloatingPointType>(RandomSize, RandomData);

  firestarter::log::trace() << "Copy memory to device nr. " << DeviceIndex;
  /* fill A and B with random data */
  replicateData(DeviceQueue, A, MatrixSize * MatrixSize, RandomData, RandomSize);
  replicateData(DeviceQueue, B, MatrixSize * MatrixSize, RandomData, RandomSize);

  {
    std::lock_guard<std::mutex> lk(WaitForInitCvMutex);

    auto ToMiB = [](const size_t Val) { return Val / 1024 / 1024; };
    firestarter::log::info() << "   GPU " << DeviceIndex << "\n"
                             << "    name:           " << Devices[DeviceIndex].get_info<sycl::info::device::name>()
                             << "\n"
                             << "    memory:         " << ToMiB(MemoryTotal) << " MiB total (using "
                             << ToMiB(MemorySize) << " MiB)\n"
                             << "    matrix size:    " << MatrixSize << "\n"
                             << "    used precision: "
                             << ((sizeof(FloatingPointType) == sizeof(double)) ? "double" : "single");

    InitCount++;
  }
  WaitForInitCv.notify_all();

  firestarter::log::trace() << "Run gemm on device nr. " << DeviceIndex;
  while (LoadVar != firestarter::LoadThreadWorkType::LoadStop) {
    firestarter::log::trace() << "Run gemm on device nr. " << DeviceIndex;
    ::oneapi::mkl::blas::gemm(DeviceQueue, ::oneapi::mkl::transpose::N, ::oneapi::mkl::transpose::N, MatrixSize,
                              MatrixSize, MatrixSize, 1, A, MatrixSize, B, MatrixSize, 0, C, MatrixSize);
    firestarter::log::trace() << "wait gemm on device nr. " << DeviceIndex;
    DeviceQueue.wait_and_throw();

    // The number of executed flop for a gemm with two square 'MatrixSize' sized matricies is 2 *
    // ('MatrixSize'^3)
    if (std::is_same_v<FloatingPointType, float>) {
      ExecutedFlop.SingleFlop += 2 * MatrixSize * MatrixSize * MatrixSize;
    } else if (std::is_same_v<FloatingPointType, double>) {
      ExecutedFlop.DoubleFlop += 2 * MatrixSize * MatrixSize * MatrixSize;
    }
  }
}

} // namespace

OneAPI::OneAPI(const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble,
               unsigned MatrixSize, int Gpus) {
  std::condition_variable WaitForInitCv;
  std::mutex WaitForInitCvMutex;
  bool InitDone = false;

  std::thread T(OneAPI::initGpus, std::ref(ExecutedFlop), std::ref(WaitForInitCv), std::ref(WaitForInitCvMutex),
                std::ref(InitDone), std::cref(LoadVar), UseFloat, UseDouble, MatrixSize, Gpus);
  InitThread = std::move(T);

  std::unique_lock<std::mutex> Lk(WaitForInitCvMutex);
  // wait for gpus to initialize
  WaitForInitCv.wait(Lk, [&InitDone] { return InitDone; });
}

void OneAPI::initGpus(GpuFlop& ExecutedFlop, std::condition_variable& WaitForInitCv,
                      std::mutex& WaitForInitCvMutex, bool& InitDone,
                      const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat,
                      bool UseDouble, unsigned MatrixSize, int Gpus) {
  std::condition_variable GpuThreadsWaitForInitCv;
  std::mutex GpuThreadsWaitForInitCvMutex;
  std::vector<std::thread> GpuThreads;

  if (Gpus != 0) {
    auto Platforms = sycl::platform::get_platforms();

    if (Platforms.empty()) {
      std::cerr << "No SYCL platforms found." << std::endl;
      return;
    }

    // Choose a platform based on specific criteria (e.g., device type)
    // TODO(Issue #75): We may select the incorrect platform with gpu devices of the wrong vendor/type.
    auto DevCount = 0;
    for (const auto& Platform : Platforms) {
      auto Devices = Platform.get_devices();
      DevCount = 0;
      for (const auto& Device : Devices) {
        if (Device.is_gpu()) { // Choose GPU, you can use other criteria
          DevCount++;
        }
      }
    }

    if (DevCount) {
      std::atomic<int> InitCount = 0;
      int UseDoubleConverted;

      if (UseFloat) {
        UseDoubleConverted = 0;
      } else if (UseDouble) {
        UseDoubleConverted = 1;
      } else {
        UseDoubleConverted = 2;
      }

      firestarter::log::info()
#ifdef _WIN32
          << "\n  The Task Manager might show a low GPU utilization."
#endif
          << "\n  graphics processor characteristics:";

      // use all GPUs if the user gave no information about use_device
      if (Gpus < 0) {
        Gpus = DevCount;
      }

      if (Gpus > DevCount) {
        firestarter::log::warn() << "You requested more OneAPI devices than available. "
                                    "Maybe you set OneAPI_VISIBLE_DEVICES?";
        firestarter::log::warn() << "FIRESTARTER will use " << DevCount << " of the requested " << Gpus
                                 << " OneAPI device(s)";
        Gpus = DevCount;
      }

      {
        const std::lock_guard<std::mutex> Lk(GpuThreadsWaitForInitCvMutex);

        for (int I = 0; I < Gpus; ++I) {
          const auto Precision = getPrecision(I, UseDoubleConverted);
          if (Precision == -1) {
            firestarter::log::warn() << "This should not have happened. Could not get precision via SYCL.";
          }
          void (*LoadFunc)(GpuFlop&, std::condition_variable&, std::mutex&, int, std::atomic<int>&,
                           const volatile firestarter::LoadThreadWorkType&, unsigned) =
              Precision ? createLoad<double> : createLoad<float>;

          std::thread T(LoadFunc, std::ref(ExecutedFlop), std::ref(GpuThreadsWaitForInitCv),
                        std::ref(GpuThreadsWaitForInitCvMutex), I, std::ref(InitCount), std::cref(LoadVar), MatrixSize);
          GpuThreads.emplace_back(std::move(T));
        }
      }

      {
        std::unique_lock<std::mutex> Lk(GpuThreadsWaitForInitCvMutex);
        // wait for all threads to initialize
        GpuThreadsWaitForInitCv.wait(Lk, [&] { return InitCount == Gpus; });
      }
    } else {
      firestarter::log::info() << "    - No OneAPI"
                               << " devices. Just stressing CPU(s). Maybe use "
                                  "FIRESTARTER instead of FIRESTARTER_OneAPI?";
    }
  } else {
    firestarter::log::info() << "    --gpus 0 is set. Just stressing CPU(s). Maybe use "
                                "FIRESTARTER instead of FIRESTARTER_OneAPI?";
  }

  {
    const std::lock_guard<std::mutex> Lk(WaitForInitCvMutex);
    InitDone = true;
  }
  // notify that init is done
  WaitForInitCv.notify_all();

  /* join computation threads */
  for (auto& Thread : GpuThreads) {
    Thread.join();
  }
}

} // namespace firestarter::oneapi
