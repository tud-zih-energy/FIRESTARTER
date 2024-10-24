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

#include <firestarter/LoadWorkerData.hpp>
#include <firestarter/Logging/Log.hpp>
#include <firestarter/OneAPI/OneAPI.hpp>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <type_traits>

namespace firestarter::oneapi {

/// Helper function to generate random floating point values between 0 and 1 in an array.
/// \targ FloatingPointType The type of floating point value of the array. Either float or double.
/// \arg NumberOfElems The number of elements of the array.
/// \arg Array The array of floating point values which should be initilized with random data between 0 and 1.
template <typename FloatingPointType> void fillArrayWithRandomFloats(size_t NumberOfElems, FloatingPointType* Array) {
  static_assert(std::is_same_v<FloatingPointType, float> || std::is_same_v<FloatingPointType, double>,
                "fillArrayWithRandomFloats<FloatingPointType>: Template argument must be either float or double");

  for (size_t i = 0; i < NumberOfElems; i++) {
    Array[i] = static_cast<FloatingPointType>(std::rand()) / RAND_MAX;
  }
}

template <typename T> void replicate_data(sycl::queue& Q, T* dst, size_t dst_elems, const T* src, size_t src_elems) {
  firestarter::log::trace() << "replicate_data " << dst_elems << " elements from " << src << " to " << dst;
  while (dst_elems > 0) {
    auto copy_elems = std::min(dst_elems, src_elems);
    Q.copy(src, dst, copy_elems);
    dst += copy_elems;
    dst_elems -= copy_elems;
  }
  Q.wait();
}

static int get_precision(int device_index, int useDouble) {

  firestarter::log::trace() << "Checking useDouble " << useDouble;

  if (!useDouble) {
    return 0;
  }

  int supports_double = 0;

  auto platforms = sycl::platform::get_platforms();

  if (platforms.empty()) {
    firestarter::log::warn() << "No SYCL platforms found.";
    return -1;
  }
  // Choose a platform based on specific criteria (e.g., device type)
  // TODO(Issue #75): We may select the incorrect platform with gpu devices of the wrong vendor/type.
  sycl::platform chosenPlatform;
  auto nr_gpus = 0;
  for (const auto& platform : platforms) {
    firestarter::log::trace() << "Checking SYCL platform " << platform.get_info<sycl::info::platform::name>();
    auto devices = platform.get_devices();
    nr_gpus = 0;
    for (const auto& device : devices) {
      firestarter::log::trace() << "Checking SYCL device " << device.get_info<sycl::info::device::name>();
      if (device.is_gpu()) { // Choose GPU, you can use other criteria
        firestarter::log::trace() << " ... is GPU";
        chosenPlatform = platform;
        nr_gpus++;
      }
    }
  }

  if (!nr_gpus) {
    firestarter::log::warn() << "No suitable platform with GPU found.";
    return -1;
  }
  // Get a list of devices for the chosen platform

  firestarter::log::trace() << "Get support for double"
                            << " on device nr. " << device_index;
  auto devices = chosenPlatform.get_devices();
  if (devices[device_index].has(sycl::aspect::fp64))
    supports_double = 1;

  return supports_double;
}

static int round_up(int num_to_round, int multiple) {
  if (multiple == 0) {
    return num_to_round;
  }

  int remainder = num_to_round % multiple;
  if (remainder == 0) {
    return num_to_round;
  }

  return num_to_round + multiple - remainder;
}

// GPU index. Used to pin this thread to the GPU.
// The main difference to the CUDA/HIP version is that we do not run multiple iterations of C=A*B, just one single
// iteration.
template <typename T>
static void create_load(std::condition_variable& waitForInitCv, std::mutex& waitForInitCvMutex, int device_index,
                        std::atomic<int>& initCount, const volatile firestarter::LoadThreadWorkType& LoadVar,
                        unsigned matrixSize) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "create_load<T>: Template argument T must be either float or double");

  firestarter::log::trace() << "Starting OneAPI with given matrix size " << matrixSize;

  size_t size_use = 0;
  if (matrixSize > 0) {
    size_use = matrixSize;
  }

  size_t use_bytes;

  // reserving the GPU and initializing

  firestarter::log::trace() << "Getting device nr. " << device_index;

  auto platforms = sycl::platform::get_platforms();

  if (platforms.empty()) {
    firestarter::log::warn() << "No SYCL platforms found.";
    return;
  }

  // Choose a platform based on specific criteria (e.g., device type)
  sycl::platform chosenPlatform;
  auto nr_gpus = 0;
  for (const auto& platform : platforms) {
    auto devices = platform.get_devices();
    nr_gpus = 0;
    for (const auto& device : devices) {
      if (device.is_gpu()) { // Choose GPU, you can use other criteria
        chosenPlatform = platform;
        nr_gpus++;
      }
    }
  }

  if (!nr_gpus) {
    firestarter::log::warn() << "No suitable platform with GPU found.";
    return;
  }

  // Get a list of devices for the chosen platform
  auto devices = chosenPlatform.get_devices();

  firestarter::log::trace() << "Creating SYCL queue for computation on device nr. " << device_index;
  auto chosenDevice = devices[device_index];
  sycl::queue device_queue(chosenDevice);

  firestarter::log::trace() << "Get memory size on device nr. " << device_index;

  // getting information about the GPU memory
  size_t memory_total = devices[device_index].get_info<sycl::info::device::global_mem_size>();

  firestarter::log::trace() << "Get Memory info on device nr. " << device_index << ": has " << memory_total
                            << " B global memory";

  // check if the user has not set a matrix OR has set a too big matrixsite and
  // if this is true: set a good matrixsize
  if (!size_use || ((size_use * size_use * sizeof(T) * 3 > memory_total))) {
    size_use = round_up((int)(0.8 * sqrt(((memory_total) / (sizeof(T) * 3)))),
                        1024); // a multiple of 1024 works always well
  }

  firestarter::log::trace() << "Set OneAPI matrix size in B: " << size_use;
  use_bytes = sizeof(T) * size_use * size_use * 3;

  /* Allocate A/B/C matrices */

  firestarter::log::trace() << "Allocating memory on device nr. " << device_index;
  auto* A = sycl::malloc_device<T>(size_use * size_use, device_queue);
  auto* B = sycl::malloc_device<T>(size_use * size_use, device_queue);
  auto* C = sycl::malloc_device<T>(size_use * size_use, device_queue);

  /* Create 64 MB random data on Host */
  constexpr int rd_size = 1024 * 1024 * 64;
  auto* random_data = malloc_host<T>(rd_size, device_queue);
  fillArrayWithRandomFloats(rd_size, random_data);

  firestarter::log::trace() << "Copy memory to device nr. " << device_index;
  /* fill A and B with random data */
  replicate_data(device_queue, A, size_use * size_use, random_data, rd_size);
  replicate_data(device_queue, B, size_use * size_use, random_data, rd_size);

  {
    std::lock_guard<std::mutex> lk(waitForInitCvMutex);

#define TO_MB(x) (unsigned long)(x / 1024 / 1024)
    firestarter::log::info() << "   GPU " << device_index << "\n"
                             << "    name:           " << devices[device_index].get_info<sycl::info::device::name>()
                             << "\n"
                             << "    memory:         " << TO_MB(memory_total) << " MiB total (using "
                             << TO_MB(use_bytes) << " MiB)\n"
                             << "    matrix size:    " << size_use << "\n"
                             << "    used precision: " << ((sizeof(T) == sizeof(double)) ? "double" : "single");
#undef TO_MB

    initCount++;
  }
  waitForInitCv.notify_all();

  firestarter::log::trace() << "Run gemm on device nr. " << device_index;
  /* With this, we could run multiple gemms ...*/
  /*  auto run_gemms = [=, &device_queue](int runs) -> double {
        using namespace oneapi::mkl;
        for (int i = 0; i < runs; i++)

        return runs;
    };
  */
  while (LoadVar != firestarter::LoadThreadWorkType::LoadStop) {
    firestarter::log::trace() << "Run gemm on device nr. " << device_index;
    ::oneapi::mkl::blas::gemm(device_queue, ::oneapi::mkl::transpose::N, ::oneapi::mkl::transpose::N, size_use,
                              size_use, size_use, 1, A, size_use, B, size_use, 0, C, size_use);
    firestarter::log::trace() << "wait gemm on device nr. " << device_index;
    device_queue.wait_and_throw();
  }
}

OneAPI::OneAPI(const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble,
               unsigned MatrixSize, int Gpus) {
  std::condition_variable WaitForInitCv;
  std::mutex WaitForInitCvMutex;

  std::thread T(OneAPI::initGpus, std::ref(WaitForInitCv), std::cref(LoadVar), UseFloat, UseDouble, MatrixSize, Gpus);
  InitThread = std::move(T);

  std::unique_lock<std::mutex> Lk(WaitForInitCvMutex);
  // wait for gpus to initialize
  WaitForInitCv.wait(Lk);
}

void OneAPI::initGpus(std::condition_variable& WaitForInitCv, const volatile firestarter::LoadThreadWorkType& LoadVar,
                      bool UseFloat, bool UseDouble, unsigned MatrixSize, int Gpus) {
  std::condition_variable GpuThreadsWaitForInitCv;
  std::mutex GpuThreadsWaitForInitCvMutex;
  std::vector<std::thread> GpuThreads;

  if (Gpus) {
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
          const auto Precision = get_precision(I, UseDoubleConverted);
          if (Precision == -1) {
            firestarter::log::warn() << "This should not have happened. Could not get precision via SYCL.";
          }
          void (*LoadFunc)(std::condition_variable&, std::mutex&, int, std::atomic<int>&,
                           const volatile firestarter::LoadThreadWorkType&, unsigned) =
              Precision ? create_load<double> : create_load<float>;

          std::thread T(LoadFunc, std::ref(GpuThreadsWaitForInitCv), std::ref(GpuThreadsWaitForInitCvMutex), I,
                        std::ref(InitCount), std::cref(LoadVar), MatrixSize);
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

  // notify that init is done
  WaitForInitCv.notify_all();

  /* join computation threads */
  for (auto& Thread : GpuThreads) {
    Thread.join();
  }
}

} // namespace firestarter::oneapi