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
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

/* CUDA error checking based on CudaWrapper.h
 * https://github.com/ashwin/gDel3D/blob/master/GDelFlipping/src/gDel3D/GPU/CudaWrapper.h
 *
 * inspired by gpu_burn
 * http://wili.cc/blog/gpu-burn.html
 *****************************************************************************/

#include <algorithm>
#include <atomic>
#include <firestarter/Cuda/Cuda.hpp>
#include <firestarter/Cuda/CudaHipCompat.hpp>
#include <firestarter/LoadWorkerData.hpp>
#include <firestarter/Logging/Log.hpp>
#include <type_traits>

namespace firestarter::cuda {

constexpr const int Seed = 123;

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

/// Convert the UseDouble input (0 -> single precision, 1 -> double precision, 2 -> automatic) to either 0 or 1 for
/// float or double respectively. For CUDART_VERSION at least equal 8000 and automatic selection we check if the card a
/// singleToDoublePrecisionPerfRatio bigger than 3 and select float in this case otherwise double. In all other cases
/// automatic results in double.
/// \arg UseDouble The input that specifies either single precision, double precision or automatic selection.
/// \arg Properties The device properties.
/// \return The selected precision, either 0 or 1 for float or double respectively.
static int get_precision(int UseDouble, const compat::DeviceProperties& Properties) {
#if (CUDART_VERSION >= 8000)
  // read precision ratio (dp/sp) of GPU to choose the right variant for maximum
  // workload
  if (UseDouble == 2 && Properties.singleToDoublePrecisionPerfRatio > 3) {
    return 0;
  } else if (UseDouble) {
    return 1;
  } else {
    return 0;
  }
#else
  // as precision ratio is not supported return default/user input value
  (void)Properties;

  if (UseDouble) {
    return 1;
  } else {
    return 0;
  }
#endif
}

static int get_precision(int device_index, int useDouble) {
  size_t memory_avail, memory_total;
  compat::DeviceProperties properties;

  auto stream_or_context = compat::createContextOrStream(device_index);

  compat::accellSafeCall(compat::memGetInfo<>(memory_avail, memory_total), __FILE__, __LINE__, device_index);
  compat::accellSafeCall(compat::getDeviceProperties<>(properties, device_index), __FILE__, __LINE__, device_index);

  useDouble = get_precision(useDouble, properties);

  bool DoubleNotSupported =
#ifdef FIRESTARTER_BUILD_CUDA
      properties.major <= 1 && properties.minor <= 2;
#elif defined(FIRESTARTER_BUILD_HIP)
      !properties.hasDoubles;
#else
      true;
#endif

  // we check for double precision support on the GPU and print errormsg, when
  // the user wants to compute DP on a SP-only-Card.
  if (useDouble && DoubleNotSupported) {
    std::stringstream ss;
    ss << compat::AccelleratorString << " GPU " << device_index << ": " << properties.name << " ";

    firestarter::log::error() << ss.str() << "Doesn't support double precision.\n"
                              << ss.str() << "Compute Capability: " << properties.major << "." << properties.minor
                              << ". Requiered for double precision: >=1.3\n"
                              << ss.str() << "Stressing with single precision instead. Maybe use -f parameter.";

    useDouble = 0;
  }

  compat::accellSafeCall(compat::destroyContextOrStream<>(stream_or_context), __FILE__, __LINE__, device_index);

  return useDouble;
}

// GPU index. Used to pin this thread to the GPU.
template <typename FloatingPointType>
static void create_load(std::condition_variable& waitForInitCv, std::mutex& waitForInitCvMutex, int device_index,
                        std::atomic<int>& initCount, const volatile firestarter::LoadThreadWorkType& LoadVar,
                        int matrixSize) {
  static_assert(std::is_same_v<FloatingPointType, float> || std::is_same_v<FloatingPointType, double>,
                "create_load<FloatingPointType>: Template argument must be either float or double");

  int iterations, i;

  firestarter::log::trace() << "Starting " << compat::AccelleratorString << " with given matrix size " << matrixSize;

  size_t size_use = 0;
  if (matrixSize > 0) {
    size_use = matrixSize;
  }

  size_t use_bytes, memory_size;
  compat::DeviceProperties properties;
  compat::BlasHandle blas;
  // reserving the GPU and initializing cublas

  auto stream_or_context = compat::createContextOrStream(device_index);

  firestarter::log::trace() << "Create " << compat::AccelleratorString << " Blas on device nr. " << device_index;
  compat::accellSafeCall(compat::blasCreate<>(blas), __FILE__, __LINE__, device_index);

  firestarter::log::trace() << "Get " << compat::AccelleratorString << " device properties (e.g., support for double)"
                            << " on device nr. " << device_index;
  compat::accellSafeCall(compat::getDeviceProperties<>(properties, device_index), __FILE__, __LINE__, device_index);

  // getting information about the GPU memory
  size_t memory_avail, memory_total;
  compat::accellSafeCall(compat::memGetInfo<>(memory_avail, memory_total), __FILE__, __LINE__, device_index);
  firestarter::log::trace() << "Get " << compat::AccelleratorString << " Memory info on device nr. " << device_index
                            << ": " << memory_avail << " B avail. from " << memory_total << " B total";

  // defining memory pointers
  compat::DevicePtr<FloatingPointType> a_data_ptr;
  compat::DevicePtr<FloatingPointType> b_data_ptr;
  compat::DevicePtr<FloatingPointType> c_data_ptr;

  // check if the user has not set a matrix OR has set a too big matrixsite and
  // if this is true: set a good matrixsize
  if (!size_use || ((size_use * size_use * sizeof(FloatingPointType) * 3 > memory_avail))) {
    size_use = round_up((int)(0.8 * sqrt(((memory_avail) / (sizeof(FloatingPointType) * 3)))),
                        1024); // a multiple of 1024 works always well
  }
  firestarter::log::trace() << "Set " << compat::AccelleratorString << " matrix size: " << matrixSize;
  use_bytes = (size_t)((FloatingPointType)memory_avail);
  memory_size = sizeof(FloatingPointType) * size_use * size_use;
  iterations = (use_bytes - 2 * memory_size) / memory_size; // = 1;

  firestarter::log::trace() << "Allocating " << compat::AccelleratorString << " memory on device nr. " << device_index;

  // allocating memory on the GPU
  compat::accellSafeCall(compat::malloc<>(a_data_ptr, memory_size), __FILE__, __LINE__, device_index);
  compat::accellSafeCall(compat::malloc<>(b_data_ptr, memory_size), __FILE__, __LINE__, device_index);
  compat::accellSafeCall(compat::malloc<>(c_data_ptr, iterations * memory_size), __FILE__, __LINE__, device_index);

  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << device_index
                            << ". A: " << a_data_ptr << "(Size: " << memory_size << "B)"
                            << "\n";

  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << device_index
                            << ". B: " << b_data_ptr << "(Size: " << memory_size << "B)"
                            << "\n";
  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << device_index
                            << ". C: " << c_data_ptr << "(Size: " << iterations * memory_size << "B)"
                            << "\n";

  firestarter::log::trace() << "Initializing " << compat::AccelleratorString << " matrices a, b on device nr. "
                            << device_index << ". Using " << size_use * size_use << " elements of size "
                            << sizeof(FloatingPointType) << " Byte";
  // initialize matrix A and B on the GPU with random values
  {
    compat::RandGenerator random_gen;
    compat::accellSafeCall(compat::randCreateGeneratorPseudoRandom<>(random_gen), __FILE__, __LINE__, device_index);
    compat::accellSafeCall(compat::randSetPseudoRandomGeneratorSeed<>(random_gen, Seed), __FILE__, __LINE__,
                           device_index);
    compat::accellSafeCall(compat::generateUniform<>(random_gen, a_data_ptr, size_use * size_use), __FILE__, __LINE__,
                           device_index);
    compat::accellSafeCall(compat::generateUniform<>(random_gen, b_data_ptr, size_use * size_use), __FILE__, __LINE__,
                           device_index);
    compat::accellSafeCall(compat::randDestroyGenerator<>(random_gen), __FILE__, __LINE__, device_index);
  }

  // initialize c_data_ptr with copies of A
  for (i = 0; i < iterations; i++) {
    auto DestinationPtr =
        c_data_ptr + (size_t)(i * size_use * size_use * (float)sizeof(FloatingPointType) / (float)sizeof(c_data_ptr));
    firestarter::log::trace() << "Initializing " << compat::AccelleratorString << " matrix c-" << i << " by copying "
                              << memory_size << " byte from " << a_data_ptr << " to " << DestinationPtr << "\n";
    compat::accellSafeCall(compat::memcpyDtoD<>(DestinationPtr, a_data_ptr, memory_size), __FILE__, __LINE__,
                           device_index);
  }

  // save gpuvar->init_count and sys.out
  {
    std::lock_guard<std::mutex> lk(waitForInitCvMutex);

    auto ToMiB = [](const size_t Val) { return Val / 1024 / 1024; };
    firestarter::log::info() << "   GPU " << device_index << "\n"
                             << "    name:           " << properties.name << "\n"
                             << "    memory:         " << ToMB(memory_avail) << "/" << ToMB(memory_total)
                             << " MiB available (using " << ToMB(use_bytes) << " MiB)\n"
                             << "    matrix size:    " << size_use << "\n"
                             << "    used precision: "
                             << ((sizeof(FloatingPointType) == sizeof(double)) ? "double" : "single");

    initCount++;
  }
  waitForInitCv.notify_all();

  const FloatingPointType alpha = 1.0;
  const FloatingPointType beta = 0.0;

  int size_use_i = size_use;
  // actual stress begins here
  while (LoadVar != LoadThreadWorkType::LOAD_STOP) {
    for (i = 0; i < iterations; i++) {
      compat::accellSafeCall(compat::gemm<FloatingPointType>(
                                 blas, compat::BlasOperation::BLAS_OP_N, compat::BlasOperation::BLAS_OP_N, size_use_i,
                                 size_use_i, size_use_i, &alpha, a_data_ptr, size_use_i, b_data_ptr, size_use_i, &beta,
                                 c_data_ptr + i * size_use * size_use, size_use_i),
                             __FILE__, __LINE__, device_index);
      compat::accellSafeCall(compat::deviceSynchronize<>(), __FILE__, __LINE__, device_index);
    }
  }

  compat::accellSafeCall(compat::free<>(a_data_ptr), __FILE__, __LINE__, device_index);
  compat::accellSafeCall(compat::free<>(b_data_ptr), __FILE__, __LINE__, device_index);
  compat::accellSafeCall(compat::free<>(c_data_ptr), __FILE__, __LINE__, device_index);

  compat::accellSafeCall(compat::blasDestroy<>(blas), __FILE__, __LINE__, device_index);

  compat::accellSafeCall(compat::destroyContextOrStream<>(stream_or_context), __FILE__, __LINE__, device_index);
}

Cuda::Cuda(const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble, unsigned MatrixSize,
           int Gpus) {
  std::condition_variable WaitForInitCv;
  std::mutex WaitForInitCvMutex;

  std::thread T(Cuda::initGpus, std::ref(WaitForInitCv), std::cref(LoadVar), UseFloat, UseDouble, MatrixSize, Gpus);
  InitThread = std::move(T);

  const std::unique_lock<std::mutex> Lk(WaitForInitCvMutex);
  // wait for gpus to initialize
  WaitForInitCv.wait(Lk);
}

void Cuda::initGpus(std::condition_variable& WaitForInitCv, const volatile firestarter::LoadThreadWorkType& LoadVar,
                    bool UseFloat, bool UseDouble, unsigned MatrixSize, int Gpus) {
  std::condition_variable GpuThreadsWaitForInitCv;
  std::mutex GpuThreadsWaitForInitCvMutex;
  std::vector<std::thread> GpuThreads;

  if (Gpus) {
    compat::accellSafeCall(compat::init<>(0), __FILE__, __LINE__);

    int DevCount;
    compat::accellSafeCall(compat::getDeviceCount<>(DevCount), __FILE__, __LINE__);

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
        firestarter::log::warn() << "You requested more " << compat::AccelleratorString
                                 << " devices than available. "
                                    "Maybe you set "
                                 << compat::AccelleratorString << "_VISIBLE_DEVICES?";
        firestarter::log::warn() << "FIRESTARTER will use " << DevCount << " of the requested " << Gpus << " "
                                 << compat::AccelleratorString << " device(s)";
        Gpus = DevCount;
      }

      {
        std::lock_guard<std::mutex> Lk(GpuThreadsWaitForInitCvMutex);

        for (int I = 0; I < Gpus; ++I) {
          // if there's a GPU in the system without Double Precision support, we
          // have to correct this.
          int Precision = get_precision(I, UseDoubleConverted);
          void (*LoadFunc)(std::condition_variable&, std::mutex&, int, std::atomic<int>&, volatile uint64_t*, int) =
              Precision ? create_load<double> : create_load<float>;

          std::thread t(LoadFunc, std::ref(GpuThreadsWaitForInitCv), std::ref(GpuThreadsWaitForInitCvMutex), I,
                        std::ref(InitCount), std::cref(LoadVar), (int)MatrixSize);
        }
      }

      {
        std::unique_lock<std::mutex> Lk(GpuThreadsWaitForInitCvMutex);
        // wait for all threads to initialize
        GpuThreadsWaitForInitCv.wait(Lk, [&] { return InitCount == Gpus; });
      }
    } else {
      firestarter::log::info() << "    - No " << compat::AccelleratorString
                               << " devices. Just stressing CPU(s). Maybe use "
                                  "FIRESTARTER instead of FIRESTARTER_"
                               << compat::AccelleratorString << "?";
    }
  } else {
    firestarter::log::info() << "    --gpus 0 is set. Just stressing CPU(s). Maybe use "
                                "FIRESTARTER instead of FIRESTARTER_"
                             << compat::AccelleratorString << "?";
  }

  // notify that init is done
  WaitForInitCv.notify_all();

  /* join computation threads */
  for (auto& Thread : GpuThreads) {
    Thread.join();
  }
}

} // namespace firestarter::cuda