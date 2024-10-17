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

#include <atomic>
#include <firestarter/Cuda/Cuda.hpp>
#include <firestarter/Cuda/CudaHipCompat.hpp>
#include <firestarter/LoadWorkerData.hpp>
#include <firestarter/Logging/Log.hpp>
#include <type_traits>

namespace firestarter::cuda {

constexpr const int Seed = 123;

static auto roundUp(int NumToRound, int Multiple) -> int {
  if (Multiple == 0) {
    return NumToRound;
  }

  const int Remainder = NumToRound % Multiple;
  if (Remainder == 0) {
    return NumToRound;
  }

  return NumToRound + Multiple - Remainder;
}

/// Convert the UseDouble input (0 -> single precision, 1 -> double precision, 2 -> automatic) to either 0 or 1 for
/// float or double respectively. For CUDART_VERSION at least equal 8000 and automatic selection we check if the card a
/// singleToDoublePrecisionPerfRatio bigger than 3 and select float in this case otherwise double. In all other cases
/// automatic results in double.
/// \arg UseDouble The input that specifies either single precision, double precision or automatic selection.
/// \arg Properties The device properties.
/// \return The selected precision, either 0 or 1 for float or double respectively.
static auto getPrecision(int UseDouble, const compat::DeviceProperties& Properties) -> int {
#if (CUDART_VERSION >= 8000)
  // read precision ratio (dp/sp) of GPU to choose the right variant for maximum
  // workload
  if (UseDouble == 2 && Properties.singleToDoublePrecisionPerfRatio > 3) {
    return 0;
  }
  if (UseDouble) {
    return 1;
  }
  return 0;
#else
  // as precision ratio is not supported return default/user input value
  (void)Properties;

  if (UseDouble) {
    return 1;
  }
  return 0;

#endif
}

static auto getPrecision(int DeviceIndex, int UseDouble) -> int {
  size_t MemoryAvail;
  size_t MemoryTotal;
  compat::DeviceProperties Properties;

  auto StreamOrContext = compat::createContextOrStream(DeviceIndex);

  compat::accellSafeCall(compat::memGetInfo(MemoryAvail, MemoryTotal), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::getDeviceProperties(Properties, DeviceIndex), __FILE__, __LINE__, DeviceIndex);

  UseDouble = getPrecision(UseDouble, Properties);

  bool DoubleNotSupported =
#ifdef FIRESTARTER_BUILD_CUDA
      Properties.major <= 1 && Properties.minor <= 2;
#elif defined(FIRESTARTER_BUILD_HIP)
      !Properties.hasDoubles;
#else
      true;
#endif

  // we check for double precision support on the GPU and print errormsg, when
  // the user wants to compute DP on a SP-only-Card.
  if (UseDouble && DoubleNotSupported) {
    std::stringstream Ss;
    Ss << compat::AccelleratorString << " GPU " << DeviceIndex << ": " << Properties.name << " ";

    firestarter::log::error() << Ss.str() << "Doesn't support double precision.\n"
                              << Ss.str() << "Compute Capability: " << Properties.major << "." << Properties.minor
                              << ". Requiered for double precision: >=1.3\n"
                              << Ss.str() << "Stressing with single precision instead. Maybe use -f parameter.";

    UseDouble = 0;
  }

  compat::accellSafeCall(compat::destroyContextOrStream(StreamOrContext), __FILE__, __LINE__, DeviceIndex);

  return UseDouble;
}

// GPU index. Used to pin this thread to the GPU.
template <typename FloatingPointType>
static void createLoad(std::condition_variable& WaitForInitCv, std::mutex& WaitForInitCvMutex, int DeviceIndex,
                       std::atomic<int>& InitCount, const volatile firestarter::LoadThreadWorkType& LoadVar,
                       unsigned MatrixSize) {
  static_assert(std::is_same_v<FloatingPointType, float> || std::is_same_v<FloatingPointType, double>,
                "create_load<FloatingPointType>: Template argument must be either float or double");

  firestarter::log::trace() << "Starting " << compat::AccelleratorString << " with given matrix size " << MatrixSize;

  size_t SizeUse = 0;
  if (MatrixSize > 0) {
    SizeUse = MatrixSize;
  }

  size_t UseBytes;
  size_t MemorySize;
  compat::DeviceProperties Properties;
  compat::BlasHandle Blas;
  // reserving the GPU and initializing cublas

  auto StreamOrContext = compat::createContextOrStream(DeviceIndex);

  firestarter::log::trace() << "Create " << compat::AccelleratorString << " Blas on device nr. " << DeviceIndex;
  compat::accellSafeCall(compat::blasCreate(Blas), __FILE__, __LINE__, DeviceIndex);

  firestarter::log::trace() << "Get " << compat::AccelleratorString << " device properties (e.g., support for double)"
                            << " on device nr. " << DeviceIndex;
  compat::accellSafeCall(compat::getDeviceProperties(Properties, DeviceIndex), __FILE__, __LINE__, DeviceIndex);

  // getting information about the GPU memory
  size_t MemoryAvail;
  size_t MemoryTotal;
  compat::accellSafeCall(compat::memGetInfo(MemoryAvail, MemoryTotal), __FILE__, __LINE__, DeviceIndex);
  firestarter::log::trace() << "Get " << compat::AccelleratorString << " emory info on device nr. " << DeviceIndex
                            << ": " << MemoryAvail << " B avail. from " << MemoryTotal << " B total";

  // defining memory pointers
  compat::DevicePtr<FloatingPointType> ADataPtr;
  compat::DevicePtr<FloatingPointType> BDataPtr;
  compat::DevicePtr<FloatingPointType> CDataPtr;

  // check if the user has not set a matrix OR has set a too big matrixsite and
  // if this is true: set a good matrixsize
  if (!SizeUse || ((SizeUse * SizeUse * sizeof(FloatingPointType) * 3 > MemoryAvail))) {
    SizeUse = roundUp((int)(0.8 * sqrt(((MemoryAvail) / (sizeof(FloatingPointType) * 3)))),
                      1024); // a multiple of 1024 works always well
  }
  firestarter::log::trace() << "Set " << compat::AccelleratorString << " matrix size: " << MatrixSize;
  UseBytes = (size_t)((FloatingPointType)MemoryAvail);
  MemorySize = sizeof(FloatingPointType) * SizeUse * SizeUse;
  int Iterations = (UseBytes - 2 * MemorySize) / MemorySize; // = 1;

  firestarter::log::trace() << "Allocating " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex;

  // allocating memory on the GPU
  compat::accellSafeCall(compat::malloc<FloatingPointType>(ADataPtr, MemorySize), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::malloc<FloatingPointType>(BDataPtr, MemorySize), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::malloc<FloatingPointType>(CDataPtr, Iterations * MemorySize), __FILE__, __LINE__,
                         DeviceIndex);

  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex
                            << ". A: " << ADataPtr << "(Size: " << MemorySize << "B)"
                            << "\n";

  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex
                            << ". B: " << BDataPtr << "(Size: " << MemorySize << "B)"
                            << "\n";
  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex
                            << ". C: " << CDataPtr << "(Size: " << Iterations * MemorySize << "B)"
                            << "\n";

  firestarter::log::trace() << "Initializing " << compat::AccelleratorString << " matrices a, b on device nr. "
                            << DeviceIndex << ". Using " << SizeUse * SizeUse << " elements of size "
                            << sizeof(FloatingPointType) << " Byte";
  // initialize matrix A and B on the GPU with random values
  {
    compat::RandGenerator RandomGen;
    compat::accellSafeCall(compat::randCreateGeneratorPseudoRandom(RandomGen), __FILE__, __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::randSetPseudoRandomGeneratorSeed(RandomGen, Seed), __FILE__, __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::generateUniform<FloatingPointType>(RandomGen, ADataPtr, SizeUse * SizeUse), __FILE__,
                           __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::generateUniform<FloatingPointType>(RandomGen, BDataPtr, SizeUse * SizeUse), __FILE__,
                           __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::randDestroyGenerator(RandomGen), __FILE__, __LINE__, DeviceIndex);
  }

  // initialize c_data_ptr with copies of A
  for (int I = 0; I < Iterations; I++) {
    auto DestinationPtr =
        CDataPtr + (size_t)(I * SizeUse * SizeUse * (float)sizeof(FloatingPointType) / (float)sizeof(CDataPtr));
    firestarter::log::trace() << "Initializing " << compat::AccelleratorString << " matrix c-" << I << " by copying "
                              << MemorySize << " byte from " << ADataPtr << " to " << DestinationPtr << "\n";
    compat::accellSafeCall(compat::memcpyDtoD<FloatingPointType>(DestinationPtr, ADataPtr, MemorySize), __FILE__,
                           __LINE__, DeviceIndex);
  }

  // save gpuvar->init_count and sys.out
  {
    const std::lock_guard<std::mutex> Lk(WaitForInitCvMutex);

    auto ToiB = [](const size_t Val) { return Val / 1024 / 1024; };
    firestarter::log::info() << "   GPU " << DeviceIndex << "\n"
                             << "    name:           " << Properties.name << "\n"
                             << "    memory:         " << ToiB(MemoryAvail) << "/" << ToiB(MemoryTotal)
                             << " iB available (using " << ToiB(UseBytes) << " iB)\n"
                             << "    matrix size:    " << SizeUse << "\n"
                             << "    used precision: "
                             << ((sizeof(FloatingPointType) == sizeof(double)) ? "double" : "single");

    InitCount++;
  }
  WaitForInitCv.notify_all();

  const FloatingPointType Alpha = 1.0;
  const FloatingPointType Beta = 0.0;

  const int SizeUseI = SizeUse;
  // actual stress begins here
  while (LoadVar != firestarter::LoadThreadWorkType::LoadStop) {
    for (int I = 0; I < Iterations; I++) {
      compat::accellSafeCall(compat::gemm<FloatingPointType>(Blas, compat::BlasOperation::BLAS_OP_N,
                                                             compat::BlasOperation::BLAS_OP_N, SizeUseI, SizeUseI,
                                                             SizeUseI, Alpha, ADataPtr, SizeUseI, BDataPtr, SizeUseI,
                                                             Beta, CDataPtr + (I * SizeUse * SizeUse), SizeUseI),
                             __FILE__, __LINE__, DeviceIndex);
      compat::accellSafeCall(compat::deviceSynchronize(), __FILE__, __LINE__, DeviceIndex);
    }
  }

  compat::accellSafeCall(compat::free<FloatingPointType>(ADataPtr), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::free<FloatingPointType>(BDataPtr), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::free<FloatingPointType>(CDataPtr), __FILE__, __LINE__, DeviceIndex);

  compat::accellSafeCall(compat::blasDestroy(Blas), __FILE__, __LINE__, DeviceIndex);

  compat::accellSafeCall(compat::destroyContextOrStream(StreamOrContext), __FILE__, __LINE__, DeviceIndex);
}

Cuda::Cuda(const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble, unsigned MatrixSize,
           int Gpus) {
  std::condition_variable WaitForInitCv;
  std::mutex WaitForInitCvMutex;

  std::thread T(Cuda::initGpus, std::ref(WaitForInitCv), std::cref(LoadVar), UseFloat, UseDouble, MatrixSize, Gpus);
  InitThread = std::move(T);

  std::unique_lock<std::mutex> Lk(WaitForInitCvMutex);
  // wait for gpus to initialize
  WaitForInitCv.wait(Lk);
}

void Cuda::initGpus(std::condition_variable& WaitForInitCv, const volatile firestarter::LoadThreadWorkType& LoadVar,
                    bool UseFloat, bool UseDouble, unsigned MatrixSize, int Gpus) {
  std::condition_variable GpuThreadsWaitForInitCv;
  std::mutex GpuThreadsWaitForInitCvMutex;
  std::vector<std::thread> GpuThreads;

  if (Gpus) {
    compat::accellSafeCall(compat::init(0), __FILE__, __LINE__);

    int DevCount;
    compat::accellSafeCall(compat::getDeviceCount(DevCount), __FILE__, __LINE__);

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
        const std::lock_guard<std::mutex> Lk(GpuThreadsWaitForInitCvMutex);

        for (int I = 0; I < Gpus; ++I) {
          // if there's a GPU in the system without Double Precision support, we
          // have to correct this.
          const auto Precision = getPrecision(I, UseDoubleConverted);
          void (*LoadFunc)(std::condition_variable&, std::mutex&, int, std::atomic<int>&,
                           const volatile firestarter::LoadThreadWorkType&, unsigned) =
              Precision ? createLoad<double> : createLoad<float>;

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