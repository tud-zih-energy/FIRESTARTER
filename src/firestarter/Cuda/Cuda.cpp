/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2024 TU Dresden, Center for Information Services and High
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

/******************************************************************************
 * inspired by gpu_burn
 * http://wili.cc/blog/gpu-burn.html
 *****************************************************************************/

#include "firestarter/Cuda/Cuda.hpp"
#include "firestarter/Cuda/CudaHipCompat.hpp"
#include "firestarter/Logging/Log.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

namespace firestarter::cuda {

constexpr const int Seed = 123;

namespace {

template <std::size_t Multiple> auto roundUp(int NumToRound) -> int {
  static_assert(Multiple != 0, "Multiple may not be zero.");

  const int Remainder = NumToRound % Multiple;
  if (Remainder == 0) {
    return NumToRound;
  }

  return NumToRound + Multiple - Remainder;
}

/// Convert the UseDouble input (0 -> single precision, 1 -> double precision, 2 -> automatic) to either 0 or 1 for
/// float or double respectively. For CUDART_VERSION at least equal 8000 and automatic selection we check if the card
/// a singleToDoublePrecisionPerfRatio bigger than 3 and select float in this case otherwise double. In all other
/// cases automatic results in double.
/// \arg UseDouble The input that specifies either single precision, double precision or automatic selection.
/// \arg DeviceIndex The index of the GPU device.
/// \return The selected precision, either 0 or 1 for float or double respectively.
auto selectPrecision(int UseDouble, int DeviceIndex) -> int {
#if (CUDART_VERSION >= 8000)
  // read precision ratio (dp/sp) of GPU to choose the right variant for maximum
  // workload
  if (UseDouble == 2) {
    int SingleToDoublePrecisionPerfRatio{};
    compat::accellSafeCall(compat::getSingleToDoublePrecisionPerfRatio(SingleToDoublePrecisionPerfRatio, DeviceIndex),
                           __FILE__, __LINE__, DeviceIndex);
    if (SingleToDoublePrecisionPerfRatio > 3) {
      return 0;
    }
  }
  if (UseDouble) {
    return 1;
  }
  return 0;
#else
  // as precision ratio is not supported return default/user input value
  (void)DeviceIndex;

  if (UseDouble) {
    return 1;
  }
  return 0;

#endif
}

auto getPrecision(int DeviceIndex, int UseDouble) -> int {
  std::size_t MemoryAvail{};
  std::size_t MemoryTotal{};
  compat::DeviceProperties Properties;

  // NOLINTNEXTLINE(readability-qualified-auto)
  auto StreamOrContext = compat::createContextOrStream(DeviceIndex);

  compat::accellSafeCall(compat::memGetInfo(MemoryAvail, MemoryTotal), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::getDeviceProperties(Properties, DeviceIndex), __FILE__, __LINE__, DeviceIndex);

  UseDouble = selectPrecision(UseDouble, DeviceIndex);

  const bool DoubleNotSupported =
#ifdef FIRESTARTER_BUILD_CUDA
      Properties.major <= 1 && Properties.minor <= 2;
#else
      false;
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

/// Check if a GPU supports half precision matrix multiplications.
/// \arg DeviceIndex The index of the GPU device.
/// \return true if half precision gemm is supported.
auto halfSupported(int DeviceIndex) -> bool {
#ifdef FIRESTARTER_BUILD_CUDA
  compat::DeviceProperties Properties;
  compat::accellSafeCall(compat::getDeviceProperties(Properties, DeviceIndex), __FILE__, __LINE__, DeviceIndex);
  // FP16 arithmetic requires compute capability 5.3 or newer. Tensor cores are available from 7.0.
  return Properties.major > 5 || (Properties.major == 5 && Properties.minor >= 3);
#else
  (void)DeviceIndex;
  return true;
#endif
}

/// Initialize a half precision matrix on the device with uniform random values in [-1, 1). There is no half precision
/// random generator in (cu|hip)rand and we do not want to compile device code, so we generate a tile on the host, copy
/// it to the device and replicate it there. The values are symmetric around zero to keep the FP16 results of large
/// matrix multiplications well inside the FP16 range.
/// \arg DataPtr The device pointer to the matrix.
/// \arg NumElements The number of elements of the matrix.
/// \arg MatrixSeed The seed for the random number generator.
/// \arg DeviceIndex The index of the GPU device.
void initHalfMatrix(compat::Half* DataPtr, std::size_t NumElements, unsigned MatrixSeed, int DeviceIndex) {
  // An odd tile size makes sure that the repeated pattern does not align with the rows of the matrix.
  constexpr std::size_t TileElements = (std::size_t{1} << 24U) - 1;
  const auto NumTileElements = (std::min)(NumElements, TileElements);

  std::mt19937 Generator(MatrixSeed);
  std::uniform_real_distribution<float> Distribution(-1.0F, 1.0F);
  std::vector<compat::Half> Tile(NumTileElements);
  for (auto& Element : Tile) {
    Element = compat::floatToHalf(Distribution(Generator));
  }

  compat::accellSafeCall(compat::memcpyHtoD(DataPtr, Tile.data(), NumTileElements * sizeof(compat::Half)), __FILE__,
                         __LINE__, DeviceIndex);

  // replicate the tile by doubling the initialized region
  std::size_t Filled = NumTileElements;
  while (Filled < NumElements) {
    const auto Chunk = (std::min)(Filled, NumElements - Filled);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    compat::accellSafeCall(compat::memcpyDtoD(DataPtr + Filled, DataPtr, Chunk * sizeof(compat::Half)), __FILE__,
                           __LINE__, DeviceIndex);
    Filled += Chunk;
  }
}

// GPU index. Used to pin this thread to the GPU.
// Size use is one square matrix dim size
template <typename FloatingPointType>
void createLoad(GpuFlop& ExecutedFlop, std::condition_variable& WaitForInitCv, std::mutex& WaitForInitCvMutex,
                int DeviceIndex, std::atomic<int>& InitCount, const volatile firestarter::LoadThreadWorkType& LoadVar,
                uint64_t MatrixSize) {
  static_assert(std::is_same_v<FloatingPointType, float> || std::is_same_v<FloatingPointType, double> ||
                    std::is_same_v<FloatingPointType, compat::Half>,
                "create_load<FloatingPointType>: Template argument must be either float, double or compat::Half");

  constexpr bool IsHalf = std::is_same_v<FloatingPointType, compat::Half>;
  // Half precision gemm uses single precision for the scalars and the accumulation.
  using ScalarType = std::conditional_t<IsHalf, float, FloatingPointType>;

  firestarter::log::trace() << "Starting " << compat::AccelleratorString << " with given matrix size " << MatrixSize;

  compat::DeviceProperties Properties;
  compat::BlasHandle Blas{};
  // reserving the GPU and initializing cublas

  // NOLINTNEXTLINE(readability-qualified-auto)
  auto StreamOrContext = compat::createContextOrStream(DeviceIndex);

  firestarter::log::trace() << "Create " << compat::AccelleratorString << " Blas on device nr. " << DeviceIndex;
  compat::accellSafeCall(compat::blasCreate(Blas), __FILE__, __LINE__, DeviceIndex);

  firestarter::log::trace() << "Get " << compat::AccelleratorString << " device properties (e.g., support for double)"
                            << " on device nr. " << DeviceIndex;
  compat::accellSafeCall(compat::getDeviceProperties(Properties, DeviceIndex), __FILE__, __LINE__, DeviceIndex);

  // getting information about the GPU memory
  std::size_t MemoryAvail{};
  std::size_t MemoryTotal{};
  compat::accellSafeCall(compat::memGetInfo(MemoryAvail, MemoryTotal), __FILE__, __LINE__, DeviceIndex);
  firestarter::log::trace() << "Get " << compat::AccelleratorString << " emory info on device nr. " << DeviceIndex
                            << ": " << MemoryAvail << " B avail. from " << MemoryTotal << " B total";

  // Defining memory pointers. ADataPtr and BDataPtr will point to a square matrix. CDataPtr may be one or multiple
  // square matrices.
  FloatingPointType* ADataPtr{};
  FloatingPointType* BDataPtr{};
  FloatingPointType* CDataPtr{};

  // If the matrix size is not set or three square matricies with dim size of SizeUse do not fit into the available
  // memory, select the size so that 3 square matricies will fit into the available device memory where the dim size
  // is a multiple of 1024. There may be edge cases with small device memory that results in matricies that are not
  // multiples of 1024.
  std::size_t MemorySize = sizeof(FloatingPointType) * MatrixSize * MatrixSize;
  if (!MatrixSize || (MemorySize * 3 > MemoryAvail)) {
    // a multiple of 1024 works always well
    MatrixSize = roundUp<1024>(0.8 * std::sqrt(MemoryAvail / sizeof(FloatingPointType) / 3));
    MemorySize = sizeof(FloatingPointType) * MatrixSize * MatrixSize;
  }

  firestarter::log::trace() << "Set " << compat::AccelleratorString << " matrix size: " << MatrixSize;
  // Calculate the numnber of C matricies based on the available memory and the matrix size in B.
  const auto Iterations = (MemoryAvail - 2 * MemorySize) / MemorySize;
  // The numner of used memory are two time the matrix size in B (Matrix A and B) plus the number of matricies in C.
  const auto UseBytes = (2 + Iterations) * MemorySize;

  firestarter::log::trace() << "Allocating " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex;

  // allocating memory on the GPU
  compat::accellSafeCall(compat::malloc<FloatingPointType>(&ADataPtr, MemorySize), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::malloc<FloatingPointType>(&BDataPtr, MemorySize), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::malloc<FloatingPointType>(&CDataPtr, Iterations * MemorySize), __FILE__, __LINE__,
                         DeviceIndex);

  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex
                            << ". A: " << ADataPtr << " (Size: " << MemorySize << "B)";
  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex
                            << ". B: " << BDataPtr << " (Size: " << MemorySize << "B)";
  firestarter::log::trace() << "Allocated " << compat::AccelleratorString << " memory on device nr. " << DeviceIndex
                            << ". C: " << CDataPtr << " (Size: " << (Iterations * MemorySize) << "B)";

  firestarter::log::trace() << "Initializing " << compat::AccelleratorString << " matrices a, b on device nr. "
                            << DeviceIndex << ". Using " << MatrixSize * MatrixSize << " elements of size "
                            << sizeof(FloatingPointType) << " Byte";
  // initialize matrix A and B on the GPU with random values
  if constexpr (IsHalf) {
    initHalfMatrix(ADataPtr, MatrixSize * MatrixSize, Seed, DeviceIndex);
    initHalfMatrix(BDataPtr, MatrixSize * MatrixSize, Seed + 1, DeviceIndex);
  } else {
    compat::RandGenerator RandomGen{};
    compat::accellSafeCall(compat::randCreateGeneratorPseudoRandom(RandomGen), __FILE__, __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::randSetPseudoRandomGeneratorSeed(RandomGen, Seed), __FILE__, __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::generateUniform<FloatingPointType>(RandomGen, ADataPtr, MatrixSize * MatrixSize),
                           __FILE__, __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::generateUniform<FloatingPointType>(RandomGen, BDataPtr, MatrixSize * MatrixSize),
                           __FILE__, __LINE__, DeviceIndex);
    compat::accellSafeCall(compat::randDestroyGenerator(RandomGen), __FILE__, __LINE__, DeviceIndex);
  }

  // initialize c_data_ptr with copies of A
  for (std::size_t I = 0; I < Iterations; I++) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    auto DestinationPtr = CDataPtr + (I * MatrixSize * MatrixSize);
    firestarter::log::trace() << "Initializing " << compat::AccelleratorString << " matrix c-" << I << " by copying "
                              << MemorySize << " byte from " << ADataPtr << " to " << DestinationPtr << "\n";
    compat::accellSafeCall(compat::memcpyDtoD<FloatingPointType>(DestinationPtr, ADataPtr, MemorySize), __FILE__,
                           __LINE__, DeviceIndex);
  }

  // save gpuvar->init_count and sys.out
  {
    const std::lock_guard<std::mutex> Lk(WaitForInitCvMutex);

    auto ToMiB = [](const size_t Val) { return Val / 1024 / 1024; };
    firestarter::log::info() << "   GPU " << DeviceIndex << "\n"
                             << "    name:           " << Properties.name << "\n"
                             << "    memory:         " << ToMiB(MemoryAvail) << "/" << ToMiB(MemoryTotal)
                             << " MiB available (using " << ToMiB(UseBytes) << " MiB)\n"
                             << "    matrix size:    " << MatrixSize << "\n"
                             << "    used precision: "
                             << (IsHalf ? "half (FP16 with FP32 accumulation)"
                                        : ((sizeof(FloatingPointType) == sizeof(double)) ? "double" : "single"));

    InitCount++;
  }
  WaitForInitCv.notify_all();

  const ScalarType Alpha = 1.0;
  const ScalarType Beta = 0.0;

  // actual stress begins here
  while (LoadVar != firestarter::LoadThreadWorkType::LoadStop) {
    for (std::size_t I = 0; I < Iterations; I++) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      auto CSectionPtr = CDataPtr + (I * MatrixSize * MatrixSize);
      if constexpr (IsHalf) {
        compat::accellSafeCall(compat::gemmHalf(Blas, compat::BlasOperation::BLAS_OP_N,
                                                compat::BlasOperation::BLAS_OP_N, MatrixSize, MatrixSize, MatrixSize,
                                                Alpha, ADataPtr, MatrixSize, BDataPtr, MatrixSize, Beta, CSectionPtr,
                                                MatrixSize),
                               __FILE__, __LINE__, DeviceIndex);
      } else {
        compat::accellSafeCall(compat::gemm<FloatingPointType>(Blas, compat::BlasOperation::BLAS_OP_N,
                                                               compat::BlasOperation::BLAS_OP_N, MatrixSize, MatrixSize,
                                                               MatrixSize, Alpha, ADataPtr, MatrixSize, BDataPtr,
                                                               MatrixSize, Beta, CSectionPtr, MatrixSize),
                               __FILE__, __LINE__, DeviceIndex);
      }
      compat::accellSafeCall(compat::deviceSynchronize(), __FILE__, __LINE__, DeviceIndex);

      // The number of executed flop for a gemm with two square 'MatrixSize' sized matricies is 2 *
      // ('MatrixSize'^3)
      if constexpr (std::is_same_v<FloatingPointType, float>) {
        ExecutedFlop.SingleFlop += 2 * MatrixSize * MatrixSize * MatrixSize;
      } else if constexpr (std::is_same_v<FloatingPointType, double>) {
        ExecutedFlop.DoubleFlop += 2 * MatrixSize * MatrixSize * MatrixSize;
      } else if constexpr (IsHalf) {
        ExecutedFlop.HalfFlop += 2 * MatrixSize * MatrixSize * MatrixSize;
      }
    }
  }

  compat::accellSafeCall(compat::free<FloatingPointType>(ADataPtr), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::free<FloatingPointType>(BDataPtr), __FILE__, __LINE__, DeviceIndex);
  compat::accellSafeCall(compat::free<FloatingPointType>(CDataPtr), __FILE__, __LINE__, DeviceIndex);

  compat::accellSafeCall(compat::blasDestroy(Blas), __FILE__, __LINE__, DeviceIndex);

  compat::accellSafeCall(compat::destroyContextOrStream(StreamOrContext), __FILE__, __LINE__, DeviceIndex);
}

}; // namespace

Cuda::Cuda(const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat, bool UseDouble, bool UseHalf,
           uint64_t MatrixSize, int Gpus) {
  std::condition_variable WaitForInitCv;
  std::mutex WaitForInitCvMutex;
  bool InitDone = false;

  std::thread T(Cuda::initGpus, std::ref(ExecutedFlop), std::ref(WaitForInitCv), std::ref(WaitForInitCvMutex),
                std::ref(InitDone), std::cref(LoadVar), UseFloat, UseDouble, UseHalf, MatrixSize, Gpus);
  InitThread = std::move(T);

  std::unique_lock<std::mutex> Lk(WaitForInitCvMutex);
  // wait for gpus to initialize
  WaitForInitCv.wait(Lk, [&InitDone] { return InitDone; });
}

void Cuda::initGpus(GpuFlop& ExecutedFlop, std::condition_variable& WaitForInitCv, std::mutex& WaitForInitCvMutex,
                    bool& InitDone, const volatile firestarter::LoadThreadWorkType& LoadVar, bool UseFloat,
                    bool UseDouble, bool UseHalf, uint64_t MatrixSize, int Gpus) {
  std::condition_variable GpuThreadsWaitForInitCv;
  std::mutex GpuThreadsWaitForInitCvMutex;
  std::vector<std::thread> GpuThreads;

  if (Gpus != 0) {
    compat::accellSafeCall(compat::init(0), __FILE__, __LINE__);

    int DevCount{};
    compat::accellSafeCall(compat::getDeviceCount(DevCount), __FILE__, __LINE__);

    if (DevCount) {
      std::atomic<int> InitCount = 0;
      int UseDoubleConverted{};

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
          void (*LoadFunc)(GpuFlop&, std::condition_variable&, std::mutex&, int, std::atomic<int>&,
                           const volatile firestarter::LoadThreadWorkType&, uint64_t) = nullptr;

          if (UseHalf && halfSupported(I)) {
            LoadFunc = createLoad<compat::Half>;
          } else {
            if (UseHalf) {
              firestarter::log::warn() << compat::AccelleratorString << " GPU " << I
                                       << " does not support half precision. Selecting precision automatically.";
            }
            // if there's a GPU in the system without Double Precision support, we
            // have to correct this.
            const auto Precision = getPrecision(I, UseDoubleConverted);
            LoadFunc = Precision ? createLoad<double> : createLoad<float>;
          }

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

  {
    const std::lock_guard<std::mutex> Lk(WaitForInitCvMutex);
    InitDone = true;
    // Notify while holding the lock so the constructor cannot destroy WaitForInitCv before this call.
    WaitForInitCv.notify_all();
  }

  /* join computation threads */
  for (auto& Thread : GpuThreads) {
    Thread.join();
  }
}

} // namespace firestarter::cuda