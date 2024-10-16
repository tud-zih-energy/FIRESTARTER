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

// This file provides compatibility for the minor differences between the CUDA and HIP APIs. We do this by:
// 1. Include the required header files for CUDA or HIP
// 2. Define compatibility types between CUDA and HIP. This results in all enum names to be the same in the source code.
// These types are mapped to the ones with the correct prefix. These are cu and hip, CU and HIP, cuda and hip or CUDA
// and HIP.
// 3. Define functions that converts the error code enums into strings.
// 4. Define compatibility function for cals to CUDA, HIP or one of their libraries (blas, rand etc.)

#pragma once

#include <cstddef>
#include <optional>
#include <sstream>
#include <type_traits>

namespace firestarter::cuda::compat {

#ifdef FIRESTARTER_BUILD_CUDA
// Start of CUDA compatibility types
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

enum class BlasStatusT : std::underlying_type_t<cublasStatus_t> {
  BLAS_STATUS_SUCCESS = CUBLAS_STATUS_SUCCESS,
  BLAS_STATUS_NOT_INITIALIZED = CUBLAS_STATUS_NOT_INITIALIZED,
  BLAS_STATUS_ALLOC_FAILED = CUBLAS_STATUS_ALLOC_FAILED,
  BLAS_STATUS_INVALID_VALUE = CUBLAS_STATUS_INVALID_VALUE,
  BLAS_STATUS_ARCH_MISMATCH = CUBLAS_STATUS_ARCH_MISMATCH,
  BLAS_STATUS_MAPPING_ERROR = CUBLAS_STATUS_MAPPING_ERROR,
  BLAS_STATUS_EXECUTION_FAILED = CUBLAS_STATUS_EXECUTION_FAILED,
  BLAS_STATUS_INTERNAL_ERROR = CUBLAS_STATUS_INTERNAL_ERROR,
  BLAS_STATUS_NOT_SUPPORTED = CUBLAS_STATUS_NOT_SUPPORTED,
  BLAS_STATUS_LICENSE_ERROR = CUBLAS_STATUS_LICENSE_ERROR,
};

constexpr const char* AccelleratorString = "CUDA";

enum class ErrorT : std::underlying_type_t<cuError_t> {
  Success = cudaSuccess,
};

enum class RandStatusT : std::underlying_type_t<curandStatus_t> {
  RAND_STATUS_SUCCESS = CURAND_STATUS_SUCCESS,
  RAND_STATUS_VERSION_MISMATCH = CURAND_STATUS_VERSION_MISMATCH,
  RAND_STATUS_NOT_INITIALIZED = CURAND_STATUS_NOT_INITIALIZED,
  RAND_STATUS_ALLOCATION_FAILED = CURAND_STATUS_ALLOCATION_FAILED,
  RAND_STATUS_TYPE_ERROR = CURAND_STATUS_TYPE_ERROR,
  RAND_STATUS_OUT_OF_RANGE = CURAND_STATUS_OUT_OF_RANGE,
  RAND_STATUS_LENGTH_NOT_MULTIPLE = CURAND_STATUS_LENGTH_NOT_MULTIPLE,
  RAND_STATUS_DOUBLE_PRECISION_REQUIRED = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
  RAND_STATUS_LAUNCH_FAILURE = CURAND_STATUS_LAUNCH_FAILURE,
  RAND_STATUS_PREEXISTING_FAILURE = CURAND_STATUS_PREEXISTING_FAILURE,
  RAND_STATUS_INITIALIZATION_FAILED = CURAND_STATUS_INITIALIZATION_FAILED,
  RAND_STATUS_ARCH_MISMATCH = CURAND_STATUS_ARCH_MISMATCH,
  RAND_STATUS_INTERNAL_ERROR = CURAND_STATUS_INTERNAL_ERROR,
};

using StreamOrContext = CUcontext;

template <typename FloatingPointType> using DevicePtr = CUdeviceptr;

using DeviceProperties = struct cudaDeviceProp;

using RandGenerator = curandGenerator_t;

using BlasHandle = cublasHandle_t;

using BlasStatus = cublasStatus_t;

enum class BlasOperation : std::underlying_type_t<cublasOperation_t> {
  BLAS_OP_N = CUBLAS_OP_N,
  BLAS_OP_T = CUBLAS_OP_T,
  BLAS_OP_C = CUBLAS_OP_C,
};

#elif defined(FIRESTARTER_BUILD_HIP)
// Start of HIP compatibility types

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hiprand_kernel.h>

enum class BlasStatusT : std::underlying_type_t<hipblasStatus_t> {
  BLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS,
  BLAS_STATUS_NOT_INITIALIZED = HIPBLAS_STATUS_NOT_INITIALIZED,
  BLAS_STATUS_ALLOC_FAILED = HIPBLAS_STATUS_ALLOC_FAILED,
  BLAS_STATUS_INVALID_VALUE = HIPBLAS_STATUS_INVALID_VALUE,
  BLAS_STATUS_ARCH_MISMATCH = HIPBLAS_STATUS_ARCH_MISMATCH,
  BLAS_STATUS_MAPPING_ERROR = HIPBLAS_STATUS_MAPPING_ERROR,
  BLAS_STATUS_EXECUTION_FAILED = HIPBLAS_STATUS_EXECUTION_FAILED,
  BLAS_STATUS_INTERNAL_ERROR = HIPBLAS_STATUS_INTERNAL_ERROR,
  BLAS_STATUS_NOT_SUPPORTED = HIPBLAS_STATUS_NOT_SUPPORTED,
  BLAS_STATUS_UNKNOWN = HIPBLAS_STATUS_UNKNOWN,
  BLAS_STATUS_HANDLE_IS_NULLPTR = HIPBLAS_STATUS_HANDLE_IS_NULLPTR,
  BLAS_STATUS_INVALID_ENUM = HIPBLAS_STATUS_INVALID_ENUM,
};

constexpr const char* AccelleratorString = "HIP";

enum class ErrorT ErrorT : std::underlying_type_t<hipError_t> {
  Success = hipSuccess,
};

enum class RandStatusT : std::underlying_type_t<hiprandStatus_t> {
  RAND_STATUS_SUCCESS = HIPRAND_STATUS_SUCCESS,
  RAND_STATUS_VERSION_MISMATCH = HIPRAND_STATUS_VERSION_MISMATCH,
  RAND_STATUS_NOT_INITIALIZED = HIPRAND_STATUS_NOT_INITIALIZED,
  RAND_STATUS_ALLOCATION_FAILED = HIPRAND_STATUS_ALLOCATION_FAILED,
  RAND_STATUS_TYPE_ERROR = HIPRAND_STATUS_TYPE_ERROR,
  RAND_STATUS_OUT_OF_RANGE = HIPRAND_STATUS_OUT_OF_RANGE,
  RAND_STATUS_LENGTH_NOT_MULTIPLE = HIPRAND_STATUS_LENGTH_NOT_MULTIPLE,
  RAND_STATUS_DOUBLE_PRECISION_REQUIRED = HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED,
  RAND_STATUS_LAUNCH_FAILURE = HIPRAND_STATUS_LAUNCH_FAILURE,
  RAND_STATUS_PREEXISTING_FAILURE = HIPRAND_STATUS_PREEXISTING_FAILURE,
  RAND_STATUS_INITIALIZATION_FAILED = HIPRAND_STATUS_INITIALIZATION_FAILED,
  RAND_STATUS_ARCH_MISMATCH = HIPRAND_STATUS_ARCH_MISMATCH,
  RAND_STATUS_INTERNAL_ERROR = HIPRAND_STATUS_INTERNAL_ERROR,
  RAND_STATUS_NOT_IMPLEMENTED = HIPRAND_STATUS_NOT_IMPLEMENTED,
};

using StreamOrContext = hipStream_t;

template <typename FloatingPointType> using DevicePtr = FloatingPointType*;

using DeviceProperties = struct hipDeviceProp_t;

using RandGenerator = hiprandGenerator_t;

using BlasHandle = hipblasHandle_t;

using BlasStatus = hipblasStatus_t;

enum class BlasOperation : std::underlying_type_t<hipblasOperation_t> {
  BLAS_OP_N = HIPBLAS_OP_N,
  BLAS_OP_T = HIPBLAS_OP_T,
  BLAS_OP_C = HIPBLAS_OP_C,
};

#else

#error "Attempting to compile file but neither CUDA nor HIP is used"

// Start of compatibility types for clangd

enum class BlasStatusT {
  BLAS_STATUS_SUCCESS = 0,
};

constexpr const char* AccelleratorString = "unknown";

enum class ErrorT {
  Success = 0,
};

enum class RandStatusT {
  RAND_STATUS_SUCCESS = 0,
};

using StreamOrContext = void*;

template <typename FloatingPointType> using DevicePtr = std::size_t;

using DeviceProperties = void*;

using RandGenerator = void*;

using BlasHandle = void*;

using BlasStatus = void*;

enum class BlasOperation {
  BLAS_OP_N,
  BLAS_OP_T,
  BLAS_OP_C,
};

#endif

// abstracted function for both CUDA and HIP

/// Get the error string from a call to CUDA of HIP libraries.
/// \arg Status The status code that is returned by these calls.
/// \return The error as a string.
auto getErrorString(ErrorT Error) -> const char* {
#ifdef FIRESTARTER_BUILD_CUDA
  return cudaGetErrorString(static_cast<cudaError_t>(Error));
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipGetErrorString(static_cast<hipError_t>(Error));
#else
  (void)Error;
  return "unknown";
#endif
}

/// Get the error string from a call to CUDA of HIP blas library.
/// \arg Status The status code that is returned by these calls.
/// \return The error as a string.
constexpr auto getErrorString(BlasStatusT Status) -> const char* {
  switch (Status) {
  case BlasStatusT::BLAS_STATUS_SUCCESS:
    return "blas status: success";
#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
  case BlasStatusT::BLAS_STATUS_NOT_INITIALIZED:
    return "blas status: not initialized";
  case BlasStatusT::BLAS_STATUS_ALLOC_FAILED:
    return "blas status: alloc failed";
  case BlasStatusT::BLAS_STATUS_INVALID_VALUE:
    return "blas status: invalid value";
  case BlasStatusT::BLAS_STATUS_ARCH_MISMATCH:
    return "blas status: arch mismatch";
  case BlasStatusT::BLAS_STATUS_MAPPING_ERROR:
    return "blas status: mapping error";
  case BlasStatusT::BLAS_STATUS_EXECUTION_FAILED:
    return "blas status: execution failed";
  case BlasStatusT::BLAS_STATUS_INTERNAL_ERROR:
    return "blas status: internal error";
  case BlasStatusT::BLAS_STATUS_NOT_SUPPORTED:
    return "blas status: not supported";
#endif
#ifdef FIRESTARTER_BUILD_CUDA
  case BlasStatusT::BLAS_STATUS_LICENSE_ERROR:
    return "blas status: license error";
#endif
#ifdef FIRESTARTER_BUILD_HIP
  case BlasStatusT::BLAS_STATUS_UNKNOWN:
    return "blas status: unknown";
  case BlasStatusT::BLAS_STATUS_HANDLE_IS_NULLPTR:
    return "blas status: handle is null pointer";
  case BlasStatusT::BLAS_STATUS_INVALID_ENUM:
    return "blas status: invalid enum";
#endif
  default:
    return "unknown";
  }
}

/// Get the error string from a call to CUDA of HIP random library.
/// \arg Status The status code that is returned by these calls.
/// \return The error as a string.
constexpr auto getErrorString(RandStatusT Status) -> const char* {
  switch (Status) {
  case RandStatusT::RAND_STATUS_SUCCESS:
    return "rand status: success";
#if defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP)
  case RandStatusT::RAND_STATUS_VERSION_MISMATCH:
    return "rand status: version mismatch";
  case RandStatusT::RAND_STATUS_NOT_INITIALIZED:
    return "rand status: not initialized";
  case RandStatusT::RAND_STATUS_ALLOCATION_FAILED:
    return "rand status: allocation failed";
  case RandStatusT::RAND_STATUS_TYPE_ERROR:
    return "rand status: type error";
  case RandStatusT::RAND_STATUS_OUT_OF_RANGE:
    return "rand status: out of range";
  case RandStatusT::RAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "rand status: length not multiple";
  case RandStatusT::RAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "rand status: double precision required";
  case RandStatusT::RAND_STATUS_LAUNCH_FAILURE:
    return "rand status: launch failure";
  case RandStatusT::RAND_STATUS_PREEXISTING_FAILURE:
    return "rand status: preexisting failure";
  case RandStatusT::RAND_STATUS_INITIALIZATION_FAILED:
    return "rand status: initialization failed";
  case RandStatusT::RAND_STATUS_ARCH_MISMATCH:
    return "rand status: arch mismatch";
  case RandStatusT::RAND_STATUS_INTERNAL_ERROR:
    return "rand status: internal error";
#endif
#ifdef FIRESTARTER_BUILD_HIP
  case RandStatusT::RAND_STATUS_NOT_IMPLEMENTED:
    return "rand status: not implemented";
#endif
  default:
    return "unknown";
  }
}

#ifdef FIRESTARTER_BUILD_CUDA
/// Get the error string from a call to CUDA library.
/// \arg Result The status code that is returned by these calls.
/// \return The error as a string.
auto getErrorString(CUresult Result) -> const char* {
  const char* ErrorString;
  accellSafeCall(cuGetErrorName(Result, &ErrorString), __FILE__, __LINE__);
  return ErrorString;
}
#else
// define types to not run into compile errors with if constexpr

enum class CUresult {};
// NOLINTBEGIN(readability-identifier-naming)
constexpr const int CUDA_SUCCESS = 0;
// NOLINTEND(readability-identifier-naming)
#endif

/// Use this function as a wrapper to all calls of CUDA or HIP functions. If an error occured we abort and print the
/// error code.
/// \tparam T The type of the error code returned from calls to CUDA or HIP. This may be one of BlasStatusT, ErrorT,
/// RandStatusT or CUresult.
/// \arg TVal The errorcode returned from calls to CUDA or HIP.
/// \arg File The file for the log message in which the error occured.
/// \arg Line The line for the log message in which the error occured.
/// \arg DeviceIndex if the CUDA or HIP call is associated to a specific device, the index of the device should be
/// provided here for the log message.
template <typename T>
inline void accellSafeCall(T TVal, const char* File, const int Line, std::optional<int> DeviceIndex = std::nullopt_t) {
  if constexpr (std::is_same_v<T, BlasStatusT>) {
    if (TVal == BlasStatusT::BLAS_STATUS_SUCCESS) {
      return;
    }
  } else if constexpr (std::is_same_v<T, ErrorT>) {
    if (TVal == ErrorT::Success) {
      return;
    }
  } else if constexpr (std::is_same_v<T, RandStatusT>) {
    if (TVal == RandStatusT::RAND_STATUS_SUCCESS) {
      return;
    }
  } else if constexpr (std::is_same_v<T, CUresult>) {
#ifndef FIRESTARTER_BUILD_CUDA
    static_assert(false, "Tried to call accell_safe_call with CUresult, but not building for CUDA.");
#endif
    if (TVal == CUDA_SUCCESS) {
      return;
    }
  } else {
    static_assert(false, "Tried to call accell_safe_call with an unknown type.");
  }

  std::stringstream Ss;
  Ss << AccelleratorString << " error at " << File << ":" << Line << ": error code = " << TVal << " ("
     << getErrorString(TVal) << ")";

  if (DeviceIndex) {
    Ss << ", device index: " << *DeviceIndex;
  }

  firestarter::log::error() << Ss.str();
  exit(static_cast<int>(TVal));
}

/// Wrapper to cuInit or hipInit.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg Flags The Flags forwarded to cuInit or hipInit.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto init(unsigned int Flags) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cuInit(Flags);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipInit(Flags);
#else
  (void)Flags;
  static_assert(false, "Tried to call init, but neither building for CUDA nor HIP.");
#endif
}

/// Get the number GPU devices. Wrapper to cuDeviceGetCount or hipGetDeviceCount.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg DevCount The reference to where the number of GPU devices will be written.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto getDeviceCount(int& DevCount) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cuDeviceGetCount(&DevCount);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipGetDeviceCount(&DevCount);
#else
  (void)DevCount;
  static_assert(false, "Tried to call getDeviceCount, but neither building for CUDA nor HIP.");
#endif
}

/// Create a context in case of CUDA or a stream in case of HIP on a specific device. It must be deleted with
/// destroyContextOrStream.
/// \arg DeviceIndex The device on which to create the context or stream.
/// \return The created context or stream.
auto createContextOrStream(int DeviceIndex) -> StreamOrContext {
  StreamOrContext Soc;
#ifdef FIRESTARTER_BUILD_CUDA
  firestarter::log::trace() << "Creating " << AccelleratorString << " context for computation on device nr. "
                            << DeviceIndex;
  CUdevice Device;
  accell_safe_call(cuDeviceGet(&Device, DeviceIndex), __FILE__, __LINE__, DeviceIndex);
  accell_safe_call(cuCtxCreate(&Soc, 0, device), __FILE__, __LINE__, DeviceIndex);

  firestarter::log::trace() << "Set created " << AccelleratorString << " context on device nr. " << DeviceIndex;
  ACCELL_SAFE_CALL(cuCtxSetCurrent(Soc), DeviceIndex);
#elif defined(FIRESTARTER_BUILD_HIP)
  firestarter::log::trace() << "Creating " << AccelleratorString << " Stream for computation on device nr. "
                            << DeviceIndex;
  accell_safe_call(hipSetDevice(DeviceIndex), __FILE__, __LINE__, DeviceIndex);
  accell_safe_call(hipStreamCreate(&Soc), __FILE__, __LINE__, DeviceIndex);
#else
  (void)DeviceIndex;
  static_assert(false, "Tried to call createContextOrStream, but neither building for CUDA nor HIP.");
#endif
  return Soc;
}

/// Destroy the context (CUDA) or stream (HIP) with cuCtxDestroy and hipStreamDestroy respectively.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg Soc The reference to the context or stream.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto destroyContextOrStream(StreamOrContext& Soc) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cuCtxDestroy(Soc);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipStreamDestroy(Soc);
#else
  (void)Soc;
  static_assert(false, "Tried to call destroyContextOrStream, but neither building for CUDA nor HIP.");
#endif
}

/// Create a blas handle. Wrapper to cublasCreate or hipblasCreate.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg BlasHandle The reference to a BlasHandle object which will be initialized.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto blasCreate(BlasHandle& BlasHandle) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cublasCreate(&BlasHandle);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipblasCreate(&BlasHandle);
#else
  (void)BlasHandle;
  static_assert(false, "Tried to call blasCreate, but neither building for CUDA nor HIP.");
#endif
}

/// Destory a blas handle. Wrapper to cublasDestroy or hipblasDestroy.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg BlasHandle The reference to a BlasHandle object which will be destroyed.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto blasDestory(BlasHandle& BlasHandle) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cublasDestroy(BlasHandle);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipblasDestroy(BlasHandle);
#else
  (void)BlasHandle;
  static_assert(false, "Tried to call blasDestory, but neither building for CUDA nor HIP.");
#endif
}

/// Get the properties of a specific GPU device. Wrapper to cudaGetDeviceProperties or hipGetDeviceProperties.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg Property The reference to the properties that are retrived.
/// \arg DeviceIndex The index of the GPU device for which to retrive the device properties.s
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto getDeviceProperties(DeviceProperties& Property, int DeviceIndex) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cudaGetDeviceProperties(&Property, DeviceIndex);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipGetDeviceProperties(&Property, DeviceIndex);
#else
  (void)Property;
  (void)DeviceIndex;
  static_assert(false, "Tried to call getDeviceProperties, but neither building for CUDA nor HIP.");
#endif
}

/// Get the number of memory in the current CUDA or HIP context. Wrapper to cuMemGetInfo or
/// hipMemGetInfo.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg MemoryAvail The reference to the available memory that is retrived.
/// \arg MemoryTotal The reference to the total memory that is retrived.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto memGetInfo(std::size_t& MemoryAvail, std::size_t& MemoryTotal) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cuMemGetInfo(&MemoryAvail, &MemoryTotal);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipMemGetInfo(&MemoryAvail, &MemoryTotal);
#else
  (void)MemoryAvail;
  (void)MemoryTotal;
  static_assert(false, "Tried to call memGetInfo, but neither building for CUDA nor HIP.");
#endif
}

/// Malloc device memory in the current CUDA or HIP context. Wrapper to cuMemAlloc or
/// hipMalloc.
/// \tparam ReturnType The type of the return code to these calls.
/// \tparam FloatingPointType The type of the floating point used. Either float or double.
/// \arg Ptr The reference to the device pointer which is retrieved by the malloc call.
/// \arg MemorySize The memory that is allocated on the device in bytes.
/// \returns The Error code returned from these calls.
template <typename ReturnType, typename FloatingPointType>
auto malloc(DevicePtr<FloatingPointType>& Ptr, std::size_t MemorySize) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cuMemAlloc(&Ptr, MemorySize);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipMalloc(&Ptr, MemorySize);
#else
  (void)Ptr;
  (void)MemorySize;
  static_assert(false, "Tried to call malloc, but neither building for CUDA nor HIP.");
#endif
}

/// Free device memory in the current CUDA or HIP context. Wrapper to cuMemFree or
/// hipFree.
/// \tparam ReturnType The type of the return code to these calls.
/// \tparam FloatingPointType The type of the floating point used. Either float or double.
/// \arg Ptr The reference to the device pointer which is used in the free call.
/// \returns The Error code returned from these calls.
template <typename ReturnType, typename FloatingPointType> auto free(DevicePtr<FloatingPointType>& Ptr) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cuMemFree(Ptr);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipFree(Ptr);
#else
  (void)Ptr;
  static_assert(false, "Tried to call free, but neither building for CUDA nor HIP.");
#endif
}

/// Create a random generator in the current CUDA or HIP context. Wrapper to curandCreateGenerator or
/// hiprandCreateGenerator.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg RandomGen The reference to the random generation which is retrived by the calls.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto randCreateGeneratorPseudoRandom(RandGenerator& RandomGen) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return curandCreateGenerator(&RandomGen, CURAND_RNG_PSEUDO_DEFAULT);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hiprandCreateGenerator(&RandomGen, HIPRAND_RNG_PSEUDO_DEFAULT);
#else
  (void)RandomGen;
  static_assert(false, "Tried to call randCreateGeneratorPseudoRandom, but neither building for CUDA nor HIP.");
#endif
}

/// Set the pseudo random generator seed in the current CUDA or HIP context. Wrapper to
/// curandSetPseudoRandomGeneratorSeed or hiprandSetPseudoRandomGeneratorSeed.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg RandomGen The reference to the random generator.
/// \arg Seed The seed used to initialize the pseudo random generator.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto randSetPseudoRandomGeneratorSeed(RandGenerator& RandomGen, int Seed) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return curandSetPseudoRandomGeneratorSeed(RandomGen, Seed);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hiprandSetPseudoRandomGeneratorSeed(RandomGen, Seed);
#else
  (void)RandomGen;
  (void)Seed;
  static_assert(false, "Tried to call randSetPseudoRandomGeneratorSeed, but neither building for CUDA nor HIP.");
#endif
}

/// Initialize the provided memory with with a specific number of uniform random floats. Wrapper to
/// curandGenerateUniform or hiprandGenerateUniform.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg RandomGen The reference to the random generator.
/// \arg OutputPtr The device pointer on which is initialized with specific number of uniform random floats.
/// \arg Num The number of unifrom random floats.
/// \returns The Error code returned from these calls.
template <typename ReturnType>
auto randGenerateUniform(RandGenerator& RandomGen, DevicePtr<float> OutputPtr, std::size_t Num) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return curandGenerateUniform(RandomGen, OutputPtr, Num);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hiprandGenerateUniform(RandomGen, OutputPtr, Num);
#else
  (void)RandomGen;
  (void)OutputPtr;
  (void)Num;
  static_assert(false, "Tried to call randGenerateUniform, but neither building for CUDA nor HIP.");
#endif
}

/// Initialize the provided memory with with a specific number of uniform random doubles. Wrapper to
/// curandGenerateUniformDouble or hiprandGenerateUniformDouble.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg RandomGen The reference to the random generator.
/// \arg OutputPtr The device pointer on which is initialized with specific number of uniform random floats.
/// \arg Num The number of unifrom random doubles.
/// \returns The Error code returned from these calls.
template <typename ReturnType>
auto randGenerateUniformDouble(RandGenerator& RandomGen, DevicePtr<double> OutputPtr, std::size_t Num) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return curandGenerateUniformDouble(RandomGen, OutputPtr, Num);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hiprandGenerateUniformDouble(RandomGen, OutputPtr, Num);
#else
  (void)RandomGen;
  (void)OutputPtr;
  (void)Num;
  static_assert(false, "Tried to call randGenerateUniformDouble, but neither building for CUDA nor HIP.");
#endif
}

/// Initialize the provided memory with with a specific number of uniform random floating points. Wrapper to
/// randGenerateUniform or randGenerateUniformDouble.
/// \tparam ReturnType The type of the return code to these calls.
/// \tparam FloatPointType The float point types is used. Either float or double.
/// \arg Generator The reference to the random generator.
/// \arg OutputPtr The device pointer on which is initialized with specific number of uniform random floats.
/// \arg Num The number of unifrom random doubles.
/// \returns The Error code returned from these calls.
template <typename ReturnType, typename FloatPointType>
auto generateUniform(RandGenerator& Generator, DevicePtr<FloatPointType> OutputPtr, size_t Num) -> ReturnType {
  if constexpr (std::is_same_v<FloatPointType, float>) {
    return randGenerateUniform<ReturnType>(Generator, OutputPtr, Num);
  } else if constexpr (std::is_same_v<FloatPointType, double>) {
    return randGenerateUniformDouble<ReturnType>(Generator, OutputPtr, Num);
  } else {
    static_assert(false, "generateUniform<FloatPointType>: Template argument must be either float or double");
  }
}

/// Destory a random generator in the current CUDA or HIP context. Wrapper to curandDestroyGenerator or
/// hiprandDestroyGenerator.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg RandomGen The reference to the random generation which shoule be destroyed.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto randDestroyGenerator(RandGenerator& RandomGen) -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return curandDestroyGenerator(RandomGen);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hiprandDestroyGenerator(RandomGen);
#else
  (void)RandomGen;
  static_assert(false, "Tried to call randDestroyGenerator, but neither building for CUDA nor HIP.");
#endif
}

/// Copy memory from a device pointer to another device pointer. Wrapper to cuMemcpyDtoD or hipMemcpyDtoD.
/// \tparam ReturnType The type of the return code to these calls.
/// \arg DestinationPtr The destination address.
/// \arg SourcePtr The source address.
/// \arg Size The number of bytes to copy.
/// \returns The Error code returned from these calls.
template <typename ReturnType, typename FloatPointType>
auto memcpyDtoD(DevicePtr<FloatPointType> DestinationPtr, DevicePtr<FloatPointType> SourcePtr, std::size_t Size)
    -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cuMemcpyDtoD(DestinationPtr, SourcePtr, Size);
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipMemcpyDtoD(DestinationPtr, SourcePtr, Size);
#else
  (void)DestinationPtr;
  (void)SourcePtr;
  (void)Size;
  static_assert(false, "Tried to call memcpyDtoD, but neither building for CUDA nor HIP.");
#endif
}

/// Block until the current device finished. Wrapper to cudaDeviceSynchronize or hipcudaDeviceSynchronize.
/// \tparam ReturnType The type of the return code to these calls.
/// \returns The Error code returned from these calls.
template <typename ReturnType> auto deviceSynchronize() -> ReturnType {
#ifdef FIRESTARTER_BUILD_CUDA
  return cudaDeviceSynchronize();
#elif defined(FIRESTARTER_BUILD_HIP)
  return hipcudaDeviceSynchronize();
#else
  static_assert(false, "Tried to call deviceSynchronize, but neither building for CUDA nor HIP.");
#endif
}

/// This function performs the matrix-matrix multiplication C = Alpha * op(A) * op(B) + Beta * C with op(A) and op(B)
/// described by the selected operation for Transa and Transb. BlasOperation::BLAS_OP_N maps to op(X) = X,
/// BlasOperation::BLAS_OP_T to op(X) = X transposed and BlasOperation::BLAS_OP_C to op(X) = conjugate transpose of X.
/// It wrapps (cu|hip)blas(S|D)gemm.
/// \tparam FloatPointType The float point types is used. Either float or double.
/// \arg Handle The blass handle
/// \arg Transa The operation selected for op(A)
/// \arg Transb The operation selected for op(B)
/// \arg M Number of rows of matrix op(A) and C.
/// \arg N Number of columns of matrix op(B) and C.
/// \arg K Number of columns of op(A) and rows of op(B).
/// \arg Alpha
/// \arg A
/// \arg Lda Leading dimension of two-dimensional array used to store the matrix A.
/// \arg B
/// \arg Ldb Leading dimension of two-dimensional array used to store matrix B.
/// \arg Beta
/// \arg C
/// \arg Ldc Leading dimension of a two-dimensional array used to store the matrix C.
/// \returns The Error code returned from these calls.
template <typename FloatPointType>
auto gemm(BlasHandle Handle, BlasOperation Transa, BlasOperation Transb, int& M, int& N, int& K,
          const FloatPointType* Alpha, const DevicePtr<FloatPointType>& A, int& Lda, const DevicePtr<FloatPointType>& B,
          int& Ldb, const FloatPointType* Beta, DevicePtr<FloatPointType>& C, int& Ldc) -> BlasStatus {
  if constexpr (std::is_same_v<FloatPointType, float>) {
#ifdef FIRESTARTER_BUILD_CUDA
    return cublasSgemm(Handle, Transa, Transb, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc);
#elif defined(FIRESTARTER_BUILD_HIP)
    return hipblasSgemm(Handle, Transa, Transb, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc);
#endif
  } else if constexpr (std::is_same_v<FloatPointType, double>) {
#ifdef FIRESTARTER_BUILD_CUDA
    return cublasDgemm(Handle, Transa, Transb, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc);
#elif defined(FIRESTARTER_BUILD_HIP)
    return hipblasDgemm(Handle, Transa, Transb, M, N, K, Alpha, A, Lda, B, Ldb, Beta, C, Ldc);
#endif
  } else {
    (void)Handle;
    (void)Transa;
    (void)Transb;
    (void)M;
    (void)N;
    (void)K;
    (void)Alpha;
    (void)A;
    (void)Lda;
    (void)B;
    (void)Ldb;
    (void)Beta;
    (void)C;
    (void)Ldc;
    static_assert(false, "gemm<FloatPointType>: Template argument must be either float or double");
  }

#if not(defined(FIRESTARTER_BUILD_CUDA) || defined(FIRESTARTER_BUILD_HIP))
  (void)Handle;
  (void)Transa;
  (void)Transb;
  (void)M;
  (void)N;
  (void)K;
  (void)Alpha;
  (void)A;
  (void)Lda;
  (void)B;
  (void)Ldb;
  (void)Beta;
  (void)C;
  (void)Ldc;
  static_assert(false, "Tried to call gemm, but neither building for CUDA nor HIP.");
#endif
}

} // namespace firestarter::cuda::compat