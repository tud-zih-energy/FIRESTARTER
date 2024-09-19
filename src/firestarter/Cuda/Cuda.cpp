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

#include <firestarter/Cuda/Cuda.hpp>
#include <firestarter/LoadWorkerData.hpp>
#include <firestarter/Logging/Log.hpp>

#ifdef FIRESTARTER_BUILD_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#define FS_ACCEL_PREFIX_LC_LONG cuda
#define FS_ACCEL_PREFIX_LC cu
#define FS_ACCEL_PREFIX_UC CU
#define FS_ACCEL_PREFIX_UC_LONG CUDA
#define FS_ACCEL_STRING "CUDA"
#else
#ifdef FIRESTARTER_BUILD_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hiprand_kernel.h>
#define FS_ACCEL_PREFIX_LC_LONG hip
#define FS_ACCEL_PREFIX_LC hip
#define FS_ACCEL_PREFIX_UC HIP
#define FS_ACCEL_PREFIX_UC_LONG HIP
#define FS_ACCEL_STRING "HIP"
#else
#error "Attempting to compile file but neither CUDA nor HIP is used"
#endif
#endif
#define CONCAT_(prefix, suffix) prefix##suffix
/// Concatenate `prefix, suffix` into `prefixsuffix`
#define CONCAT(prefix, suffix) CONCAT_(prefix, suffix)
//#define FS_ACCEL_ERROR_TYPE CONCAT(FS_ACCEL_PREFIX_LC_LONG,Error_t)
//#define FS_ACCEL_BLAS_STATUS_TYPE cublasStatus_t
//#define FS_ACCEL_RAND_STATUS_TYPE curandStatus_t

#include <algorithm>
#include <atomic>
#include <type_traits>

#define ACCELL_SAFE_CALL(cuerr, dev_index) accell_safe_call(cuerr, dev_index, __FILE__, __LINE__)
#define SEED 123

using namespace firestarter::cuda;

// CUDA error checking
static inline void accell_safe_call(CONCAT(FS_ACCEL_PREFIX_LC_LONG, Error_t) cuerr, int dev_index, const char* file,
                                    const int line) {
  if (cuerr != CONCAT(FS_ACCEL_PREFIX_LC_LONG, Success) && cuerr != 1) {
    firestarter::log::error() << FS_ACCEL_STRING " error at " << file << ":" << line << ": error code = " << cuerr
                              << " (" << CONCAT(FS_ACCEL_PREFIX_LC_LONG, GetErrorString)(cuerr)
                              << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}

static const char* _accellGetErrorEnum(CONCAT(FS_ACCEL_PREFIX_LC, blasStatus_t) error) {
  switch (error) {
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_SUCCESS):
    return FS_ACCEL_STRING "blas status: success";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_NOT_INITIALIZED):
    return FS_ACCEL_STRING "blas status: not initialized";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_ALLOC_FAILED):
    return FS_ACCEL_STRING "blas status: alloc failed";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_INVALID_VALUE):
    return FS_ACCEL_STRING "blas status: invalid value";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_ARCH_MISMATCH):
    return FS_ACCEL_STRING "blas status: arch mismatch";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_MAPPING_ERROR):
    return FS_ACCEL_STRING "blas status: mapping error";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_EXECUTION_FAILED):
    return FS_ACCEL_STRING "blas status: execution failed";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_INTERNAL_ERROR):
    return FS_ACCEL_STRING "blas status: internal error";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_NOT_SUPPORTED):
    return FS_ACCEL_STRING "blas status: not supported";
#ifdef FIRESTARTER_BUILD_CUDA
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_LICENSE_ERROR):
    return FS_ACCEL_STRING "blas status: license error";
#endif
#ifdef FIRESTARTER_BUILD_HIP
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_UNKNOWN):
    return FS_ACCEL_STRING "blas status: unknown";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_HANDLE_IS_NULLPTR):
    return FS_ACCEL_STRING "blas status: handle is null pointer";
  case CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_INVALID_ENUM):
    return FS_ACCEL_STRING "blas status: invalid enum";
#endif
  }

  return "<unknown>";
}

static inline void accell_safe_call(CONCAT(FS_ACCEL_PREFIX_LC, blasStatus_t) cuerr, int dev_index, const char* file,
                                    const int line) {
  if (cuerr != CONCAT(FS_ACCEL_PREFIX_UC, BLAS_STATUS_SUCCESS)) {
    firestarter::log::error() << FS_ACCEL_STRING "BLAS error at " << file << ":" << line << ": error code = " << cuerr
                              << " (" << _accellGetErrorEnum(cuerr) << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}

#ifdef FIRESTARTER_BUILD_CUDA
static inline void accell_safe_call(CONCAT(FS_ACCEL_PREFIX_UC, result) cuerr, int dev_index, const char* file,
                                    const int line) {
  if (cuerr != CONCAT(FS_ACCEL_PREFIX_UC_LONG, _SUCCESS)) {
    const char* errorString;

    ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, GetErrorName)(cuerr, &errorString), dev_index);

    firestarter::log::error() << FS_ACCEL_STRING " error at " << file << ":" << line << ": error code = " << cuerr
                              << " (" << errorString << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}
#endif

static const char* _accellrandGetErrorEnum(CONCAT(FS_ACCEL_PREFIX_LC, randStatus_t) cuerr) {
  switch (cuerr) {
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_SUCCESS):
    return FS_ACCEL_STRING "rand status: success";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_VERSION_MISMATCH):
    return FS_ACCEL_STRING "rand status: version mismatch";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_NOT_INITIALIZED):
    return FS_ACCEL_STRING "rand status: not initialized";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_ALLOCATION_FAILED):
    return FS_ACCEL_STRING "rand status: allocation failed";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_TYPE_ERROR):
    return FS_ACCEL_STRING "rand status: type error";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_OUT_OF_RANGE):
    return FS_ACCEL_STRING "rand status: out of range";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_LENGTH_NOT_MULTIPLE):
    return FS_ACCEL_STRING "rand status: length not multiple";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_DOUBLE_PRECISION_REQUIRED):
    return FS_ACCEL_STRING "rand status: double precision required";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_LAUNCH_FAILURE):
    return FS_ACCEL_STRING "rand status: launch failure";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_PREEXISTING_FAILURE):
    return FS_ACCEL_STRING "rand status: preexisting failure";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_INITIALIZATION_FAILED):
    return FS_ACCEL_STRING "rand status: initialization failed";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_ARCH_MISMATCH):
    return FS_ACCEL_STRING "rand status: arch mismatch";
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_INTERNAL_ERROR):
    return FS_ACCEL_STRING "rand status: internal error";
#ifdef FIRESTARTER_BUILD_HIP
  case CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_NOT_IMPLEMENTED):
    return FS_ACCEL_STRING "rand status: not implemented";
#endif
  }

  return "<unknown>";
}

static inline void accell_safe_call(CONCAT(FS_ACCEL_PREFIX_LC, randStatus_t) cuerr, int dev_index, const char* file,
                                    const int line) {
  if (cuerr != CONCAT(FS_ACCEL_PREFIX_UC, RAND_STATUS_SUCCESS)) {
    firestarter::log::error() << FS_ACCEL_STRING "RAND error at " << file << ":" << line << ": error code = " << cuerr
                              << " (" << _accellrandGetErrorEnum(cuerr) << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
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

#ifdef FIRESTARTER_BUILD_CUDA
static int get_precision(int useDouble, struct cudaDeviceProp properties) {
#else
#ifdef FIRESTARTER_BUILD_HIP
static int get_precision(int useDouble, struct hipDeviceProp_t properties) {
#endif
#endif
#if (CUDART_VERSION >= 8000)
  // read precision ratio (dp/sp) of GPU to choose the right variant for maximum
  // workload
  if (useDouble == 2 && properties.singleToDoublePrecisionPerfRatio > 3) {
    return 0;
  } else if (useDouble) {
    return 1;
  } else {
    return 0;
  }
}
#else
  // as precision ratio is not supported return default/user input value
  (void)properties;

  if (useDouble) {
    return 1;
  } else {
    return 0;
  }
}
#endif

static int get_precision(int device_index, int useDouble) {
  size_t memory_avail, memory_total;
#ifdef FIRESTARTER_BUILD_CUDA
  CUcontext context;
  CUdevice device;
  struct cudaDeviceProp properties;
  ACCELL_SAFE_CALL(cuDeviceGet(&device, device_index), device_index);
  ACCELL_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);
  ACCELL_SAFE_CALL(cuCtxSetCurrent(context), device_index);
#else
#ifdef FIRESTARTER_BUILD_HIP
  struct hipDeviceProp_t properties;
  ACCELL_SAFE_CALL(hipSetDevice(device_index), device_index);
#endif
#endif
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, MemGetInfo)(&memory_avail, &memory_total), device_index);
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC_LONG, GetDeviceProperties)(&properties, device_index), device_index);

  useDouble = get_precision(useDouble, properties);

  // we check for double precision support on the GPU and print errormsg, when
  // the user wants to compute DP on a SP-only-Card.
  if (useDouble && properties.major <= 1 && properties.minor <= 2) {
    std::stringstream ss;
    ss << FS_ACCEL_STRING " GPU " << device_index << ": " << properties.name << " ";

    firestarter::log::error() << ss.str() << "Doesn't support double precision.\n"
                              << ss.str() << "Compute Capability: " << properties.major << "." << properties.minor
                              << ". Requiered for double precision: >=1.3\n"
                              << ss.str() << "Stressing with single precision instead. Maybe use -f parameter.";

    useDouble = 0;
  }

#ifdef FIRESTARTER_BUILD_CUDA
  ACCELL_SAFE_CALL(cuCtxDestroy(context), device_index);
#endif

  return useDouble;
}

#ifdef FIRESTARTER_BUILD_CUDA
static int get_msize(int device_index, int useDouble) {
  CUcontext context;
  CUdevice device;
  size_t memory_avail, memory_total;

  ACCELL_SAFE_CALL(cuDeviceGet(&device, device_index), device_index);
  ACCELL_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);
  ACCELL_SAFE_CALL(cuCtxSetCurrent(context), device_index);
  ACCELL_SAFE_CALL(cuMemGetInfo(&memory_avail, &memory_total), device_index);

  ACCELL_SAFE_CALL(cuCtxDestroy(context), device_index);

  return round_up((int)(0.8 * sqrt(((memory_avail) / ((useDouble ? sizeof(double) : sizeof(float)) * 3)))),
                  1024); // a multiple of 1024 works always well
}
#endif

static CONCAT(FS_ACCEL_PREFIX_LC, blasStatus_t)
    gemm(CONCAT(FS_ACCEL_PREFIX_LC, blasHandle_t) handle, CONCAT(FS_ACCEL_PREFIX_LC, blasOperation_t) transa,
         CONCAT(FS_ACCEL_PREFIX_LC, blasOperation_t) transb, int& m, int& n, int& k, const float* alpha, const float* A,
         int& lda, const float* B, int& ldb, const float* beta, float* C, int& ldc) {
  return CONCAT(FS_ACCEL_PREFIX_LC, blasSgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static CONCAT(FS_ACCEL_PREFIX_LC, blasStatus_t)
    gemm(CONCAT(FS_ACCEL_PREFIX_LC, blasHandle_t) handle, CONCAT(FS_ACCEL_PREFIX_LC, blasOperation_t) transa,
         CONCAT(FS_ACCEL_PREFIX_LC, blasOperation_t) transb, int& m, int& n, int& k, const double* alpha,
         const double* A, int& lda, const double* B, int& ldb, const double* beta, double* C, int& ldc) {
  return CONCAT(FS_ACCEL_PREFIX_LC, blasDgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

static CONCAT(FS_ACCEL_PREFIX_LC, randStatus_t)
    generateUniform(CONCAT(FS_ACCEL_PREFIX_LC, randGenerator_t) generator, float* outputPtr, size_t num) {
  return CONCAT(FS_ACCEL_PREFIX_LC, randGenerateUniform)(generator, outputPtr, num);
}

static CONCAT(FS_ACCEL_PREFIX_LC, randStatus_t)
    generateUniform(CONCAT(FS_ACCEL_PREFIX_LC, randGenerator_t) generator, double* outputPtr, size_t num) {
  return CONCAT(FS_ACCEL_PREFIX_LC, randGenerateUniformDouble)(generator, outputPtr, num);
}

// GPU index. Used to pin this thread to the GPU.
template <typename T>
static void create_load(std::condition_variable& waitForInitCv, std::mutex& waitForInitCvMutex, int device_index,
                        std::atomic<int>& initCount, volatile unsigned long long* loadVar, int matrixSize) {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "create_load<T>: Template argument T must be either float or double");

  int iterations, i;

  firestarter::log::trace() << "Starting CUDA/HIP with given matrix size " << matrixSize;

  size_t size_use = 0;
  if (matrixSize > 0) {
    size_use = matrixSize;
  }

  size_t use_bytes, memory_size;
#ifdef FIRESTARTER_BUILD_CUDA
  CUcontext context;
  struct cudaDeviceProp properties;
  CUdevice device;
  cublasHandle_t cublas;
#else
#ifdef FIRESTARTER_BUILD_HIP
  hipStream_t stream;
  struct hipDeviceProp_t properties;
  hipDevice_t device;
  hipblasHandle_t cublas;
#endif
#endif
  // reserving the GPU and initializing cublas

  firestarter::log::trace() << "Getting " FS_ACCEL_STRING " device nr. " << device_index;
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, DeviceGet)(&device, device_index), device_index);

#ifdef FIRESTARTER_BUILD_CUDA
  firestarter::log::trace() << "Creating " FS_ACCEL_STRING " context for computation on device nr. " << device_index;
  ACCELL_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);

  firestarter::log::trace() << "Set created " FS_ACCEL_STRING " context on device nr. " << device_index;
  ACCELL_SAFE_CALL(cuCtxSetCurrent(context), device_index);
#else
#ifdef FIRESTARTER_BUILD_HIP
  firestarter::log::trace() << "Creating " FS_ACCEL_STRING " Stream for computation on device nr. " << device_index;
  ACCELL_SAFE_CALL(hipSetDevice(device_index), device_index);
  ACCELL_SAFE_CALL(hipStreamCreate(&stream), device_index);
#endif
#endif

  firestarter::log::trace() << "Create " FS_ACCEL_STRING " Blas on device nr. " << device_index;
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, blasCreate)(&cublas), device_index);

  firestarter::log::trace() << "Get " FS_ACCEL_STRING " device properties (e.g., support for double)"
                            << " on device nr. " << device_index;
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC_LONG, GetDeviceProperties)(&properties, device_index), device_index);

  // getting information about the GPU memory
  size_t memory_avail, memory_total;
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, MemGetInfo)(&memory_avail, &memory_total), device_index);

  firestarter::log::trace() << "Get " FS_ACCEL_STRING " Memory info on device nr. " << device_index << ": "
                            << memory_avail << " B avail. from " << memory_total << " B total";

  // defining memory pointers
#ifdef FIRESTARTER_BUILD_CUDA
  CUdeviceptr a_data_ptr;
  CUdeviceptr b_data_ptr;
  CUdeviceptr c_data_ptr;
#else
#ifdef FIRESTARTER_BUILD_HIP
  T* a_data_ptr;
  T* b_data_ptr;
  T* c_data_ptr;
#endif
#endif

  // check if the user has not set a matrix OR has set a too big matrixsite and
  // if this is true: set a good matrixsize
  if (!size_use || ((size_use * size_use * sizeof(T) * 3 > memory_avail))) {
    size_use = round_up((int)(0.8 * sqrt(((memory_avail) / (sizeof(T) * 3)))),
                        1024); // a multiple of 1024 works always well
  }
  firestarter::log::trace() << "Set " FS_ACCEL_STRING " matrix size: " << matrixSize;
  use_bytes = (size_t)((T)memory_avail);
  memory_size = sizeof(T) * size_use * size_use;
  iterations = (use_bytes - 2 * memory_size) / memory_size; // = 1;

  firestarter::log::trace() << "Allocating " FS_ACCEL_STRING " memory on device nr. " << device_index;

  // allocating memory on the GPU
#ifdef FIRESTARTER_BUILD_CUDA
  ACCELL_SAFE_CALL(cuMemAlloc(&a_data_ptr, memory_size), device_index);
  ACCELL_SAFE_CALL(cuMemAlloc(&b_data_ptr, memory_size), device_index);
  ACCELL_SAFE_CALL(cuMemAlloc(&c_data_ptr, iterations * memory_size), device_index);
#else
#ifdef FIRESTARTER_BUILD_HIP
  ACCELL_SAFE_CALL(hipMalloc(&a_data_ptr, memory_size), device_index);
  ACCELL_SAFE_CALL(hipMalloc(&b_data_ptr, memory_size), device_index);
  ACCELL_SAFE_CALL(hipMalloc(&c_data_ptr, iterations * memory_size), device_index);
#endif
#endif

  firestarter::log::trace() << "Allocated " FS_ACCEL_STRING " memory on device nr. " << device_index
                            << ". A: " << a_data_ptr << "(Size: " << memory_size << "B)"
                            << "\n";

  firestarter::log::trace() << "Allocated " FS_ACCEL_STRING " memory on device nr. " << device_index
                            << ". B: " << b_data_ptr << "(Size: " << memory_size << "B)"
                            << "\n";
  firestarter::log::trace() << "Allocated " FS_ACCEL_STRING " memory on device nr. " << device_index
                            << ". C: " << c_data_ptr << "(Size: " << iterations * memory_size << "B)"
                            << "\n";

  firestarter::log::trace() << "Initializing " FS_ACCEL_STRING " matrices a, b on device nr. " << device_index
                            << ". Using " << size_use * size_use << " elements of size " << sizeof(T) << " Byte";
  // initialize matrix A and B on the GPU with random values
  CONCAT(FS_ACCEL_PREFIX_LC, randGenerator_t) random_gen;
  ACCELL_SAFE_CALL(
      CONCAT(FS_ACCEL_PREFIX_LC, randCreateGenerator)(&random_gen, CONCAT(FS_ACCEL_PREFIX_UC, RAND_RNG_PSEUDO_DEFAULT)),
      device_index);
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, randSetPseudoRandomGeneratorSeed)(random_gen, SEED), device_index);
  ACCELL_SAFE_CALL(generateUniform(random_gen, (T*)a_data_ptr, size_use * size_use), device_index);
  ACCELL_SAFE_CALL(generateUniform(random_gen, (T*)b_data_ptr, size_use * size_use), device_index);
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, randDestroyGenerator)(random_gen), device_index);

  // initialize c_data_ptr with copies of A
  for (i = 0; i < iterations; i++) {
    firestarter::log::trace() << "Initializing " FS_ACCEL_STRING " matrix c-" << i << " by copying " << memory_size
                              << " byte from " << a_data_ptr << " to "
                              << c_data_ptr +
                                     (size_t)(i * size_use * size_use * (float)sizeof(T) / (float)sizeof(c_data_ptr))
                              << "\n";
    ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, MemcpyDtoD)(
                         c_data_ptr + (size_t)(i * size_use * size_use * (float)sizeof(T) / (float)sizeof(c_data_ptr)),
                         a_data_ptr, memory_size),
                     device_index);
  }

  // save gpuvar->init_count and sys.out
  {
    std::lock_guard<std::mutex> lk(waitForInitCvMutex);

#define TO_MB(x) (unsigned long)(x / 1024 / 1024)
    firestarter::log::info() << "   GPU " << device_index << "\n"
                             << "    name:           " << properties.name << "\n"
                             << "    memory:         " << TO_MB(memory_avail) << "/" << TO_MB(memory_total)
                             << " MiB available (using " << TO_MB(use_bytes) << " MiB)\n"
                             << "    matrix size:    " << size_use << "\n"
                             << "    used precision: " << ((sizeof(T) == sizeof(double)) ? "double" : "single");
#undef TO_MB

    initCount++;
  }
  waitForInitCv.notify_all();

  const T alpha = 1.0;
  const T beta = 0.0;

  int size_use_i = size_use;
  // actual stress begins here
  while (*loadVar != LOAD_STOP) {
    for (i = 0; i < iterations; i++) {
      ACCELL_SAFE_CALL(gemm(cublas, CONCAT(FS_ACCEL_PREFIX_UC, BLAS_OP_N), CONCAT(FS_ACCEL_PREFIX_UC, BLAS_OP_N),
                            size_use_i, size_use_i, size_use_i, &alpha, (const T*)a_data_ptr, size_use_i,
                            (const T*)b_data_ptr, size_use_i, &beta, (T*)c_data_ptr + i * size_use * size_use,
                            size_use_i),
                       device_index);
      ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC_LONG, DeviceSynchronize)(), device_index);
    }
  }

#ifdef FIRESTARTER_BUILD_CUDA
  ACCELL_SAFE_CALL(cuMemFree(a_data_ptr), device_index);
  ACCELL_SAFE_CALL(cuMemFree(b_data_ptr), device_index);
  ACCELL_SAFE_CALL(cuMemFree(c_data_ptr), device_index);
#else
#ifdef FIRESTARTER_BUILD_HIP
  ACCELL_SAFE_CALL(hipFree(a_data_ptr), device_index);
  ACCELL_SAFE_CALL(hipFree(b_data_ptr), device_index);
  ACCELL_SAFE_CALL(hipFree(c_data_ptr), device_index);
#endif
#endif
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, blasDestroy)(cublas), device_index);
#ifdef FIRESTARTER_BUILD_CUDA
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, CtxDestroy)(context), device_index);
#else
#ifdef FIRESTARTER_BUILD_HIP
  ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, StreamDestroy)(stream), device_index);
#endif
#endif
}

Cuda::Cuda(volatile unsigned long long* loadVar, bool useFloat, bool useDouble, unsigned matrixSize, int gpus) {
  std::thread t(Cuda::initGpus, std::ref(_waitForInitCv), loadVar, useFloat, useDouble, matrixSize, gpus);
  _initThread = std::move(t);

  std::unique_lock<std::mutex> lk(_waitForInitCvMutex);
  // wait for gpus to initialize
  _waitForInitCv.wait(lk);
}

void Cuda::initGpus(std::condition_variable& cv, volatile unsigned long long* loadVar, bool useFloat, bool useDouble,
                    unsigned matrixSize, int gpus) {
  std::condition_variable waitForInitCv;
  std::mutex waitForInitCvMutex;

  if (gpus) {
    ACCELL_SAFE_CALL(CONCAT(FS_ACCEL_PREFIX_LC, Init)(0), -1);
    int devCount;
#ifdef FIRESTARTER_BUILD_CUDA
    ACCELL_SAFE_CALL(cuDeviceGetCount(&devCount), -1);
#else
#ifdef FIRESTARTER_BUILD_HIP
    ACCELL_SAFE_CALL(hipGetDeviceCount(&devCount), -1);
#endif
#endif

    if (devCount) {
      std::vector<std::thread> gpuThreads;
      std::atomic<int> initCount = 0;
      int use_double;

      if (useFloat) {
        use_double = 0;
      } else if (useDouble) {
        use_double = 1;
      } else {
        use_double = 2;
      }

      firestarter::log::info()
#ifdef _WIN32
          << "\n  The Task Manager might show a low GPU utilization."
#endif
          << "\n  graphics processor characteristics:";

      // use all GPUs if the user gave no information about use_device
      if (gpus < 0) {
        gpus = devCount;
      }

      if (gpus > devCount) {
        firestarter::log::warn() << "You requested more " FS_ACCEL_STRING " devices than available. "
                                    "Maybe you set " FS_ACCEL_STRING "_VISIBLE_DEVICES?";
        firestarter::log::warn() << "FIRESTARTER will use " << devCount << " of the requested " << gpus
                                 << " " FS_ACCEL_STRING " device(s)";
        gpus = devCount;
      }

      {
        std::lock_guard<std::mutex> lk(waitForInitCvMutex);

        for (int i = 0; i < gpus; ++i) {
          // if there's a GPU in the system without Double Precision support, we
          // have to correct this.
          int precision = get_precision(i, use_double);

          if (precision) {
            std::thread t(create_load<double>, std::ref(waitForInitCv), std::ref(waitForInitCvMutex), i,
                          std::ref(initCount), loadVar, (int)matrixSize);
            gpuThreads.push_back(std::move(t));
          } else {
            std::thread t(create_load<float>, std::ref(waitForInitCv), std::ref(waitForInitCvMutex), i,
                          std::ref(initCount), loadVar, (int)matrixSize);
            gpuThreads.push_back(std::move(t));
          }
        }
      }

      {
        std::unique_lock<std::mutex> lk(waitForInitCvMutex);
        // wait for all threads to initialize
        waitForInitCv.wait(lk, [&] { return initCount == gpus; });
      }

      // notify that init is done
      cv.notify_all();

      /* join computation threads */
      for (auto& t : gpuThreads) {
        t.join();
      }
    } else {
      firestarter::log::info() << "    - No " FS_ACCEL_STRING " devices. Just stressing CPU(s). Maybe use "
                                  "FIRESTARTER instead of FIRESTARTER_" FS_ACCEL_STRING "?";
      cv.notify_all();
    }
  } else {
    firestarter::log::info() << "    --gpus 0 is set. Just stressing CPU(s). Maybe use "
                                "FIRESTARTER instead of FIRESTARTER_" FS_ACCEL_STRING "?";
    cv.notify_all();
  }
}
