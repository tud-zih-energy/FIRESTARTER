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


#ifndef FS_USE_HIP
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#else
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand_kernel.h>
#endif

#include <algorithm>
#include <atomic>
#include <type_traits>

#define CUDA_SAFE_CALL(cuerr, dev_index)                                       \
  cuda_safe_call(cuerr, dev_index, __FILE__, __LINE__)
#define SEED 123

using namespace firestarter::cuda;


// Check if FS_USE_HIP is defined and set the error prefix accordingly
#ifndef FS_USE_HIP // CUDA case
    #define PREFIX CU
    #define PREFIX_LC cu
    #define PREFIX_LC_LONG cuda
    #define HANDLE_TYPE(type) cu##type##_t
    #define DEVICE_PROP struct cudaDeviceProp
#else // HIP case
    #define PREFIX HIP
    #define PREFIX_LC hip
    #define PREFIX_LC_LONG hip
    #define HANDLE_TYPE(type) hip##type##_t
    #define DEVICE_PROP struct hipDeviceProp
#endif
#define CONCATENATE(a, b) a##b

// CUDA error checking
static inline void CONCATENATE(PREFIX_LC_LONG,_safe_call)(CONCATENATE(PREFIX_LC_LONG,Error_t) cuerr, int dev_index,
                                  const char *file, const int line) {
  if (cuerr != CONCATENATE(PREFIX_LC_LONG,Success) && cuerr != 1) {
    firestarter::log::error()
        << "CUDA error at " << file << ":" << line << ": error code = " << cuerr
        << " (" << cudaGetErrorString(cuerr)
        << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}

static const char *_cudaGetErrorEnum(HANDLE_TYPE(blasStatus) error) {
    switch (error) {
        CONCATENATE(PREFIX,BLAS_SUCCESS): return "success";
        CONCATENATE(PREFIX,BLAS_NOT_INITIALIZED): return "not initialized";
        CONCATENATE(PREFIX,BLAS_ALLOC_FAILED): return "alloc failed";
        CONCATENATE(PREFIX,BLAS_INVALID_VALUE): return "invalid value";
        CONCATENATE(PREFIX,BLAS_ARCH_MISMATCH): return "arch mismatch";
        CONCATENATE(PREFIX,BLAS_MAPPING_ERROR): return "mapping error";
        CONCATENATE(PREFIX,BLAS_EXECUTION_FAILED): return "execution failed";
        CONCATENATE(PREFIX,BLAS_INTERNAL_ERROR): return "internal error";
        CONCATENATE(PREFIX,BLAS_NOT_SUPPORTED): return "not supported";
#ifdef FS_USE_HIP  // only avail for HIP
        CONCATENATE(PREFIX,BLAS_UNKNOWN): return "unknown";
        CONCATENATE(PREFIX,BLAS_HANDLE_IS_NULLPTR): return "handle is nullptr";
        CONCATENATE(PREFIX,BLAS_INVALID_ENUM): return "invalid enum";
#else  // only avail for CUDA
        CONCATENATE(PREFIX,BLAS_LICENSE_ERROR): return "license error";
#endif  // end only avail for some arch
        default: return "<unknown>";
    }

  return "<unknown>";
}

static inline void cuda_safe_call(HANDLE_TYPE(blasStatus) cuerr, int dev_index,
                                  const char *file, const int line) {
  if (cuerr != CONCATENATE(PREFIX,BLAS_STATUS_SUCCESS)) {
    firestarter::log::error()
        << "CUBLAS error at " << file << ":" << line
        << ": error code = " << cuerr << " (" << _cudaGetErrorEnum(cuerr)
        << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}


static const char *_curandGetErrorEnum(HANDLE_TYPE(randStatus) cuerr) {
  switch (cuerr) {
    CONCATENATE(PREFIX,RAND_SUCCESS): return "success";
    CONCATENATE(PREFIX,RAND_VERSION_MISMATCH): return "version mismatch";
    CONCATENATE(PREFIX,RAND_NOT_INITIALIZED): return "not initialized";
    CONCATENATE(PREFIX,RAND_ALLOCATION_FAILED): return "alloc failed";
    CONCATENATE(PREFIX,RAND_TYPE_ERROR): return "type error";
    CONCATENATE(PREFIX,RAND_OUT_OF_RANGE): return "out of range";
    CONCATENATE(PREFIX,RAND_LENGTH_NOT_MULTIPLE): return "length not multiple";
    CONCATENATE(PREFIX,RAND_DOUBLE_PRECISION_REQUIRED): return "double precision required";
    CONCATENATE(PREFIX,RAND_LAUNCH_FAILURE): return "launch failure";
    CONCATENATE(PREFIX,RAND_PREEXISTING_FAILURE): return "preexisting failure";
    CONCATENATE(PREFIX,RAND_INITIALIZATION_FAILED): return "initialization failed";
    CONCATENATE(PREFIX,RAND_ARCH_MISMATCH): return "arch mismatch";
    CONCATENATE(PREFIX,RAND_INTERNAL_ERROR): return "internal error";
#ifdef FS_USE_HIP // only avail for HIP
    CONCATENATE(PREFIX,RAND_NOT_IMPLEMENTED: return "not implemented";
#endif // only avail for HIP
  }

  return "<unknown>";
}

static inline void cuda_safe_call(HANDLE_TYPE(randStatus), int dev_index,
                                  const char *file, const int line) {
  if (cuerr != CONCATENATE(PREFIX,RAND_STATUS_SUCCESS)) {
    firestarter::log::error()
        << "cuRAND error at " << file << ":" << line
        << ": error code = " << cuerr << " (" << _curandGetErrorEnum(cuerr)
        << "), device index: " << dev_index;
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

#if (CUDART_VERSION >= 8000)
// read precision ratio (dp/sp) of GPU to choose the right variant for maximum
// workload
static int get_precision(int useDouble, struct CONCATENATE(PREFIX_LC_LONG,DeviceProp) properties) {
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
static int get_precision(int useDouble, DEVICE_PROP properties) {
  (void)properties;

  if (useDouble) {
    return 1;
  } else {
    return 0;
  }
}
#endif

#ifndef 
static int get_precision(int device_index, int useDouble) {
  CUcontext context;
  CUdevice device;
  size_t memory_avail, memory_total;
  struct CONCATENATE(PREFIX_LC_LONG,DeviceProp) properties;
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC_LONG,GetDeviceProperties)(&properties, device_index),
                 device_index);

  useDouble = get_precision(useDouble, properties);

  // we check for double precision support on the GPU and print errormsg, when
  // the user wants to compute DP on a SP-only-Card.
  if (useDouble && properties.major <= 1 && properties.minor <= 2) {
    std::stringstream ss;
    ss << "GPU " << device_index << ": " << properties.name << " ";

    firestarter::log::error()
        << ss.str() << "Doesn't support double precision.\n"
        << ss.str() << "Compute Capability: " << properties.major << "."
        << properties.minor << ". Requiered for double precision: >=1.3\n"
        << ss.str()
        << "Stressing with single precision instead. Maybe use -f parameter.";

    useDouble = 0;
  }
  return useDouble;
}

static int get_msize(int device_index, int useDouble) {
  CUcontext context;
  CUdevice device;
  size_t memory_avail, memory_total;

  CUDA_SAFE_CALL(cuDeviceGet(&device, device_index), device_index);
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);
  CUDA_SAFE_CALL(cuCtxSetCurrent(context), device_index);
  CUDA_SAFE_CALL(cuMemGetInfo(&memory_avail, &memory_total), device_index);

  CUDA_SAFE_CALL(cuCtxDestroy(context), device_index);

  return round_up(
      (int)(0.8 * sqrt(((memory_avail) /
                        ((useDouble ? sizeof(double) : sizeof(float)) * 3)))),
      1024); // a multiple of 1024 works always well
}

static CONCATENATE(PREFIX_LC,blasStatus_t) gemm(CONCATENATE(PREFIX_LC,blasHandle_t) handle, CONCATENATE(PREFIX_LC,blasOperation_t) transa,
                           CONCATENATE(PREFIX_LC,blasOperation_t) transb, int &m, int &n, int &k,
                           const float *alpha, const float *A, int &lda,
                           const float *B, int &ldb, const float *beta,
                           float *C, int &ldc) {
  return CONCATENATE(PREFIX_LC,blasSgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

static CONCATENATE(PREFIX_LC,blasStatus_t) gemm(CONCATENATE(PREFIX_LC,blasHandle_t) handle, CONCATENATE(PREFIX_LC,blasOperation_t) transa,
                           CONCATENATE(PREFIX_LC,blasOperation_t) transb, int &m, int &n, int &k,
                           const double *alpha, const double *A, int &lda,
                           const double *B, int &ldb, const double *beta,
                           double *C, int &ldc) {
  return CONCATENATE(PREFIX_LC,blasDgemm)(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

static CONCATENATE(PREFIX_LC,randStatus_t) generateUniform(CONCATENATE(PREFIX_LC,randGenerator_t) generator,
                                      float *outputPtr, size_t num) {
  return CONCATENATE(PREFIX_LC,randGenerateUniform)(generator, outputPtr, num);
}

static CONCATENATE(PREFIX_LC,randStatus_t) generateUniform(CONCATENATE(PREFIX_LC,randGenerator_t) generator,
                                      double *outputPtr, size_t num) {
  return CONCATENATE(PREFIX_LC,randGenerateUniformDouble)(generator, outputPtr, num);
}

// GPU index. Used to pin this thread to the GPU.
template <typename T>
static void create_load(std::condition_variable &waitForInitCv,
                        std::mutex &waitForInitCvMutex, int device_index,
                        std::atomic<int> &initCount,
                        volatile unsigned long long *loadVar, int matrixSize) {
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "create_load<T>: Template argument T must be either float or double");

  int iterations, i;

#ifndef FS_USE_HIP
  firestarter::log::trace() << "Starting CUDA with given matrix size "
                            << matrixSize;
#else
  firestarter::log::trace() << "Starting HIP with given matrix size "
                            << matrixSize;
#endif
  size_t size_use = 0;
  if (matrixSize > 0) {
    size_use = matrixSize;
  }

  size_t use_bytes, memory_size;

#ifndef FS_USE_HIP // CUDA
  CUcontext context;
  struct cudaDeviceProp properties;
  CUdevice device;
  cublasHandle_t cublas;
#else // HIP
  struct hipDeviceProp_t properties;
  hipDevice_t device;
  hipblasHandle_t cublas;
#endif

  // reserving the GPU and initializing cublas

  firestarter::log::trace() << "Getting accelerator device nr. " << device_index;
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,DeviceGet)(&device, device_index), device_index);

#ifndef FS_USE_HIP
  firestarter::log::trace() << "Creating CUDA context for computation on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);

  firestarter::log::trace() << "Set created CUDA context on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(cuCtxSetCurrent(context), device_index);
#else
  firestarter::log::trace() << "Set HIP device for computation on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(hipSetDevice(device_index), device_index);
  firestarter::log::trace() << "Create HIP stream on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(hipStreamCreate(&stream), device_index);
#endif
  firestarter::log::trace() << "Create BLAS on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,blasCreate)(&cublas), device_index);

  firestarter::log::trace() << "Get accelerator device properties (e.g., support for double)"
                     << " on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC_LONG,GetDeviceProperties)(&properties, device_index),
                 device_index);

  // getting information about the GPU memory
  size_t memory_avail, memory_total;
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC_LONG,MemGetInfo)(&memory_avail, &memory_total), device_index);

  firestarter::log::trace() << "Get accelerator Memory info on device nr. "
                     << device_index
                     <<": " << memory_avail << " B avail. from "
                     << memory_total << " B total";

  // defining memory pointers
#ifndef FS_USE_HIP // CUDA
  CUdeviceptr a_data_ptr;
  CUdeviceptr b_data_ptr;
  CUdeviceptr c_data_ptr;
#else // HIP
  T* a_data_ptr;
  T* b_data_ptr;
  T* c_data_ptr;
#endif

  // check if the user has not set a matrix OR has set a too big matrixsite and
  // if this is true: set a good matrixsize
  if (!size_use || ((size_use * size_use * sizeof(T) * 3 > memory_avail))) {
    size_use = round_up((int)(0.8 * sqrt(((memory_avail) / (sizeof(T) * 3)))),
                        1024); // a multiple of 1024 works always well
  }
  firestarter::log::trace() << "Set accelerator matrix size: " << matrixSize;
  use_bytes = (size_t)((T)memory_avail);
  memory_size = sizeof(T) * size_use * size_use;
  iterations = (use_bytes - 2 * memory_size) / memory_size; // = 1;

  firestarter::log::trace()
      << "Allocating accelerator memory on device nr. "
      << device_index;

  // allocating memory on the GPU
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,MemAlloc)(&a_data_ptr, memory_size), device_index);
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,MemAlloc)(&b_data_ptr, memory_size), device_index);
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,MemAlloc)(&c_data_ptr, iterations * memory_size),
                 device_index);

  firestarter::log::trace() << "Allocated accelerator memory on device nr. "
                     << device_index
                     <<". A: " << a_data_ptr << "(Size: "
                     << memory_size << "B)"
                     << "\n";

  firestarter::log::trace() << "Allocated accelerator memory on device nr. "
                     << device_index
                     <<". B: " << b_data_ptr << "(Size: "
                     << memory_size << "B)"
                     << "\n";
  firestarter::log::trace() << "Allocated accelerator memory on device nr. "
                     << device_index
                     <<". C: " << c_data_ptr << "(Size: "
                     << iterations * memory_size << "B)"
                     << "\n";

  firestarter::log::trace() << "Initializing accelerator matrices a, b on device nr. "
                            << device_index
                            << ". Using "
                            << size_use * size_use
                            << " elements of size "
                            << sizeof(T) << " Byte";
  // initialize matrix A and B on the GPU with random values
  CONCATENATE(PREFIX_LC,randGenerator_t) random_gen;
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,randCreateGenerator)(&random_gen, CURAND_RNG_PSEUDO_DEFAULT),
                 device_index);
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,randSetPseudoRandomGeneratorSeed)(random_gen, SEED),
                 device_index);
  CUDA_SAFE_CALL(
      generateUniform(random_gen, (T *)a_data_ptr, size_use * size_use),
      device_index);
  CUDA_SAFE_CALL(
      generateUniform(random_gen, (T *)b_data_ptr, size_use * size_use),
      device_index);
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,randDestroyGenerator)(random_gen), device_index);

  // initialize c_data_ptr with copies of A
  for (i = 0; i < iterations; i++) {
      firestarter::log::trace() << "Initializing accelerator matrix c-"
                                << i
                                << " by copying "
                                << memory_size
                                << " byte from "
                                << a_data_ptr
                                << " to "
                                << c_data_ptr + (size_t)(i * size_use * size_use * (float)sizeof(T)/(float)sizeof(c_data_ptr))
                                << "\n";
    CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,MemcpyDtoD)(c_data_ptr + (size_t)(i * size_use * size_use * (float)sizeof(T)/(float)sizeof(c_data_ptr)),
                                a_data_ptr, memory_size),
                   device_index);
  }

  // save gpuvar->init_count and sys.out
  {
    std::lock_guard<std::mutex> lk(waitForInitCvMutex);

#define TO_MB(x) (unsigned long)(x / 1024 / 1024)
  firestarter::log::info()
      << "   GPU " << device_index << "\n"
      << "    name:           " << properties.name << "\n"
      << "    memory:         " << TO_MB(memory_avail) << "/"
      << TO_MB(memory_total) << " MiB available (using " << TO_MB(use_bytes)
      << " MiB)\n"
      << "    matrix size:    " << size_use << "\n"
      << "    used precision: "
      << ((sizeof(T) == sizeof(double)) ? "double" : "single");
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
      CUDA_SAFE_CALL(gemm(PCONCATENATE(PREFIX_LC,blas), CONCATENATE(PREFIX,BLAS_OP_N), CONCATENATE(PREFIX,BLAS_OP_N), size_use_i, size_use_i,
                          size_use_i, &alpha, (const T *)a_data_ptr, size_use_i,
                          (const T *)b_data_ptr, size_use_i, &beta,
                          (T *)c_data_ptr + i * size_use * size_use, size_use_i),
                     device_index);
      CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC_LONG,DeviceSynchronize)(), device_index);
    }
  }

  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,MemFree)(a_data_ptr), device_index);
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,MemFree)(b_data_ptr), device_index);
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,MemFree)(c_data_ptr), device_index);
  CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,blasDestroy)(cublas), device_index);
#ifndef FS_USE_HIP // CUDA
  CUDA_SAFE_CALL(cuCtxDestroy(context), device_index);
#else // HIP
  CUDA_SAFE_CALL(hipStreamDestroy(stream), device_index);
#endif
}

Cuda::Cuda(volatile unsigned long long *loadVar, bool useFloat, bool useDouble,
           unsigned matrixSize, int gpus) {
  std::thread t(Cuda::initGpus, std::ref(_waitForInitCv), loadVar, useFloat,
                useDouble, matrixSize, gpus);
  _initThread = std::move(t);

  std::unique_lock<std::mutex> lk(_waitForInitCvMutex);
  // wait for gpus to initialize
  _waitForInitCv.wait(lk);
}

void Cuda::initGpus(std::condition_variable &cv,
                    volatile unsigned long long *loadVar, bool useFloat,
                    bool useDouble, unsigned matrixSize, int gpus) {
  std::condition_variable waitForInitCv;
  std::mutex waitForInitCvMutex;

  if (gpus) {
    CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,Init)(0), -1);
    int devCount;
    CUDA_SAFE_CALL(CONCATENATE(PREFIX_LC,DeviceGetCount)(&devCount), -1);

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
#ifndef FS_USE_HIP // CUDA
        firestarter::log::warn()
            << "You requested more CUDA devices than available. "
               "Maybe you set CUDA_VISIBLE_DEVICES?";
        firestarter::log::warn()
            << "FIRESTARTER will use " << devCount << " of the requested "
            << gpus << " CUDA device(s)";
#else // HIP
        firestarter::log::warn()
            << "You requested more HIP devices than available.";
        firestarter::log::warn()
            << "FIRESTARTER will use " << devCount << " of the requested "
            << gpus << " HIP device(s)";
#endif
        gpus = devCount;
      }

      {
        std::lock_guard<std::mutex> lk(waitForInitCvMutex);

        for (int i = 0; i < gpus; ++i) {
          // if there's a GPU in the system without Double Precision support, we
          // have to correct this.
          int precision = get_precision(i, use_double);

          if (precision) {
            std::thread t(create_load<double>, std::ref(waitForInitCv),
                          std::ref(waitForInitCvMutex), i, std::ref(initCount),
                          loadVar, (int)matrixSize);
            gpuThreads.push_back(std::move(t));
          } else {
            std::thread t(create_load<float>, std::ref(waitForInitCv),
                          std::ref(waitForInitCvMutex), i, std::ref(initCount),
                          loadVar, (int)matrixSize);
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
      for (auto &t : gpuThreads) {
        t.join();
      }
    } else {
#ifndef FS_USE_HIP // CUDA
      firestarter::log::info()
          << "    - No CUDA devices. Just stressing CPU(s). Maybe use "
             "FIRESTARTER instead of FIRESTARTER_CUDA?";
#else // HIP
          << "    - No HIP devices. Just stressing CPU(s). Maybe use "
             "FIRESTARTER instead of FIRESTARTER_HIP?";
#endif
      cv.notify_all();
    }
  } else {
#ifndef FS_USE_HIP // CUDA
    firestarter::log::info()
        << "    --gpus 0 is set. Just stressing CPU(s). Maybe use "
           "FIRESTARTER instead of FIRESTARTER_HIP?";
#else // HIP
        << "    --gpus 0 is set. Just stressing CPU(s). Maybe use "
           "FIRESTARTER instead of FIRESTARTER_HIP?";
#endif
    cv.notify_all();
  }
}
