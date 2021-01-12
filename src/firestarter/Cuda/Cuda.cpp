/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <type_traits>

#define CUDA_SAFE_CALL(cuerr, dev_index)                                       \
  cuda_safe_call(cuerr, dev_index, __FILE__, __LINE__)
#define SEED 123

using namespace firestarter::cuda;

// CUDA error checking
static inline void cuda_safe_call(cudaError_t cuerr, int dev_index,
                                  const char *file, const int line) {
  if (cuerr != cudaSuccess && cuerr != 1) {
    firestarter::log::error()
        << "CUDA error at " << file << ":" << line << ": error code = " << cuerr
        << " (" << cudaGetErrorString(cuerr)
        << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

static inline void cuda_safe_call(cublasStatus_t cuerr, int dev_index,
                                  const char *file, const int line) {
  if (cuerr != CUBLAS_STATUS_SUCCESS) {
    firestarter::log::error()
        << "CUBLAS error at " << file << ":" << line
        << ": error code = " << cuerr << " (" << _cudaGetErrorEnum(cuerr)
        << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}

static inline void cuda_safe_call(CUresult cuerr, int dev_index,
                                  const char *file, const int line) {
  if (cuerr != CUDA_SUCCESS) {
    const char *errorString;

    CUDA_SAFE_CALL(cuGetErrorName(cuerr, &errorString), dev_index);

    firestarter::log::error()
        << "CUDA error at " << file << ":" << line << ": error code = " << cuerr
        << " (" << errorString << "), device index: " << dev_index;
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

template <typename T> static T *allocateAndFillup(int size) {
  auto array = static_cast<T *>(malloc(sizeof(T) * size * size));
  if (!array) {
    firestarter::log::error()
        << "Could not allocate memory for GPU computation";
    exit(ENOMEM);
  }
  std::srand(SEED);
  for (int i = 0; i < size * size; ++i) {
    array[i] = (T)(std::rand() % 1000000) / 100000.0;
  }
  return array;
}

#if (CUDART_VERSION >= 8000)
// read precision ratio (dp/sp) of GPU to choose the right variant for maximum
// workload
static int get_precision(int useDouble, struct cudaDeviceProp properties) {
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
static int get_precision(int useDouble, struct cudaDeviceProp properties) {
  (void)properties;

  if (useDouble) {
    return 1;
  } else {
    return 0;
  }
}
#endif

static int get_precision(int device_index, int useDouble) {
  CUcontext context;
  CUdevice device;
  size_t memory_avail, memory_total;
  struct cudaDeviceProp properties;

  CUDA_SAFE_CALL(cuDeviceGet(&device, device_index), device_index);
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);
  CUDA_SAFE_CALL(cuCtxSetCurrent(context), device_index);
  CUDA_SAFE_CALL(cuMemGetInfo(&memory_avail, &memory_total), device_index);
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device_index),
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

  CUDA_SAFE_CALL(cuCtxDestroy(context), device_index);

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

static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int &m, int &n, int &k,
                           const float *alpha, const float *A, int &lda,
                           const float *B, int &ldb, const float *beta,
                           float *C, int &ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int &m, int &n, int &k,
                           const double *alpha, const double *A, int &lda,
                           const double *B, int &ldb, const double *beta,
                           double *C, int &ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

// GPU index. Used to pin this thread to the GPU.
template <typename T>
static void create_load(std::condition_variable &waitForInitCv,
                        std::mutex &waitForInitCvMutex, int device_index,
                        std::atomic<int> &initCount,
                        volatile unsigned long long *loadVar, int matrixSize,
                        T *A, T *B) {
  static_assert(
      std::is_same<T, float>::value || std::is_same<T, double>::value,
      "create_load<T>: Template argument T must be either float or double");

  int iterations, i;

  int size_use = 0;
  if (matrixSize > 0) {
    size_use = matrixSize;
  }

  CUcontext context;
  size_t use_bytes, memory_size;
  struct cudaDeviceProp properties;

  // reserving the GPU and initializing cublas
  CUdevice device;
  cublasHandle_t cublas;
  CUDA_SAFE_CALL(cuDeviceGet(&device, device_index), device_index);
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);
  CUDA_SAFE_CALL(cuCtxSetCurrent(context), device_index);
  CUDA_SAFE_CALL(cublasCreate(&cublas), device_index);
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device_index),
                 device_index);

  // getting information about the GPU memory
  size_t memory_avail, memory_total;
  CUDA_SAFE_CALL(cuMemGetInfo(&memory_avail, &memory_total), device_index);

  // defining memory pointers
  CUdeviceptr a_data_ptr;
  CUdeviceptr b_data_ptr;
  CUdeviceptr c_data_ptr;

  // check if the user has not set a matrix OR has set a too big matrixsite and
  // if this is true: set a good matrixsize
  if (!size_use || ((size_use * size_use * sizeof(T) * 3 > memory_avail))) {
    size_use = round_up((int)(0.8 * sqrt(((memory_avail) / (sizeof(T) * 3)))),
                        1024); // a multiple of 1024 works always well
  }
  use_bytes = (size_t)((T)memory_avail);
  memory_size = sizeof(T) * size_use * size_use;
  iterations = (use_bytes - 2 * memory_size) / memory_size; // = 1;

  // allocating memory on the GPU
  CUDA_SAFE_CALL(cuMemAlloc(&a_data_ptr, memory_size), device_index);
  CUDA_SAFE_CALL(cuMemAlloc(&b_data_ptr, memory_size), device_index);
  CUDA_SAFE_CALL(cuMemAlloc(&c_data_ptr, iterations * memory_size),
                 device_index);

  // moving matrices A and B to the GPU
  CUDA_SAFE_CALL(cuMemcpyHtoD(a_data_ptr, A, memory_size), device_index);
  CUDA_SAFE_CALL(cuMemcpyHtoD(b_data_ptr, B, memory_size), device_index);

  // initialize c_data_ptr with copies of A
  for (i = 0; i < iterations; i++) {
    CUDA_SAFE_CALL(
        cuMemcpyHtoD(c_data_ptr + i * size_use * size_use, A, memory_size),
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
        << TO_MB(memory_total) << " MB available (using " << TO_MB(use_bytes)
        << " MB)\n"
        << "    matrix size:    " << size_use << "\n"
        << "    used precision: "
        << ((sizeof(T) == sizeof(double)) ? "double" : "single");
#undef TO_MB

    initCount++;
  }
  waitForInitCv.notify_all();

  const T alpha = 1.0;
  const T beta = 0.0;

  // actual stress begins here
  while (*loadVar != LOAD_STOP) {
    for (i = 0; i < iterations; i++) {
      CUDA_SAFE_CALL(gemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, size_use, size_use,
                          size_use, &alpha, (const T *)a_data_ptr, size_use,
                          (const T *)b_data_ptr, size_use, &beta,
                          (T *)c_data_ptr + i * size_use * size_use, size_use),
                     device_index);
      CUDA_SAFE_CALL(cudaDeviceSynchronize(), device_index);
    }
  }

  CUDA_SAFE_CALL(cuMemFree(a_data_ptr), device_index);
  CUDA_SAFE_CALL(cuMemFree(b_data_ptr), device_index);
  CUDA_SAFE_CALL(cuMemFree(c_data_ptr), device_index);
  CUDA_SAFE_CALL(cublasDestroy(cublas), device_index);
  CUDA_SAFE_CALL(cuCtxDestroy(context), device_index);
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
    CUDA_SAFE_CALL(cuInit(0), -1);
    int devCount;
    CUDA_SAFE_CALL(cuDeviceGetCount(&devCount), -1);

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

      firestarter::log::info() << "\n  graphics processor characteristics:";

      // use all GPUs if the user gave no information about use_device
      if (gpus < 0) {
        gpus = devCount;
      }

      if (gpus > devCount) {
        firestarter::log::warn()
            << "You requested more CUDA devices than available. "
               "Maybe you set CUDA_VISIBLE_DEVICES?";
        firestarter::log::warn()
            << "FIRESTARTER will use " << devCount << " of the requested "
            << gpus << " CUDA device(s)";
        gpus = devCount;
      }

      // we allocate two arrays A and B our inputs each for single and double
      // precision
      int maxSingleMatrixSize = 0;
      int maxDoubleMatrixSize = 0;

      std::vector<int> precisionVector;

      for (int i = 0; i < gpus; ++i) {
        int precision = get_precision(i, use_double);
        precisionVector.push_back(precision);
        int mSize = get_msize(i, precision);

        // DO NOT remove the parentheses aroud std::max
        // https://stackoverflow.com/a/2789509
        if (precision) {
          maxDoubleMatrixSize = (std::max)(maxDoubleMatrixSize, mSize);
        } else {
          maxSingleMatrixSize = (std::max)(maxSingleMatrixSize, mSize);
        }
      }

      auto ASingle = allocateAndFillup<float>(maxSingleMatrixSize);
      auto BSingle = allocateAndFillup<float>(maxSingleMatrixSize);

      auto ADouble = allocateAndFillup<double>(maxDoubleMatrixSize);
      auto BDouble = allocateAndFillup<double>(maxDoubleMatrixSize);

      {
        std::lock_guard<std::mutex> lk(waitForInitCvMutex);

        for (int i = 0; i < gpus; ++i) {
          // if there's a GPU in the system without Double Precision support, we
          // have to correct this.
          int precision = precisionVector[i];

          if (precision) {
            std::thread t(create_load<double>, std::ref(waitForInitCv),
                          std::ref(waitForInitCvMutex), i, std::ref(initCount),
                          loadVar, (int)matrixSize, ADouble, BDouble);
            gpuThreads.push_back(std::move(t));
          } else {
            std::thread t(create_load<float>, std::ref(waitForInitCv),
                          std::ref(waitForInitCvMutex), i, std::ref(initCount),
                          loadVar, (int)matrixSize, ASingle, BSingle);
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

      if (ASingle != nullptr) {
        free(ASingle);
      }
      if (BSingle != nullptr) {
        free(BSingle);
      }
      if (ADouble != nullptr) {
        free(ADouble);
      }
      if (BDouble != nullptr) {
        free(BDouble);
      }
    } else {
      firestarter::log::info()
          << "    - No CUDA devices. Just stressing CPU(s). Maybe use "
             "FIRESTARTER instead of FIRESTARTER_CUDA?";
      cv.notify_all();
    }
  } else {
    firestarter::log::info()
        << "    --gpus 0 is set. Just stressing CPU(s). Maybe use "
           "FIRESTARTER instead of FIRESTARTER_CUDA?";
    cv.notify_all();
  }
}
