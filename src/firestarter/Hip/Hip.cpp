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

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hiprand_kernel.h>

#include <algorithm>
#include <atomic>
#include <type_traits>

#define CUDA_SAFE_CALL(cuerr, dev_index)                                       \
  cuda_safe_call(cuerr, dev_index, __FILE__, __LINE__)
#define HIPRAND_SAFE_CALL(cuerr, dev_index)                                    \
  hiprand_safe_call(cuerr, dev_index, __FILE__, __LINE__)
#define HIPBLAS_SAFE_CALL(cuerr, dev_index)                                    \
  hipblas_safe_call(cuerr, dev_index, __FILE__, __LINE__)

#define SEED 123

using namespace firestarter::cuda;

// CUDA error checking
static inline void cuda_safe_call(hipError_t cuerr, int dev_index,
                                  const char *file, const int line) {
  if (cuerr != hipSuccess && cuerr != 1) {
    firestarter::log::error()
        << "HIP error at " << file << ":" << line << ": error code = " << cuerr
        << " (" << hipGetErrorString(cuerr)
        << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}


static const char *_cudaGetErrorEnum(hipblasStatus_t error) {
  switch (error) {
  case HIPBLAS_STATUS_SUCCESS:
    return "HIPBLAS_STATUS_SUCCESS";
  case HIPBLAS_STATUS_NOT_INITIALIZED:
    return "HIPBLAS_STATUS_NOT_INITIALIZED";
  case HIPBLAS_STATUS_ALLOC_FAILED:
    return "HIPBLAS_STATUS_ALLOC_FAILED";
  case HIPBLAS_STATUS_INVALID_VALUE:
    return "HIPBLAS_STATUS_INVALID_VALUE";
  case HIPBLAS_STATUS_ARCH_MISMATCH:
    return "HIPBLAS_STATUS_ARCH_MISMATCH";
  case HIPBLAS_STATUS_MAPPING_ERROR:
    return "HIPBLAS_STATUS_MAPPING_ERROR";
  case HIPBLAS_STATUS_EXECUTION_FAILED:
    return "HIPBLAS_STATUS_EXECUTION_FAILED";
  case HIPBLAS_STATUS_INTERNAL_ERROR:
    return "HIPBLAS_STATUS_INTERNAL_ERROR";
  case HIPBLAS_STATUS_NOT_SUPPORTED:
    return "HIPBLAS_STATUS_NOT_SUPPORTED";
  case HIPBLAS_STATUS_UNKNOWN:
    return "HIPBLAS_STATUS_UNKNOWN";
  }

  return "<unknown>";
}

static inline void hipblas_safe_call(hipblasStatus_t cuerr, int dev_index,
				     const char *file, const int line) {
  if (cuerr != HIPBLAS_STATUS_SUCCESS) {
    firestarter::log::error()
        << "HIPBLAS error at " << file << ":" << line
        << ": error code = " << cuerr << " (" << _cudaGetErrorEnum(cuerr)
        << "), device index: " << dev_index;
    exit(cuerr);
  }

  return;
}

static const char *_curandGetErrorEnum(hiprandStatus_t cuerr) {
  switch (cuerr) {
  case HIPRAND_STATUS_SUCCESS:
    return "HIPRAND_STATUS_SUCCESS";
  case HIPRAND_STATUS_VERSION_MISMATCH:
    return "HIPRAND_STATUS_VERSION_MISMATCH";
  case HIPRAND_STATUS_NOT_INITIALIZED:
    return "HIPRAND_STATUS_NOT_INITIALIZED";
  case HIPRAND_STATUS_ALLOCATION_FAILED:
    return "HIPRAND_STATUS_ALLOCATION_FAILED";
  case HIPRAND_STATUS_TYPE_ERROR:
    return "HIPRAND_STATUS_TYPE_ERROR";
  case HIPRAND_STATUS_OUT_OF_RANGE:
    return "HIPRAND_STATUS_OUT_OF_RANGE";
  case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE";
  case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case HIPRAND_STATUS_LAUNCH_FAILURE:
    return "HIPRAND_STATUS_LAUNCH_FAILURE";
  case HIPRAND_STATUS_PREEXISTING_FAILURE:
    return "HIPRAND_STATUS_PREEXISTING_FAILURE";
  case HIPRAND_STATUS_INITIALIZATION_FAILED:
    return "HIPRAND_STATUS_INITIALIZATION_FAILED";
  case HIPRAND_STATUS_ARCH_MISMATCH:
    return "HIPRAND_STATUS_ARCH_MISMATCH";
  case HIPRAND_STATUS_INTERNAL_ERROR:
    return "HIPRAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

static inline void hiprand_safe_call(hiprandStatus_t cuerr, int dev_index,
                                  const char *file, const int line) {
  if (cuerr != HIPRAND_STATUS_SUCCESS) {
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
static int get_precision(int useDouble, struct hipDeviceProp_t properties) {
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
static int get_precision(int useDouble, struct hipDeviceProp_t properties) {
  (void)properties;

  if (useDouble) {
    return 1;
  } else {
    return 0;
  }
}
#endif

static int get_precision(int device_index, int useDouble) {
  hipCtx_t context;
  hipDevice_t device;
  size_t memory_avail, memory_total;
  struct hipDeviceProp_t properties;

  CUDA_SAFE_CALL(hipDeviceGet(&device, device_index), device_index);
  CUDA_SAFE_CALL(hipCtxCreate(&context, 0, device), device_index);
  CUDA_SAFE_CALL(hipCtxSetCurrent(context), device_index);
  CUDA_SAFE_CALL(hipMemGetInfo(&memory_avail, &memory_total), device_index);
  CUDA_SAFE_CALL(hipGetDeviceProperties(&properties, device_index),
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

  CUDA_SAFE_CALL(hipCtxDestroy(context), device_index);

  return useDouble;
}

static hipblasStatus_t gemm(hipblasHandle_t handle, hipblasOperation_t transa,
                           hipblasOperation_t transb, int &m, int &n, int &k,
                           const float *alpha, const float *A, int &lda,
                           const float *B, int &ldb, const float *beta,
                           float *C, int &ldc) {
  return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

static hipblasStatus_t gemm(hipblasHandle_t handle, hipblasOperation_t transa,
                           hipblasOperation_t transb, int &m, int &n, int &k,
                           const double *alpha, const double *A, int &lda,
                           const double *B, int &ldb, const double *beta,
                           double *C, int &ldc) {
  return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

static hiprandStatus_t generateUniform(hiprandGenerator_t generator,
                                      float *outputPtr, size_t num) {
  return hiprandGenerateUniform(generator, outputPtr, num);
}

static hiprandStatus_t generateUniform(hiprandGenerator_t generator,
                                      double *outputPtr, size_t num) {
  return hiprandGenerateUniformDouble(generator, outputPtr, num);
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

  firestarter::log::trace() << "Starting CUDA with given matrix size "
                            << matrixSize;

  size_t size_use = 0;
  if (matrixSize > 0) {
    size_use = matrixSize;
  }

  hipCtx_t context;
  size_t use_bytes, memory_size;
  struct hipDeviceProp_t properties;

  // reserving the GPU and initializing cublas
  hipDevice_t device;
  hipblasHandle_t cublas;

  firestarter::log::trace() << "Getting CUDA device nr. " << device_index;
  CUDA_SAFE_CALL(hipDeviceGet(&device, device_index), device_index);

  firestarter::log::trace() << "Creating CUDA context for computation on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(hipCtxCreate(&context, 0, device), device_index);

  firestarter::log::trace() << "Set crated CUDA context on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(hipCtxSetCurrent(context), device_index);

  firestarter::log::trace() << "Create CuBlas on device nr. "
                     << device_index;
  HIPBLAS_SAFE_CALL(hipblasCreate(&cublas), device_index);

  firestarter::log::trace() << "Get CUDA device properties (e.g., support for double)"
                     << " on device nr. "
                     << device_index;
  CUDA_SAFE_CALL(hipGetDeviceProperties(&properties, device_index),
                 device_index);

  // getting information about the GPU memory
  size_t memory_avail, memory_total;
  CUDA_SAFE_CALL(hipMemGetInfo(&memory_avail, &memory_total), device_index);

  firestarter::log::trace() << "Get CUDA Memory info on device nr. "
                     << device_index
                     <<": " << memory_avail << " B avail. from "
                     << memory_total << " B total";

  // defining memory pointers
  T* a_data_ptr;
  T* b_data_ptr;
  T* c_data_ptr;

  // check if the user has not set a matrix OR has set a too big matrixsite and
  // if this is true: set a good matrixsize
  if (!size_use || ((size_use * size_use * sizeof(T) * 3 > memory_avail))) {
    size_use = round_up((int)(0.8 * sqrt(((memory_avail) / (sizeof(T) * 3)))),
                        1024); // a multiple of 1024 works always well
  }
  firestarter::log::trace() << "Set CUDA matrix size: " << matrixSize;
  use_bytes = (size_t)((T)memory_avail);
  memory_size = sizeof(T) * size_use * size_use;
  iterations = (use_bytes - 2 * memory_size) / memory_size; // = 1;

  firestarter::log::trace()
      << "Allocating CUDA memory on device nr. "
      << device_index;

  // allocating memory on the GPU
  CUDA_SAFE_CALL(hipMalloc(&a_data_ptr, memory_size), device_index);
  CUDA_SAFE_CALL(hipMalloc(&b_data_ptr, memory_size), device_index);
  CUDA_SAFE_CALL(hipMalloc(&c_data_ptr, iterations * memory_size),
                 device_index);

  firestarter::log::trace() << "Allocated CUDA memory on device nr. "
                     << device_index
                     <<". A: " << a_data_ptr << "(Size: "
                     << memory_size << "B)"
                     << "\n";

  firestarter::log::trace() << "Allocated CUDA memory on device nr. "
                     << device_index
                     <<". B: " << b_data_ptr << "(Size: "
                     << memory_size << "B)"
                     << "\n";
  firestarter::log::trace() << "Allocated CUDA memory on device nr. "
                     << device_index
                     <<". C: " << c_data_ptr << "(Size: "
                     << iterations * memory_size << "B)"
                     << "\n";

  firestarter::log::trace() << "Initializing CUDA matrices a, b on device nr. "
                            << device_index
                            << ". Using "
                            << size_use * size_use
                            << " elements of size "
                            << sizeof(T) << " Byte";
  // initialize matrix A and B on the GPU with random values
  hiprandGenerator_t random_gen;
  HIPRAND_SAFE_CALL(hiprandCreateGenerator(&random_gen, HIPRAND_RNG_PSEUDO_DEFAULT),
                 device_index);
  HIPRAND_SAFE_CALL(hiprandSetPseudoRandomGeneratorSeed(random_gen, SEED),
                 device_index);
  HIPRAND_SAFE_CALL(
      generateUniform(random_gen, (T *)a_data_ptr, size_use * size_use),
      device_index);
  HIPRAND_SAFE_CALL(
      generateUniform(random_gen, (T *)b_data_ptr, size_use * size_use),
      device_index);
  HIPRAND_SAFE_CALL(hiprandDestroyGenerator(random_gen), device_index);

  // initialize c_data_ptr with copies of A
  for (i = 0; i < iterations; i++) {
      firestarter::log::trace() << "Initializing CUDA matrix c-"
                                << i
                                << " by copying "
                                << memory_size
                                << " byte from "
                                << a_data_ptr
                                << " to "
				<< c_data_ptr + i * size_use * size_use
                                << "\n";
      CUDA_SAFE_CALL(hipMemcpyDtoD(c_data_ptr + i * size_use * size_use,
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
      HIPBLAS_SAFE_CALL(gemm(cublas, HIPBLAS_OP_N, HIPBLAS_OP_N, size_use_i, size_use_i,
                          size_use_i, &alpha, (const T *)a_data_ptr, size_use_i,
                          (const T *)b_data_ptr, size_use_i, &beta,
                          (T *)c_data_ptr + i * size_use * size_use, size_use_i),
                     device_index);
      CUDA_SAFE_CALL(hipDeviceSynchronize(), device_index);
    }
  }

  CUDA_SAFE_CALL(hipFree(a_data_ptr), device_index);
  CUDA_SAFE_CALL(hipFree(b_data_ptr), device_index);
  CUDA_SAFE_CALL(hipFree(c_data_ptr), device_index);
  HIPBLAS_SAFE_CALL(hipblasDestroy(cublas), device_index);
  CUDA_SAFE_CALL(hipCtxDestroy(context), device_index);
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
    CUDA_SAFE_CALL(hipInit(0), -1);
    int devCount;
    CUDA_SAFE_CALL(hipGetDeviceCount(&devCount), -1);

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
        firestarter::log::warn()
            << "You requested more CUDA devices than available. "
               "Maybe you set CUDA_VISIBLE_DEVICES?";
        firestarter::log::warn()
            << "FIRESTARTER will use " << devCount << " of the requested "
            << gpus << " CUDA device(s)";
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
