/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2019 TU Dresden, Center for Information Services and High
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

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <errno.h>
#include <sys/types.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "gpu.h"
#include "firestarter_global.h"


#define CUDA_SAFE_CALL( cuerr, dev_index ) cuda_safe_call( cuerr, dev_index, __FILE__, __LINE__ )
#define SEED 123

static volatile gpustruct_t * gpuvar;
static void *A = NULL;
static void *B = NULL;
static int filled = 0;
static int max_msize = 0;

static pthread_cond_t wait_for_init_cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t wait_for_init_mutex = PTHREAD_MUTEX_INITIALIZER;


//CUDA error checking
static inline void cuda_safe_call( cudaError_t cuerr, int dev_index, const char * file, const int line ) {
    if ( cuerr != cudaSuccess && cuerr != 1)
    {
        fprintf( stderr, "    - CUDA error at %s:%i : error code = %i(%s), device index:%i\n",
                 file, line, cuerr, cudaGetErrorString( cuerr ), dev_index );
        exit(cuerr);
    }

    return;
}

static int round_up(int num_to_round, int multiple) {  
    if(multiple == 0) {  
        return num_to_round;
    }  

    int remainder = num_to_round % multiple; 
    if (remainder == 0) {
        return num_to_round;
    }

    return num_to_round + multiple - remainder; 
}


#define FILL_SUPPORT(DATATYPE,SIZE) \
    do { \
        int i; \
        DATATYPE *array = malloc(sizeof(DATATYPE)*SIZE*SIZE); \
        if(!array) { \
            fprintf(stderr, "Could not allocate memory for GPU computation\n"); \
            exit(ENOMEM); \
        } \
        srand48(SEED); \
        for (i=0; i < SIZE*SIZE; i++) { \
            array[i] = (DATATYPE) (lrand48() % 1000000) / 100000.0; \
        } \
        return array; \
    } \
    while( 0 ) \

static void* fillup(int useD, int size) {
    if(useD) {
        FILL_SUPPORT(double, size);
    }
    else {
        FILL_SUPPORT(float, size);
    }
}

#if ( CUDART_VERSION >= 8000 )  
//read precision ratio (dp/sp) of GPU to choose the right variant for maximum workload
static int get_precision(struct cudaDeviceProp properties) {
    if(gpuvar->use_double == 2 && properties.singleToDoublePrecisionPerfRatio > 3){
        return 0;
    } else if(gpuvar->use_double) {
        return 1;
    } else {
        return 0;
    }
}
#else
//as precision ratio is not supported return default/user input value  
static int get_precision(struct cudaDeviceProp properties) {
    if(gpuvar->use_double) {
        return 1;
    } else {
        return 0;
    }
}
#endif

static void* create_load(void * index) {
    int device_index = *((int*)index);   //GPU index. Used to pin this pthread to the GPU.
    int iterations, i;
    int pthread_use_double; //local per-thread variable, if there's a GPU in the system without Double Precision support.
    int size_use = 0;
    if (gpuvar->msize > 0){
        size_use = gpuvar->msize;
    }

    CUcontext context;
    size_t use_bytes, memory_size;
    struct cudaDeviceProp properties;

    //reserving the GPU and initializing cublas
    CUdevice device;
    cublasHandle_t cublas;
    CUDA_SAFE_CALL(cuDeviceGet(&device, device_index), device_index);
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);
    CUDA_SAFE_CALL(cuCtxSetCurrent(context), device_index);
    CUDA_SAFE_CALL(cublasCreate(&cublas), device_index);
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device_index), device_index);

    pthread_use_double = get_precision(properties);

    pthread_mutex_lock(&wait_for_init_mutex);
    if(pthread_use_double) {
        if (!filled || filled == 2) {
            A = fillup(pthread_use_double, max_msize);
            B = fillup(pthread_use_double, max_msize);
            filled = 1;
        }
    }
    else {
        if(!filled || filled == 1) {
            A = fillup(pthread_use_double, max_msize);
            B = fillup(pthread_use_double, max_msize);
            filled = 2;
        }
    }
    pthread_mutex_unlock(&wait_for_init_mutex);

    //getting information about the GPU memory
    size_t memory_avail, memory_total;
    CUDA_SAFE_CALL(cuMemGetInfo(&memory_avail,&memory_total), device_index);

    //defining memory pointers
    CUdeviceptr a_data_ptr;
    CUdeviceptr b_data_ptr;
    CUdeviceptr c_data_ptr;

    //we check for double precision support on the GPU and print errormsg, when the user wants to compute DP on a SP-only-Card.
    if(pthread_use_double && properties.major<=1 && properties.minor<=2) {
        fprintf(stderr,"    - GPU %d: %s Doesn't support double precision.\n",device_index,properties.name);
        fprintf(stderr,"    - GPU %d: %s Compute Capability: %d.%d. Requiered for double precision: >=1.3\n",device_index,properties.name,properties.major,properties.minor);
        fprintf(stderr,"    - GPU %d: %s Stressing with single precision instead. Maybe use -f parameter.\n",device_index,properties.name);
        pthread_use_double=0;
    }

    //check if the user has not set a matrix OR has set a too big matrixsite and if this is true: set a good matrixsize
    if( !size_use || ( ( size_use * size_use * pthread_use_double?sizeof(double):sizeof(float) * 3 > memory_avail ) ) ) {
        size_use=round_up((int)(0.8*sqrt(((memory_avail)/((pthread_use_double?sizeof(double):sizeof(float))*3)))),1024); //a multiple of 1024 works always well
    }
    if( pthread_use_double ) {
        use_bytes = (size_t)((double)memory_avail);
        memory_size = sizeof(double) * size_use * size_use;
    }
    else {
        use_bytes = (size_t)((float)memory_avail);
        memory_size = sizeof(float) * size_use * size_use;
    }
    iterations = (use_bytes - 2*memory_size) / memory_size; // = 1;

    //allocating memory on the GPU
    CUDA_SAFE_CALL(cuMemAlloc(&a_data_ptr, memory_size), device_index);
    CUDA_SAFE_CALL(cuMemAlloc(&b_data_ptr, memory_size), device_index);
    CUDA_SAFE_CALL(cuMemAlloc(&c_data_ptr, iterations*memory_size), device_index);

    //moving matrices A and B to the GPU
    CUDA_SAFE_CALL(cuMemcpyHtoD(a_data_ptr, A, memory_size), device_index);
    CUDA_SAFE_CALL(cuMemcpyHtoD(b_data_ptr, B, memory_size), device_index);

    //initialize c_data_ptr with copies of A
    for (i = 0; i < iterations; i++ ) {
        CUDA_SAFE_CALL(cuMemcpyHtoD(c_data_ptr + i*size_use*size_use, A, memory_size), device_index);
    }

    //save gpuvar->init_count and sys.out
    pthread_mutex_lock(&wait_for_init_mutex);

    if(gpuvar->verbose) {
        printf("   GPU %d\n", device_index);
        printf("    name:           %s\n", properties.name);
        printf("    memory:         %lu/%lu MB available (using %lu MB)\n", memory_avail/1024ul/1024ul, memory_total/1024ul/1024ul, use_bytes/1024/1024);
        printf("    matrix size:    %d\n", size_use);
        printf("    used precision: %s\n", pthread_use_double ? "double" : "single");
    }

    gpuvar->init_count++;

    //check whether all GPU threads initialized their workload, if so allow other threads to continue
    if ( gpuvar->init_count == gpuvar->use_device )
    {
      pthread_cond_signal( &wait_for_init_cond );
    }
    pthread_mutex_unlock(&wait_for_init_mutex);


    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double alpha_double = 1.0;
    const double beta_double = 0.0;

    //actual stress begins here
    while(*gpuvar->loadvar != LOAD_STOP) {
        for (i = 0; i < iterations; i++) {
            if(pthread_use_double) {
                CUDA_SAFE_CALL(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         size_use, size_use, size_use, &alpha_double,
                                         (const double*)a_data_ptr, size_use,
                                         (const double*)b_data_ptr, size_use,
                                         &beta_double,
                                         (double*)c_data_ptr + i*size_use*size_use, size_use), device_index);
            }
            else {
                CUDA_SAFE_CALL(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         size_use, size_use, size_use, &alpha,
                                         (const float*)a_data_ptr, size_use,
                                         (const float*)b_data_ptr, size_use,
                                         &beta,
                                         (float*)c_data_ptr + i*size_use*size_use, size_use), device_index);
            }
            CUDA_SAFE_CALL(cudaDeviceSynchronize(), device_index);
        }
    }

    CUDA_SAFE_CALL(cuMemFree(a_data_ptr), device_index);
    CUDA_SAFE_CALL(cuMemFree(b_data_ptr), device_index);
    CUDA_SAFE_CALL(cuMemFree(c_data_ptr), device_index);
    CUDA_SAFE_CALL(cublasDestroy(cublas), device_index);
    CUDA_SAFE_CALL(cuCtxDestroy(context), device_index);

    return NULL;
}

static int get_msize(int device_index) {
    CUcontext context;
    CUdevice device;
    int use_double;
    size_t memory_avail, memory_total;
    struct cudaDeviceProp properties;

    CUDA_SAFE_CALL(cuDeviceGet(&device, device_index), device_index);
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, device), device_index);
    CUDA_SAFE_CALL(cuCtxSetCurrent(context), device_index);
    CUDA_SAFE_CALL(cuMemGetInfo(&memory_avail,&memory_total), device_index);
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device_index), device_index);

    use_double = get_precision(properties);

    CUDA_SAFE_CALL(cuCtxDestroy(context), device_index);

    return round_up((int)(0.8*sqrt(((memory_avail)/((use_double?sizeof(double):sizeof(float))*3)))),1024); //a multiple of 1024 works always well
}

void* init_gpu(void * gpu) {
    gpuvar = (gpustruct_t*)gpu;
    
    if(gpuvar->use_device) {
        CUDA_SAFE_CALL(cuInit(0), -1);
        int devCount;
        CUDA_SAFE_CALL(cuDeviceGetCount(&devCount), -1);

        if (devCount) {
            int *dev = malloc(sizeof(int)*devCount);;
            pthread_t gputhreads[devCount]; //creating as many threads as GPUs in the system.

            if(gpuvar->verbose) {
                printf("\n  graphics processor characteristics:\n");
            }


            if( gpuvar->use_device == -1 ) { //use all GPUs if the user gave no information about use_device 
                gpuvar->use_device = devCount;
            }

            if ( gpuvar->use_device > devCount ) {
                printf("    - You requested more CUDA devices than available. Maybe you set CUDA_VISIBLE_DEVICES?\n");
                printf("    - FIRESTARTER will use %d of the requested %d CUDA device(s)\n",devCount,gpuvar->use_device);
                gpuvar->use_device = devCount;
            }

            for(int i=0; i<gpuvar->use_device; ++i) {
                int tmp = get_msize(i);

                if(tmp > max_msize) {
                    max_msize = tmp;
                }
            }

            gpuvar->init_count = 0;
            pthread_mutex_lock(&wait_for_init_mutex);


            for(int i=0; i<gpuvar->use_device; ++i) {
                dev[i] = i; //creating separate ints, so no race-condition happens when pthread_create submits the address
                pthread_create(&gputhreads[i],NULL,create_load,(void *)&(dev[i]));
            }

            pthread_cond_wait(&wait_for_init_cond, &wait_for_init_mutex);
            gpuvar->loadingdone = 1;
            pthread_mutex_unlock(&wait_for_init_mutex);

            /* join computation threads */
            for(int i=0; i<gpuvar->use_device; ++i) {
                pthread_join(gputhreads[i],NULL);
            }

            free(dev);
        }
        else {
            if(gpuvar->verbose) {
                printf("    - No CUDA devices. Just stressing CPU(s). Maybe use FIRESTARTER instead of FIRESTARTER_CUDA?\n");
            }
            gpuvar->loadingdone = 1;
        }
    }
    else {
        if(gpuvar->verbose) {
            printf("    --gpus 0 is set. Just stressing CPU(s). Maybe use FIRESTARTER instead of FIRESTARTER_CUDA?\n");
        }
        gpuvar->loadingdone = 1;
    }

    free(A);
    free(B);

    return NULL;
}
