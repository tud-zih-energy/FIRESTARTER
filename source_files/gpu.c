/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2017 TU Dresden, Center for Information Services and High
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

#define CUDA_ERROR_CHECK
#define CUDA_SAFE_CALL( cuerr ) cuda_safe_call( cuerr, __FILE__, __LINE__ )

static void *A,*B;
static unsigned int size,useDouble,useDevice,verbose; //matrixsize,switch for double precision and numbers of GPUs to use.
static gpustruct * gpuvar=NULL;

//CUDA Error Checking
static inline void cuda_safe_call( cudaError_t cuerr, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cuerr != cudaSuccess)
    {
        fprintf( stderr, "    - CUDA_SAFE_CALL() failed at %s:%i : %i %s\n",
                 file, line, cuerr,cudaGetErrorString( cuerr ) );
        exit(cuerr);
    }
#endif

    return;
}

int roundUp(int numToRound, int multiple)  
{  
 if(multiple == 0)  
 {  
  return numToRound;  
 }  

 int remainder = numToRound % multiple; 
 if (remainder == 0)
  {
    return numToRound; 
  }

 return numToRound + multiple - remainder; 
} 

int ipow(int base,int exp) {
    int result = 1;
    while (exp) {
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    }
    return result;
}

static int bbs(void) {
    static int s;
    static int runned = 0;
    int n = 131*17;
    if(runned == 0) {
        runned = 1;
        s = 123; // this is the seed
    }
    s = ipow(s,2) % n;
    return s;
}

void* fillup(void* array, int useD, int size) {
    int i;
    if(useD) {
        double frac;
        double *dbl=(double*)array;
        dbl = malloc(sizeof(double)*size*size);
        for (i=0; i<size*size; i++) {
            if(i % 128 == 0) {
                frac = (double) bbs() / 10000;
                dbl[i]=(double) bbs() + frac;
            }
        }
    } else {
        float frac;
        float *flt = (float*)array;
        flt = malloc(sizeof(float)*size*size);
        for (i=0; i<size*size; i++) {
            if(i % 128 == 0) {
                frac = (float) bbs() / 10000;
                flt[i]=(float) bbs() + frac;
            }
        }
    }
    return array;
}
void* startBurn(void *index) {
    int d_devIndex = *((int*)index);   //GPU Index. Used to pin this pthread to the GPU.
    int d_iters,i;
    int pthread_useDouble = useDouble; //local per-thread variable, if there's a GPU in the system without Double Precision support.
    int size_use=0;
    if (globsize>0){
        size_use=globsize;
    }
    void *A,*B;
    CUcontext d_ctx;
    size_t useBytes, d_resultSize;
    struct cudaDeviceProp properties;

    //Reserving the GPU and Initializing cublas
    CUdevice d_dev;
    cublasHandle_t d_cublas;
    CUDA_SAFE_CALL(cuDeviceGet(&d_dev, d_devIndex));
    CUDA_SAFE_CALL(cuCtxCreate(&d_ctx, 0, d_dev));
    CUDA_SAFE_CALL(cuCtxSetCurrent(d_ctx));
    CUDA_SAFE_CALL(cublasCreate(&d_cublas));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties,d_devIndex));
    //getting Informations about the GPU Memory
    size_t availMemory, totalMemory;
    CUDA_SAFE_CALL(cuMemGetInfo(&availMemory,&totalMemory));

    //Defining Memory Pointers
    CUdeviceptr d_Adata;
    CUdeviceptr d_Bdata;
    CUdeviceptr d_Cdata;

    //we check for double precision support on the GPU and print errormsg, when the user wants to compute DP on a SP-only-Card.
    if(pthread_useDouble && properties.major<=1 && properties.minor<=2){
    fprintf(stderr,"    - GPU %d: %s Doesn't support double precision.\n",d_devIndex,properties.name);
    fprintf(stderr,"    - GPU %d: %s Compute Capability: %d.%d. Requiered for double precision: >=1.3\n",d_devIndex,properties.name,properties.major,properties.minor);
    fprintf(stderr,"    - GPU %d: %s Stressing with single precision instead. Maybe use -f parameter.\n",d_devIndex,properties.name);
    pthread_useDouble=0;
    }
    //check if the user has not set a matrix OR has set a too big matrixsite and if this is true: set a good matrixsize
    if(!size_use || (((size_use*size_use)*(12*(pthread_useDouble+1)))>availMemory)){
        size_use=roundUp((int)(0.8*sqrt(((availMemory)/(12*(pthread_useDouble+1))))),1024); //a multiple of 1024 works always well
    }
    if(pthread_useDouble) {
        useBytes = (size_t)((double)availMemory);
        d_resultSize = sizeof(double)*size_use*size_use;
    } else {
        useBytes = (size_t)((float)availMemory);
        d_resultSize = sizeof(float)*size_use*size_use;
    }
    d_iters = (useBytes - 2*d_resultSize)/d_resultSize;
    //Allocating memory on the GPU
    A=fillup(A,pthread_useDouble,size_use);
    B=fillup(B,pthread_useDouble,size_use);
    CUDA_SAFE_CALL(cuMemAlloc(&d_Adata, d_resultSize));
    CUDA_SAFE_CALL(cuMemAlloc(&d_Bdata, d_resultSize));
    CUDA_SAFE_CALL(cuMemAlloc(&d_Cdata, d_iters*d_resultSize));
    // Moving matrices A and B to the GPU
    CUDA_SAFE_CALL(cuMemcpyHtoD_v2(d_Adata, A, d_resultSize));
    CUDA_SAFE_CALL(cuMemcpyHtoD_v2(d_Bdata, B, d_resultSize));
    static const float alpha = 1.0f;
    static const float beta = 0.0f;
    static const double alphaD = 1.0;
    static const double betaD = 0.0;
    if(verbose) printf("    - GPU %d: %s Initialized with %lu MB of memory (%lu MB available, using %lu MB of it) and Matrix Size: %d.\n",d_devIndex,properties.name,totalMemory/1024ul/1024ul, availMemory/1024ul/1024ul, useBytes/1024/1024,size_use);
    //Actual stress begins here, we set the loadingdone variable on true so that the CPU workerthreads can start too. But only gputhread #0 is setting the variable, to prohibite race-conditioning...
    if(d_devIndex==0){
      gpuvar->loadingdone=1;
    }
    for(;;) {
        for (i = 0; i < d_iters; i++) {
            if(pthread_useDouble) {
                CUDA_SAFE_CALL(cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         size_use, size_use, size_use, &alphaD,
                                         (const double*)d_Adata, size_use,
                                         (const double*)d_Bdata, size_use,
                                         &betaD,
                                         (double*)d_Cdata + i*size_use*size_use, size_use));
            } else {
                CUDA_SAFE_CALL(cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                         size_use, size_use, size_use, &alpha,
                                         (const float*)d_Adata, size_use,
                                         (const float*)d_Bdata, size_use,
                                         &beta,
                                         (float*)d_Cdata + i*size_use*size_use, size_use));
            }
        }
    }
    CUDA_SAFE_CALL(cuMemFree_v2(d_Adata));
    CUDA_SAFE_CALL(cuMemFree_v2(d_Bdata));
    CUDA_SAFE_CALL(cuMemFree_v2(d_Cdata));
    return NULL;
}

#ifdef CUDA_ONLY
int main(int argc, char ** argv){
    gpustruct * gpu = malloc(sizeof(gpustruct));
    gpu->useDevice=-1;
    gpu->useDouble=1;
    gpu->msize=0;
    gpu->verbose=1;
    gpu->loadingdone=1;
#endif
#ifdef CUDA
void* initgpu(void *gpu) {
#endif
    gpuvar=(gpustruct*)gpu;
    useDevice = gpuvar->useDevice;  //how many GPUs to use
    verbose   = gpuvar->verbose;    //Verbosity
    if(gpuvar->msize>0)
    {
        globsize=gpuvar->msize;
    }
    if(useDevice) {
        CUDA_SAFE_CALL(cuInit(0));
        int devCount;
        CUDA_SAFE_CALL(cuDeviceGetCount(&devCount));
        if (devCount) {
            int *dev=malloc(sizeof(int)*devCount);;
            pthread_t gputhreads[devCount]; //creating as many threads as GPUs in the System.
            if(verbose) printf("\n  graphics processor characteristics:\n");
            int i;
            if( useDevice==-1 ) { //use all GPUs if the user gave no information about useDevice 
                useDevice=devCount;
            }
            if ( useDevice > devCount ) {
                printf("    - You requested more CUDA devices than available. Maybe you set CUDA_VISIBLE_DEVICES?\n");
                printf("    - FIRESTARTER will use %d of the requested %d CUDA device(s)\n",devCount,useDevice);
                useDevice=devCount;
            }

            for(i=0; i<useDevice; ++i) {
                dev[i]=i; //creating seperate ints, so no race-condition happens when pthread_create submits the adress
                pthread_create(&gputhreads[i],NULL,startBurn,(void *)&(dev[i]));
            }

            /* join computation threads */
            for(i=0; i<useDevice; ++i) {
                pthread_join(gputhreads[i],NULL);
            }

            free(dev);
        }
        else {
            if(verbose) printf("    - No CUDA devices. Just stressing CPU(s). Maybe use FIRESTARTER instead of FIRESTARTER_CUDA?\n");
            gpuvar->loadingdone=1;
        }
    }
    else {
        if(verbose) printf("    --gpus 0 is set. Just stressing CPU(s). Maybe use FIRESTARTER instead of FIRESTARTER_CUDA?\n");
        gpuvar->loadingdone=1;
    }
#ifdef CUDA
    return NULL;
#endif
#ifdef CUDA_ONLY
    return 0;
#endif
}
