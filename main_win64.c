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

#include "work.h"
#include "help.h"
#include "cpu.h"
#include "firestarter_global.h"
#include <windows.h>
#include <inttypes.h>
#include <getopt.h>

unsigned long long LOAD_VAR=LOAD_HIGH;  /* shared variable that specifies load level */
unsigned int features;                  /* bitmask for CPU features */
unsigned long long clockrate;           /* measured clockrate (via TSC) */

/*
 * load characteristics as defind by -p and -l
 */
long PERIOD = 100000, LOAD = 100;
long load_time, idle_time;

/* thread handling */
DWORD threadDescriptor;
HANDLE * threads;

/* Ctrl-C handler */
BOOL WINAPI ConsoleHandler(DWORD type);

/* global to be accessible by Ctrl-C handler */
threaddata_t * threaddata;
int nr_threads=-1,verbose=0;
struct timeval ts;
unsigned long long start_time,end_time;

/*
 * low load function
 * borrowed from work.c; TODO remove redundant code
 */
int low_load_function(volatile unsigned long long addrHigh, unsigned int period) __attribute__((noinline));
int low_load_function(volatile unsigned long long addrHigh, unsigned int period)
{
    int nap;

    nap = period / 100;
    __asm__ __volatile__ ("mfence;"
                  "cpuid;" ::: "eax", "ebx", "ecx", "edx");
    while(*((volatile unsigned long long *)addrHigh) == LOAD_LOW){
        __asm__ __volatile__ ("mfence;"
                      "cpuid;" ::: "eax", "ebx", "ecx", "edx");
        usleep(nap);
        __asm__ __volatile__ ("mfence;"
                      "cpuid;" ::: "eax", "ebx", "ecx", "edx");
    }

    return 0;
}

static int has_feature(int feature)
{
    if (feature&features) return 1;
    else return 0;
}

static void list_functions(){

    printf("\n available load-functions:\n");
    printf("  ID   | NAME                           | available on this system\n");
    printf("  ----------------------------------------------------------------\n");
    if (has_feature(AVX512)) printf("  %4.4s | %.30s | yes\n","1","FUNC_KNL_XEONPHI_AVX512_4T                             ");
    else printf("  %4.4s | %.30s | no \n","1","FUNC_KNL_XEONPHI_AVX512_4T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","2","FUNC_SKL_COREI_FMA_1T                             ");
    else printf("  %4.4s | %.30s | no \n","2","FUNC_SKL_COREI_FMA_1T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","3","FUNC_SKL_COREI_FMA_2T                             ");
    else printf("  %4.4s | %.30s | no \n","3","FUNC_SKL_COREI_FMA_2T                             ");
    if (has_feature(AVX512)) printf("  %4.4s | %.30s | yes\n","4","FUNC_SKL_XEONEP_AVX512_1T                             ");
    else printf("  %4.4s | %.30s | no \n","4","FUNC_SKL_XEONEP_AVX512_1T                             ");
    if (has_feature(AVX512)) printf("  %4.4s | %.30s | yes\n","5","FUNC_SKL_XEONEP_AVX512_2T                             ");
    else printf("  %4.4s | %.30s | no \n","5","FUNC_SKL_XEONEP_AVX512_2T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","6","FUNC_HSW_COREI_FMA_1T                             ");
    else printf("  %4.4s | %.30s | no \n","6","FUNC_HSW_COREI_FMA_1T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","7","FUNC_HSW_COREI_FMA_2T                             ");
    else printf("  %4.4s | %.30s | no \n","7","FUNC_HSW_COREI_FMA_2T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","8","FUNC_HSW_XEONEP_FMA_1T                             ");
    else printf("  %4.4s | %.30s | no \n","8","FUNC_HSW_XEONEP_FMA_1T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","9","FUNC_HSW_XEONEP_FMA_2T                             ");
    else printf("  %4.4s | %.30s | no \n","9","FUNC_HSW_XEONEP_FMA_2T                             ");
    if (has_feature(AVX)) printf("  %4.4s | %.30s | yes\n","10","FUNC_SNB_COREI_AVX_1T                             ");
    else printf("  %4.4s | %.30s | no \n","10","FUNC_SNB_COREI_AVX_1T                             ");
    if (has_feature(AVX)) printf("  %4.4s | %.30s | yes\n","11","FUNC_SNB_COREI_AVX_2T                             ");
    else printf("  %4.4s | %.30s | no \n","11","FUNC_SNB_COREI_AVX_2T                             ");
    if (has_feature(AVX)) printf("  %4.4s | %.30s | yes\n","12","FUNC_SNB_XEONEP_AVX_1T                             ");
    else printf("  %4.4s | %.30s | no \n","12","FUNC_SNB_XEONEP_AVX_1T                             ");
    if (has_feature(AVX)) printf("  %4.4s | %.30s | yes\n","13","FUNC_SNB_XEONEP_AVX_2T                             ");
    else printf("  %4.4s | %.30s | no \n","13","FUNC_SNB_XEONEP_AVX_2T                             ");
    if (has_feature(SSE2)) printf("  %4.4s | %.30s | yes\n","14","FUNC_NHM_COREI_SSE2_1T                             ");
    else printf("  %4.4s | %.30s | no \n","14","FUNC_NHM_COREI_SSE2_1T                             ");
    if (has_feature(SSE2)) printf("  %4.4s | %.30s | yes\n","15","FUNC_NHM_COREI_SSE2_2T                             ");
    else printf("  %4.4s | %.30s | no \n","15","FUNC_NHM_COREI_SSE2_2T                             ");
    if (has_feature(SSE2)) printf("  %4.4s | %.30s | yes\n","16","FUNC_NHM_XEONEP_SSE2_1T                             ");
    else printf("  %4.4s | %.30s | no \n","16","FUNC_NHM_XEONEP_SSE2_1T                             ");
    if (has_feature(SSE2)) printf("  %4.4s | %.30s | yes\n","17","FUNC_NHM_XEONEP_SSE2_2T                             ");
    else printf("  %4.4s | %.30s | no \n","17","FUNC_NHM_XEONEP_SSE2_2T                             ");
    if (has_feature(FMA4)) printf("  %4.4s | %.30s | yes\n","18","FUNC_BLD_OPTERON_FMA4_1T                             ");
    else printf("  %4.4s | %.30s | no \n","18","FUNC_BLD_OPTERON_FMA4_1T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","19","FUNC_ZEN_EPYC_ZEN_FMA_1T                             ");
    else printf("  %4.4s | %.30s | no \n","19","FUNC_ZEN_EPYC_ZEN_FMA_1T                             ");
    if (has_feature(FMA)) printf("  %4.4s | %.30s | yes\n","20","FUNC_ZEN_EPYC_ZEN_FMA_2T                             ");
    else printf("  %4.4s | %.30s | no \n","20","FUNC_ZEN_EPYC_ZEN_FMA_2T                             ");

    return;
}

static int get_function(unsigned int id){
    int func=FUNC_UNKNOWN;

    switch(id){
       case 1:
         func = FUNC_KNL_XEONPHI_AVX512_4T;
         break;
       case 2:
         func = FUNC_SKL_COREI_FMA_1T;
         break;
       case 3:
         func = FUNC_SKL_COREI_FMA_2T;
         break;
       case 4:
         func = FUNC_SKL_XEONEP_AVX512_1T;
         break;
       case 5:
         func = FUNC_SKL_XEONEP_AVX512_2T;
         break;
       case 6:
         func = FUNC_HSW_COREI_FMA_1T;
         break;
       case 7:
         func = FUNC_HSW_COREI_FMA_2T;
         break;
       case 8:
         func = FUNC_HSW_XEONEP_FMA_1T;
         break;
       case 9:
         func = FUNC_HSW_XEONEP_FMA_2T;
         break;
       case 10:
         func = FUNC_SNB_COREI_AVX_1T;
         break;
       case 11:
         func = FUNC_SNB_COREI_AVX_2T;
         break;
       case 12:
         func = FUNC_SNB_XEONEP_AVX_1T;
         break;
       case 13:
         func = FUNC_SNB_XEONEP_AVX_2T;
         break;
       case 14:
         func = FUNC_NHM_COREI_SSE2_1T;
         break;
       case 15:
         func = FUNC_NHM_COREI_SSE2_2T;
         break;
       case 16:
         func = FUNC_NHM_XEONEP_SSE2_1T;
         break;
       case 17:
         func = FUNC_NHM_XEONEP_SSE2_2T;
         break;
       case 18:
         func = FUNC_BLD_OPTERON_FMA4_1T;
         break;
       case 19:
         func = FUNC_ZEN_EPYC_ZEN_FMA_1T;
         break;
       case 20:
         func = FUNC_ZEN_EPYC_ZEN_FMA_2T;
         break;
       default:
         fprintf(stderr, "\nError: unknown function id: %i, see --avail for available ids\n\n", id);
    }

    return func;
}

static DWORD WINAPI WorkerThread(void* threadParams)
{
  threaddata_t * data = (threaddata_t*) threadParams; 
  void * p;

  data->iterations=0;
  data->start_tsc=timestamp();

  switch (data->FUNCTION) {
    case FUNC_KNL_XEONPHI_AVX512_4T:
      p =  _mm_malloc(65762661,4096);
      data->addrMem = (unsigned long long) p;
      init_knl_xeonphi_avx512_4t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_knl_xeonphi_avx512_4t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SKL_COREI_FMA_1T:
      p =  _mm_malloc(106725392,4096);
      data->addrMem = (unsigned long long) p;
      init_skl_corei_fma_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_skl_corei_fma_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SKL_COREI_FMA_2T:
      p =  _mm_malloc(53362704,4096);
      data->addrMem = (unsigned long long) p;
      init_skl_corei_fma_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_skl_corei_fma_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SKL_XEONEP_AVX512_1T:
      p =  _mm_malloc(1051099152,4096);
      data->addrMem = (unsigned long long) p;
      init_skl_xeonep_avx512_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_skl_xeonep_avx512_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SKL_XEONEP_AVX512_2T:
      p =  _mm_malloc(525549584,4096);
      data->addrMem = (unsigned long long) p;
      init_skl_xeonep_avx512_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_skl_xeonep_avx512_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_HSW_COREI_FMA_1T:
      p =  _mm_malloc(106725392,4096);
      data->addrMem = (unsigned long long) p;
      init_hsw_corei_fma_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_hsw_corei_fma_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_HSW_COREI_FMA_2T:
      p =  _mm_malloc(53362704,4096);
      data->addrMem = (unsigned long long) p;
      init_hsw_corei_fma_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_hsw_corei_fma_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_HSW_XEONEP_FMA_1T:
      p =  _mm_malloc(107773968,4096);
      data->addrMem = (unsigned long long) p;
      init_hsw_xeonep_fma_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_hsw_xeonep_fma_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_HSW_XEONEP_FMA_2T:
      p =  _mm_malloc(53886992,4096);
      data->addrMem = (unsigned long long) p;
      init_hsw_xeonep_fma_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_hsw_xeonep_fma_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SNB_COREI_AVX_1T:
      p =  _mm_malloc(106725392,4096);
      data->addrMem = (unsigned long long) p;
      init_snb_corei_avx_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_snb_corei_avx_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SNB_COREI_AVX_2T:
      p =  _mm_malloc(53362704,4096);
      data->addrMem = (unsigned long long) p;
      init_snb_corei_avx_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_snb_corei_avx_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SNB_XEONEP_AVX_1T:
      p =  _mm_malloc(107773968,4096);
      data->addrMem = (unsigned long long) p;
      init_snb_xeonep_avx_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_snb_xeonep_avx_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_SNB_XEONEP_AVX_2T:
      p =  _mm_malloc(53886992,4096);
      data->addrMem = (unsigned long long) p;
      init_snb_xeonep_avx_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_snb_xeonep_avx_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_NHM_COREI_SSE2_1T:
      p =  _mm_malloc(106725392,4096);
      data->addrMem = (unsigned long long) p;
      init_nhm_corei_sse2_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_nhm_corei_sse2_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_NHM_COREI_SSE2_2T:
      p =  _mm_malloc(53362704,4096);
      data->addrMem = (unsigned long long) p;
      init_nhm_corei_sse2_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_nhm_corei_sse2_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_NHM_XEONEP_SSE2_1T:
      p =  _mm_malloc(107249680,4096);
      data->addrMem = (unsigned long long) p;
      init_nhm_xeonep_sse2_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_nhm_xeonep_sse2_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_NHM_XEONEP_SSE2_2T:
      p =  _mm_malloc(53624848,4096);
      data->addrMem = (unsigned long long) p;
      init_nhm_xeonep_sse2_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_nhm_xeonep_sse2_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_BLD_OPTERON_FMA4_1T:
      p =  _mm_malloc(106709008,4096);
      data->addrMem = (unsigned long long) p;
      init_bld_opteron_fma4_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_bld_opteron_fma4_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_ZEN_EPYC_ZEN_FMA_1T:
      p =  _mm_malloc(107544592,4096);
      data->addrMem = (unsigned long long) p;
      init_zen_epyc_zen_fma_1t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_zen_epyc_zen_fma_1t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    case FUNC_ZEN_EPYC_ZEN_FMA_2T:
      p =  _mm_malloc(53772304,4096);
      data->addrMem = (unsigned long long) p;
      init_zen_epyc_zen_fma_2t(data);
      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){
        asm_work_zen_epyc_zen_fma_2t(data);
        low_load_function(data->addrHigh, PERIOD);
      }
      break;
    default:
      return EXIT_FAILURE;       
  }

  data->stop_tsc=timestamp();

  return 0;
}

int print_report(){
  unsigned long long start_tsc,stop_tsc,iterations=0;
  double runtime;
  int i;

  if (verbose){
    printf("\nperformance report:\n\n");

    start_tsc=threaddata[0].start_tsc;
    stop_tsc=threaddata[0].stop_tsc;
    for(i = 0; i < nr_threads; i++){
      printf("Thread %i: %I64u iterations, tsc_delta: %I64u\n",i,threaddata[i].iterations, threaddata[i].stop_tsc - threaddata[i].start_tsc );
      iterations+=threaddata[i].iterations;
      if (start_tsc > threaddata[i].start_tsc) start_tsc = threaddata[i].start_tsc;
      if (stop_tsc < threaddata[i].stop_tsc) stop_tsc = threaddata[i].stop_tsc;
    }
    printf("\ntotal iterations: %I64u\n",iterations);
    runtime=(double)(end_time - start_time) / 1000000;
    printf("runtime: %.2f seconds (%I64u cycles)\n\n",runtime, stop_tsc - start_tsc);

    printf("estimated floating point performance: %.2f GFLOPS\n", (double)threaddata[0].flops*0.000000001*(double)iterations/runtime);
    printf("estimated memory bandwidth*: %.2f GB/s\n", (double)threaddata[0].bytes*0.000000001*(double)iterations/runtime);
    printf("\n* this estimate is highly unreliable if --function is used in order to select\n");
    printf("  a function that is not optimized for your architecture, or if FIRESTARTER is\n");
    printf("  executed on an unsupported architecture!\n");

    printf("\n");
  }

  return 0;
}

int main(int argc, char *argv[]){
  SYSTEM_INFO SysInfo;
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION * ProcInfo;
  char * buffer, * proc_name, * vendor;
  unsigned long nr_cpu=0, nr_core=0, threads_per_core=0, nr_pkg=0, cores_per_package=0,nr_numa_node=0;
  unsigned long ProcInfoSize,ProcInfoCount;
  int family, model, stepping;
  int time=0,i,c;
  int func = FUNC_NOT_DEFINED;


  static struct option long_options[] = {
    {"copyright",   no_argument,        0, 'c'},
    {"help",        no_argument,        0, 'h'},
    {"version",     no_argument,        0, 'v'},
    {"warranty",    no_argument,        0, 'w'},
    {"report",      no_argument,        0, 'r'},
    {"avail",       no_argument,        0, 'a'},
    {"function",    required_argument,  0, 'i'},
    {"threads",     required_argument,  0, 'n'},
    {"timeout",     required_argument,  0, 't'},
    {"load",        required_argument,  0, 'l'},
    {"period",      required_argument,  0, 'p'},
    {0,             0,                  0,  0 }
  };

  /* gather information about assignment of CPUs to core and packages */
  GetSystemInfo(&SysInfo);
  nr_cpu = SysInfo.dwNumberOfProcessors;
  GetLogicalProcessorInformation(NULL,&ProcInfoSize);
  ProcInfo = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *) malloc(ProcInfoSize);
  if (GetLogicalProcessorInformation(ProcInfo,&ProcInfoSize)){
    ProcInfoCount = ProcInfoSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (i=0;i<ProcInfoCount;i++){
      switch (ProcInfo[i].Relationship){
        case RelationNumaNode:
          nr_numa_node++;
          break;
        case RelationProcessorCore:
          nr_core++;
          break;
        case RelationProcessorPackage:
          nr_pkg++;
          break;
        default:
          break;
      }
    }
    if ((nr_core > 0) && (nr_pkg > 0)) cores_per_package = nr_core/nr_pkg;
    if ((nr_cpu > 0) && (nr_core >0)) threads_per_core = nr_cpu/nr_core;
  }

  if (!SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler,TRUE)) {
     fprintf(stderr, "Error: Unable to install Ctrl-C handler!\n");
     return EXIT_FAILURE;
  }

  //cpuid based feature detection
  buffer = (char*) malloc(_HW_DETECT_MAX_OUTPUT);
  get_cpu_isa_extensions(buffer,_HW_DETECT_MAX_OUTPUT);
  if (strstr(buffer,"MMX")!=NULL)       features |= MMX;
  if (strstr(buffer,"MMX_EXT")!=NULL)   features |= MMX_EXT;
  if (strstr(buffer,"SSE")!=NULL)       features |= SSE;
  if (strstr(buffer,"SSE2")!=NULL)      features |= SSE2;
  if (strstr(buffer,"SSE3")!=NULL)      features |= SSE3;
  if (strstr(buffer,"SSSE3")!=NULL)     features |= SSSE3;
  if (strstr(buffer,"SSE4.1")!=NULL)    features |= SSE4_1;
  if (strstr(buffer,"SSE4.2")!=NULL)    features |= SSE4_2;
  if (strstr(buffer,"SSE4A")!=NULL)     features |= SSE4A;
  if (strstr(buffer,"AVX")!=NULL)       features |= AVX;
  if (strstr(buffer,"AVX2")!=NULL)      features |= AVX2;
  if (strstr(buffer,"FMA")!=NULL)       features |= FMA;
  if (strstr(buffer,"FMA4")!=NULL)      features |= FMA4;
  if (strstr(buffer,"AVX512"))          features |= AVX512;

  vendor = (char*) malloc(_HW_DETECT_MAX_OUTPUT);
  proc_name = (char*) malloc(_HW_DETECT_MAX_OUTPUT);
  get_cpu_vendor(vendor,_HW_DETECT_MAX_OUTPUT);
  get_cpu_name(proc_name,_HW_DETECT_MAX_OUTPUT);
  family=get_cpu_family();
  model=get_cpu_model();
  stepping=get_cpu_stepping();

  while(1){
    c = getopt_long(argc, argv, "chvwrai:n:t:l:p:", long_options, NULL);
    if(c == -1) break;

    errno = 0;
    switch(c)
    {
      case 'c':
        show_copyright();
        return EXIT_SUCCESS;
      case 'h':
        show_help_win64();
        return EXIT_SUCCESS;
      case 'v':
        show_version();
        return EXIT_SUCCESS;
      case 'w':
        show_warranty();
        return EXIT_SUCCESS;
      case 'a':
        show_version();
        list_functions();
        return EXIT_SUCCESS;
      case 'i':
        func=get_function((unsigned int)strtol(optarg,NULL,10));
        if (func==FUNC_UNKNOWN) return EXIT_FAILURE;
        break;
      case 'r':
        verbose = 1;
        break;
      case 'n':
        nr_threads=(unsigned int)strtol(optarg,NULL,10);
        break;
      case 't':
        time=(unsigned int)strtol(optarg,NULL,10);
        break;
      case 'l':
        LOAD = strtol(optarg,NULL,10);
        if ((errno != 0) || (LOAD < 0) || (LOAD > 100)) {
           printf("Error: load out of range or not a number: %s\n",optarg);
           return EXIT_FAILURE;
        }
        break;
      case 'p':
        PERIOD = strtol(optarg,NULL,10);
        if ((errno != 0) || (PERIOD <= 0)) {
           printf("Error: period out of range or not a number: %s\n",optarg);
           return EXIT_FAILURE;
        }
        break;
      case ':':   // Missing argument
        return EXIT_FAILURE;
      case '?':   // Unknown option
        return EXIT_FAILURE;
    }
  }
  if(optind < argc)
  {
    printf("Error: too many parameters!\n");
    return EXIT_FAILURE;
  }

  LOAD = ( PERIOD * LOAD ) / 100;
  if ((LOAD == PERIOD) || (LOAD == 0)) PERIOD = 0;    // disable interupts for 100% and 0% load case
  if (LOAD == 0) LOAD_VAR = LOAD_LOW;                 // use low load routine
  
  /* using millisecond granularity as long usleeps (> 1000000 usecs) do not seem to work propperly, short usleeps (<50000 usecs) do not work anyway => no short periods under windows */
  load_time = LOAD/1000;
  idle_time = (PERIOD - LOAD)/1000;

  printf("FIRESTARTER - A Processor Stress Test Utility\n");
  printf("Copyright (C) %i TU Dresden, Center for Information Services and High Performance Computing\n\n",COPYRIGHT_YEAR);
  printf("This program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\n");
  printf("You should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n");

  /* select number of threads */
  if ((nr_threads == -1)&&(nr_cpu>0)) nr_threads = nr_cpu; // use all logical processors if not specified otherwise
  else if (nr_threads == -1){
    printf("\nNumber of threads not specified on command line (-n) and autodetection of available CPUs failed. Please specify the number of threads:\n");
    scanf("%d",&nr_threads);
  }

  /* select code path according to family/model */
  if (func == FUNC_NOT_DEFINED){
     if ((strcmp("GenuineIntel", vendor) == 0) || (strcmp("AuthenticAMD", vendor) == 0))
     {
        switch (family) {
          case 6:
            switch (model) {
              case 87:
                if (has_feature(AVX512)) {
                    if (threads_per_core == 4) func = FUNC_KNL_XEONPHI_AVX512_4T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: AVX512 is requiered for architecture \"KNL\", but is not supported!\n");
                }
                break;
              case 78:
              case 94:
                if (has_feature(FMA)) {
                    if (threads_per_core == 1) func = FUNC_SKL_COREI_FMA_1T;
                    if (threads_per_core == 2) func = FUNC_SKL_COREI_FMA_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: FMA is requiered for architecture \"SKL\", but is not supported!\n");
                }
                break;
              case 85:
                if (has_feature(AVX512)) {
                    if (threads_per_core == 1) func = FUNC_SKL_XEONEP_AVX512_1T;
                    if (threads_per_core == 2) func = FUNC_SKL_XEONEP_AVX512_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: AVX512 is requiered for architecture \"SKL\", but is not supported!\n");
                }
                break;
              case 60:
              case 61:
              case 69:
              case 70:
              case 71:
                if (has_feature(FMA)) {
                    if (threads_per_core == 1) func = FUNC_HSW_COREI_FMA_1T;
                    if (threads_per_core == 2) func = FUNC_HSW_COREI_FMA_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: FMA is requiered for architecture \"HSW\", but is not supported!\n");
                }
                break;
              case 63:
              case 79:
                if (has_feature(FMA)) {
                    if (threads_per_core == 1) func = FUNC_HSW_XEONEP_FMA_1T;
                    if (threads_per_core == 2) func = FUNC_HSW_XEONEP_FMA_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: FMA is requiered for architecture \"HSW\", but is not supported!\n");
                }
                break;
              case 42:
              case 58:
                if (has_feature(AVX)) {
                    if (threads_per_core == 1) func = FUNC_SNB_COREI_AVX_1T;
                    if (threads_per_core == 2) func = FUNC_SNB_COREI_AVX_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: AVX is requiered for architecture \"SNB\", but is not supported!\n");
                }
                break;
              case 45:
              case 62:
                if (has_feature(AVX)) {
                    if (threads_per_core == 1) func = FUNC_SNB_XEONEP_AVX_1T;
                    if (threads_per_core == 2) func = FUNC_SNB_XEONEP_AVX_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: AVX is requiered for architecture \"SNB\", but is not supported!\n");
                }
                break;
              case 30:
              case 37:
              case 23:
                if (has_feature(SSE2)) {
                    if (threads_per_core == 1) func = FUNC_NHM_COREI_SSE2_1T;
                    if (threads_per_core == 2) func = FUNC_NHM_COREI_SSE2_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: SSE2 is requiered for architecture \"NHM\", but is not supported!\n");
                }
                break;
              case 26:
              case 44:
                if (has_feature(SSE2)) {
                    if (threads_per_core == 1) func = FUNC_NHM_XEONEP_SSE2_1T;
                    if (threads_per_core == 2) func = FUNC_NHM_XEONEP_SSE2_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: SSE2 is requiered for architecture \"NHM\", but is not supported!\n");
                }
                break;
            default:
                fprintf(stderr, "\nWarning: %s family %i, model %i is not supported by this version of FIRESTARTER!\n         Check project website for updates.\n",vendor,family,model);
            }
            break;
          case 21:
            switch (model) {
              case 1:
              case 2:
              case 3:
                if (has_feature(FMA4)) {
                    if (threads_per_core == 1) func = FUNC_BLD_OPTERON_FMA4_1T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: FMA4 is requiered for architecture \"BLD\", but is not supported!\n");
                }
                break;
            default:
                fprintf(stderr, "\nWarning: %s family %i, model %i is not supported by this version of FIRESTARTER!\n         Check project website for updates.\n",vendor,family,model);
            }
            break;
          case 23:
            switch (model) {
              case 1:
              case 8:
              case 17:
              case 24:
                if (has_feature(FMA)) {
                    if (threads_per_core == 1) func = FUNC_ZEN_EPYC_ZEN_FMA_1T;
                    if (threads_per_core == 2) func = FUNC_ZEN_EPYC_ZEN_FMA_2T;
                    if (func == FUNC_NOT_DEFINED) {
                        fprintf(stderr, "Warning: no code path for %lu threads per core!\n",threads_per_core);
                    }
                }
                if (func == FUNC_NOT_DEFINED) {
                    fprintf(stderr, "\nWarning: ZEN_FMA is requiered for architecture \"ZEN\", but is not supported!\n");
                }
                break;
            default:
                fprintf(stderr, "\nWarning: %s family %i, model %i is not supported by this version of FIRESTARTER!\n         Check project website for updates.\n",vendor,family,model);
            }
            break;
          default:
            fprintf(stderr, "\nWarning: family %i processors are not supported by this version of FIRESTARTER!\n         Check project website for updates.\n",family);
        }
     }
     else {
        fprintf(stderr, "Warning: %s processors not supported by this version of FIRESTARTER!\n         Check project website for updates.\n",vendor);
     }
  }

  /* use AVX512 as fallback if available*/
  if ((func == FUNC_NOT_DEFINED)&&(has_feature(AVX512))) {
      /* use function for correct number of threads per core if available */
      if(threads_per_core == 1) {
          func = FUNC_SKL_XEONEP_AVX512_1T;
          fprintf(stderr, "Warning: using function FUNC_SKL_XEONEP_AVX512_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      if(threads_per_core == 2) {
          func = FUNC_SKL_XEONEP_AVX512_2T;
          fprintf(stderr, "Warning: using function FUNC_SKL_XEONEP_AVX512_2T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      /* use function for 1 threads per core if no function for actual number of thread per core exists*/
      if (func == FUNC_NOT_DEFINED)
      {
          func = FUNC_SKL_XEONEP_AVX512_1T;
          fprintf(stderr, "Warning: using function FUNC_SKL_XEONEP_AVX512_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
  }
  /* use FMA4 as fallback if available*/
  if ((func == FUNC_NOT_DEFINED)&&(has_feature(FMA4))) {
      /* use function for correct number of threads per core if available */
      if(threads_per_core == 1) {
          func = FUNC_BLD_OPTERON_FMA4_1T;
          fprintf(stderr, "Warning: using function FUNC_BLD_OPTERON_FMA4_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      /* use function for 1 threads per core if no function for actual number of thread per core exists*/
      if (func == FUNC_NOT_DEFINED)
      {
          func = FUNC_BLD_OPTERON_FMA4_1T;
          fprintf(stderr, "Warning: using function FUNC_BLD_OPTERON_FMA4_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
  }
  /* use FMA as fallback if available*/
  if ((func == FUNC_NOT_DEFINED)&&(has_feature(FMA))) {
      /* use function for correct number of threads per core if available */
      if(threads_per_core == 1) {
          func = FUNC_HSW_COREI_FMA_1T;
          fprintf(stderr, "Warning: using function FUNC_HSW_COREI_FMA_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      if(threads_per_core == 2) {
          func = FUNC_HSW_COREI_FMA_2T;
          fprintf(stderr, "Warning: using function FUNC_HSW_COREI_FMA_2T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      /* use function for 1 threads per core if no function for actual number of thread per core exists*/
      if (func == FUNC_NOT_DEFINED)
      {
          func = FUNC_HSW_COREI_FMA_1T;
          fprintf(stderr, "Warning: using function FUNC_HSW_COREI_FMA_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
  }
  /* use FMA as fallback if available*/
  if ((func == FUNC_NOT_DEFINED)&&(has_feature(FMA))) {
      /* use function for correct number of threads per core if available */
      if(threads_per_core == 1) {
          func = FUNC_HSW_COREI_FMA_1T;
          fprintf(stderr, "Warning: using function FUNC_HSW_COREI_FMA_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      if(threads_per_core == 2) {
          func = FUNC_HSW_COREI_FMA_2T;
          fprintf(stderr, "Warning: using function FUNC_HSW_COREI_FMA_2T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      /* use function for 1 threads per core if no function for actual number of thread per core exists*/
      if (func == FUNC_NOT_DEFINED)
      {
          func = FUNC_HSW_COREI_FMA_1T;
          fprintf(stderr, "Warning: using function FUNC_HSW_COREI_FMA_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
  }
  /* use AVX as fallback if available*/
  if ((func == FUNC_NOT_DEFINED)&&(has_feature(AVX))) {
      /* use function for correct number of threads per core if available */
      if(threads_per_core == 1) {
          func = FUNC_SNB_COREI_AVX_1T;
          fprintf(stderr, "Warning: using function FUNC_SNB_COREI_AVX_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      if(threads_per_core == 2) {
          func = FUNC_SNB_COREI_AVX_2T;
          fprintf(stderr, "Warning: using function FUNC_SNB_COREI_AVX_2T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      /* use function for 1 threads per core if no function for actual number of thread per core exists*/
      if (func == FUNC_NOT_DEFINED)
      {
          func = FUNC_SNB_COREI_AVX_1T;
          fprintf(stderr, "Warning: using function FUNC_SNB_COREI_AVX_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
  }
  /* use SSE2 as fallback if available*/
  if ((func == FUNC_NOT_DEFINED)&&(has_feature(SSE2))) {
      /* use function for correct number of threads per core if available */
      if(threads_per_core == 1) {
          func = FUNC_NHM_COREI_SSE2_1T;
          fprintf(stderr, "Warning: using function FUNC_NHM_COREI_SSE2_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      if(threads_per_core == 2) {
          func = FUNC_NHM_COREI_SSE2_2T;
          fprintf(stderr, "Warning: using function FUNC_NHM_COREI_SSE2_2T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
      /* use function for 1 threads per core if no function for actual number of thread per core exists*/
      if (func == FUNC_NOT_DEFINED)
      {
          func = FUNC_NHM_COREI_SSE2_1T;
          fprintf(stderr, "Warning: using function FUNC_NHM_COREI_SSE2_1T as fallback.\n");
          fprintf(stderr, "         You can use the parameter --function to try other functions.\n");
      }
  }

  /* last resort: select function manually */
  if (func == FUNC_NOT_DEFINED){
    printf("\nNo function specified on command line (-i) and automatic selection of code-path failed.\n");
    list_functions();
    printf("\nPlease enter function ID:\n");

    scanf("%d",&func);
    func=get_function(func);
    if (func==FUNC_UNKNOWN) return EXIT_FAILURE;
  }

  clockrate=get_cpu_clockrate(0,0);

  if (verbose) {
    printf("\nhardware information:\n");
    printf(" - number of processors: %lu\n",nr_pkg);
    printf(" - number of physical processor cores: %lu\n",nr_core);
    printf(" - number of logical processors: %lu\n",nr_cpu);
    printf(" - number of cores per processor: %lu\n",cores_per_package);
    printf(" - number of threads per core: %lu\n",threads_per_core);
    printf(" - number of NUMA nodes: %lu\n",nr_numa_node);
    printf(" - processor name: %s\n",proc_name);
    printf(" - processor clockrate: %I64u MHz\n",clockrate/1000000);
    printf(" - processor family, model, and stepping: %i - %i - %i\n",family,model,stepping);
    printf(" - processor features: %s\n",buffer);
    printf("\n");
  }

  if (time) printf("\nRunning FIRESTARTER with %i threads for %i seconds.",nr_threads,time);
  else printf("\nRunning FIRESTARTER with %i threads. Press Ctrl-C to abort.",nr_threads);

  switch (func) {
    case FUNC_KNL_XEONPHI_AVX512_4T:
      printf("\nTaking AVX512 code path optimized for Knights_Landing - 4 thread(s) per core\n");
      break;
    case FUNC_SKL_COREI_FMA_1T:
      printf("\nTaking FMA code path optimized for Skylake - 1 thread(s) per core\n");
      break;
    case FUNC_SKL_COREI_FMA_2T:
      printf("\nTaking FMA code path optimized for Skylake - 2 thread(s) per core\n");
      break;
    case FUNC_SKL_XEONEP_AVX512_1T:
      printf("\nTaking AVX512 code path optimized for Skylake-SP - 1 thread(s) per core\n");
      break;
    case FUNC_SKL_XEONEP_AVX512_2T:
      printf("\nTaking AVX512 code path optimized for Skylake-SP - 2 thread(s) per core\n");
      break;
    case FUNC_HSW_COREI_FMA_1T:
      printf("\nTaking FMA code path optimized for Haswell - 1 thread(s) per core\n");
      break;
    case FUNC_HSW_COREI_FMA_2T:
      printf("\nTaking FMA code path optimized for Haswell - 2 thread(s) per core\n");
      break;
    case FUNC_HSW_XEONEP_FMA_1T:
      printf("\nTaking FMA code path optimized for Haswell-EP - 1 thread(s) per core\n");
      break;
    case FUNC_HSW_XEONEP_FMA_2T:
      printf("\nTaking FMA code path optimized for Haswell-EP - 2 thread(s) per core\n");
      break;
    case FUNC_SNB_COREI_AVX_1T:
      printf("\nTaking AVX code path optimized for Sandy Bridge - 1 thread(s) per core\n");
      break;
    case FUNC_SNB_COREI_AVX_2T:
      printf("\nTaking AVX code path optimized for Sandy Bridge - 2 thread(s) per core\n");
      break;
    case FUNC_SNB_XEONEP_AVX_1T:
      printf("\nTaking AVX code path optimized for Sandy Bridge-EP - 1 thread(s) per core\n");
      break;
    case FUNC_SNB_XEONEP_AVX_2T:
      printf("\nTaking AVX code path optimized for Sandy Bridge-EP - 2 thread(s) per core\n");
      break;
    case FUNC_NHM_COREI_SSE2_1T:
      printf("\nTaking SSE2 code path optimized for Nehalem - 1 thread(s) per core\n");
      break;
    case FUNC_NHM_COREI_SSE2_2T:
      printf("\nTaking SSE2 code path optimized for Nehalem - 2 thread(s) per core\n");
      break;
    case FUNC_NHM_XEONEP_SSE2_1T:
      printf("\nTaking SSE2 code path optimized for Nehalem-EP - 1 thread(s) per core\n");
      break;
    case FUNC_NHM_XEONEP_SSE2_2T:
      printf("\nTaking SSE2 code path optimized for Nehalem-EP - 2 thread(s) per core\n");
      break;
    case FUNC_BLD_OPTERON_FMA4_1T:
      printf("\nTaking FMA4 code path optimized for Bulldozer - 1 thread(s) per core\n");
      break;
    case FUNC_ZEN_EPYC_ZEN_FMA_1T:
      printf("\nTaking ZEN_FMA code path optimized for Naples - 1 thread(s) per core\n");
      break;
    case FUNC_ZEN_EPYC_ZEN_FMA_2T:
      printf("\nTaking ZEN_FMA code path optimized for Naples - 2 thread(s) per core\n");
      break;
  }

  threaddata = (threaddata_t*) malloc(nr_threads * sizeof(threaddata_t));
  threads = malloc(nr_threads*sizeof(HANDLE));

  gettimeofday(&ts,NULL);
  start_time=ts.tv_sec*1000000+ts.tv_usec;

  /* start worker threads */
  for (i=0;i<nr_threads;i++){
    threaddata[i].addrHigh = (unsigned long long) &LOAD_VAR;
    threaddata[i].FUNCTION = func;

    threads[i]=CreateThread(
      NULL,
      0,
      WorkerThread,
      &(threaddata[i]),
      0,
      &threadDescriptor);
  }

  if (time) { // timeout specified, terminated automatically when it is exceeded
    if (PERIOD == 0){ // 0% or 100% load
      Sleep(time*1000);
      gettimeofday(&ts,NULL);
      end_time=ts.tv_sec*1000000+ts.tv_usec;
    }
    else{
      do {
        Sleep(load_time);
        if (LOAD_VAR != LOAD_STOP) LOAD_VAR=LOAD_LOW;
        Sleep(idle_time);
        if (LOAD_VAR != LOAD_STOP) LOAD_VAR=LOAD_HIGH;

        gettimeofday(&ts,NULL);
        end_time=ts.tv_sec*1000000+ts.tv_usec; 
      }
      while (end_time-start_time <= (unsigned long long)time * 1000000);
    }

    LOAD_VAR=LOAD_STOP;
    for (i=0;i<nr_threads;i++){
      WaitForSingleObject(threads[i],INFINITE);
    }
    print_report();
  }
  else{ // no timeout specified (stopped by Ctrl-C, see handler at the bottom)
    if (PERIOD == 0){ // 0% or 100% load
      while (1) Sleep(2000000000); // master thraed sleeps until interupted (does not generate load)
    }
    else
    {
      while (1) {
        Sleep(load_time);
        if (LOAD_VAR != LOAD_STOP) LOAD_VAR=LOAD_LOW;
        Sleep(idle_time);
        if (LOAD_VAR != LOAD_STOP) LOAD_VAR=LOAD_HIGH;
      }
    }
  }

  return 0;
}

/* Handler for Ctrl-C 
 *
 * signals worker threads to stop and terminates program orderly
 */
BOOL WINAPI ConsoleHandler(DWORD type)
{
  int i;

  switch(type) {
     case CTRL_C_EVENT:
        do {
         LOAD_VAR=LOAD_STOP;
         Sleep(10);
        }
        while (LOAD_VAR!=LOAD_STOP); // catch unlikely case that master thread changes the LOAD_VAR back to LOAD_LOW or LOAD_HIGH (interupt directly after "if (LOAD_VAR != LOAD_STOP)" is evaluated)
        gettimeofday(&ts,NULL);
        end_time=ts.tv_sec*1000000+ts.tv_usec; 
        for (i=0;i<nr_threads;i++){
          WaitForSingleObject(threads[i],INFINITE);
        }
        print_report();
        return 0;
        break;
     default:
        break;
  }
  return TRUE;
}

