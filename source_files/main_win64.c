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

#include "work.h"
#include "help.h"
#include "firestarter_global.h"
#include <windows.h>
#include <getopt.h>

// FUNC_NOT_DEFINED defined in firestarter_global.h
#define FUNC_SSE        1
#define FUNC_AVX        2
#define FUNC_FMA        3

unsigned long long HIGH=1;  /* shared variable that specifies load level */

int exitCondition;

struct threadParams{
    int param1;
    int param2;
};

static void list_functions(){

  show_version();

  printf("\n available load-functions:\n");
  printf(" sse  - performs load functions using SSE2 instructions\n");
  printf(" avx  - performs load functions using AVX instructions\n");
  printf(" fma  - performs load functions using FMA3 instructions\n");
  return;
}

static DWORD WINAPI fmaThread(void* threadParams)
{
  threaddata_t * data = malloc(sizeof(threaddata_t));
  void * p= _mm_malloc(6703104*8,4096);

  data->addrMem = (unsigned long long) p;
  data->addrHigh = (unsigned long long) &HIGH;
  init_hsw_corei_fma_2t(data);
  asm_work_hsw_corei_fma_2t(data);
  return 0;
}

static DWORD WINAPI sseThread(void* threadParams)
{
  threaddata_t * data = malloc(sizeof(threaddata_t));
  void * p= _mm_malloc(6703104*8,4096);

  data->addrMem = (unsigned long long) p;
  data->addrHigh = (unsigned long long) &HIGH;
  init_nhm_xeonep_sse2_2t(data);
  asm_work_nhm_xeonep_sse2_2t(data);
  return 0;
}

static DWORD WINAPI avxThread(void* threadParams)
{
  threaddata_t * data = malloc(sizeof(threaddata_t));
  void * p= _mm_malloc(6703104*8,4096);

  data->addrMem = (unsigned long long) p;
  data->addrHigh = (unsigned long long) &HIGH;
  init_snb_corei_avx_2t(data);
  asm_work_snb_corei_avx_2t(data);
  return 0;
}

int main(int argc, char *argv[]){
  int nr_threads=0,i,time=0;
  char func = FUNC_NOT_DEFINED;
  char sse_input[256];
  int c;

  static struct option long_options[] = {
    {"copyright",   no_argument,        0, 'c'},
    {"help",        no_argument,        0, 'h'},
    {"version",     no_argument,        0, 'v'},
    {"warranty",    no_argument,        0, 'w'},
    {"avail",       no_argument,        0, 'a'},
    {"function",    required_argument,  0, 'f'},
    {"threads",     required_argument,  0, 'n'},
    {"timeout",     required_argument,  0, 't'},
    {0,             0,                  0,  0 }
  };

  while(1){
    c = getopt_long(argc, argv, "chvwaf:n:t:", long_options, NULL);
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
        list_functions();
        return EXIT_SUCCESS;
      case 'f':
        if (!strstr(optarg,"sse")) func = FUNC_SSE;
        else if (!strstr(optarg,"avx")) func = FUNC_AVX;
        else if (!strstr(optarg,"fma")) func = FUNC_FMA;
        else {
          printf("Error: unknown function \"%s\"! See --avail for available functions.\n",optarg);
          return EXIT_FAILURE;
        }
        break;
      case 'n':
        nr_threads=atoi(optarg);
        break;
      case 't':
        time=atoi(optarg);
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

  memset(sse_input,0,256);
  printf("FIRESTARTER - A Processor Stress Test Utility\n");
  printf("Copyright (C) %i TU Dresden, Center for Information Services and High Performance Computing\n\n",COPYRIGHT_YEAR);
  printf("This program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\n");
  printf("You should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n");

  if (nr_threads == 0){
    printf("\nNumber of threads not specified on command line (-n). Please specify the number of threads:\n");
    scanf("%d",&nr_threads);
  }
  if (time == 0){ 
    printf("\nNo timeout specified on command line (-t). Please specify the runtime in seconds:\n");
    scanf("%d",&time);
  }
  if (func == FUNC_NOT_DEFINED){
    func = FUNC_SSE; //default
    printf("\nNo function specified on command line (-f).\n");

    printf("\nDo you want to use AVX Instructions?(y/n):\n");
    scanf("%1s",sse_input);
    printf("\n\n");
    if (sse_input[0]=='y' || sse_input[0] == 'Y') {
      func = FUNC_AVX;
    }
    printf("\nDo you want to use FMA Instructions?(y/n):\n");
    scanf("%1s",sse_input);
    printf("\n\n");
    if (sse_input[0]=='y' || sse_input[0] == 'Y') {
      func = FUNC_FMA;
    }
  }
  if (func == FUNC_SSE)
    printf("Running FIRESTARTER - %s Version\n"
      "With %i Threads for %i Seconds\n"
      ,"SSE2",nr_threads,time);
  else if (func == FUNC_AVX)
    printf("Running FIRESTARTER - %s Version\n"
      "With %i Threads for %i Seconds\n"
      ,"AVX",nr_threads,time);
  else if (func == FUNC_FMA)
    printf("Running FIRESTARTER - %s Version\n"
      "With %i Threads for %i Seconds\n"
      ,"FMA",nr_threads,time);

  DWORD threadDescriptor;
  HANDLE * threads=malloc(nr_threads*sizeof(HANDLE));
  if (func == FUNC_SSE) for (i=0;i<nr_threads;i++){
      threads[i]=CreateThread(
          NULL,
          0,
          sseThread,
          NULL,
          0,
          &threadDescriptor);
  }
  if (func == FUNC_AVX) for (i=0;i<nr_threads;i++){
      threads[i]=CreateThread(
          NULL,
          0,
          avxThread,
          NULL,
          0,
          &threadDescriptor);
  }
  if (func == FUNC_FMA) for (i=0;i<nr_threads;i++){
      threads[i]=CreateThread(
          NULL,
          0,
          fmaThread,
          NULL,
          0,
          &threadDescriptor);
  }
  Sleep(time*1000);
  for (i=0;i<nr_threads;i++)
    TerminateThread(threads[i],1);
}

