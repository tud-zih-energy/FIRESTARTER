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

#define VERBOSE 0

unsigned long long HIGH=1;  /* shared variable that specifies load level */

static void list_functions(){

    printf("\n available load-functions:\n");
    printf("  ID   | NAME                           \n");
    printf("  --------------------------------------\n");
$$ list available functions with their respective id
$TEMPLATE main_win64_c.list_functions(dest,architectures,templates)

    return;
}

static int get_function(unsigned int id){
    int func=FUNC_UNKNOWN;

    switch(id){
$$ select function based on specified id
$TEMPLATE main_win64_c.get_function_cases(dest,architectures,templates)
       default:
         fprintf(stderr, "\nError: unknown function id: %i, see --avail for available ids\n\n", id);
         return EXIT_FAILURE;
    }

    return func;
}

static DWORD WINAPI WorkerThread(void* threadParams)
{
  int func = * (int *) threadParams; 
  threaddata_t * data = (threaddata_t*) malloc(sizeof(threaddata_t));
  void * p;

  data->addrHigh = (unsigned long long) &HIGH;

  switch (func) {
$$ select function
$TEMPLATE main_win64_c.WorkerThread_select_function(dest,architectures,templates)
    default:
      fprintf(stderr, "\nError: unknown function id: %i, see --avail for available ids\n\n", func);
      return EXIT_FAILURE;       
  }
  return 0;
}

int main(int argc, char *argv[]){
  SYSTEM_INFO SysInfo;
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION * ProcInfo;
  unsigned long nr_cpu=0, nr_core=0, threads_per_core=0, nr_pkg=0, cores_per_package=0,nr_numa_node=0;
  unsigned long ProcInfoSize,ProcInfoCount;
  int nr_threads=-1,time=0,i,c;
  int func = FUNC_NOT_DEFINED;

  static struct option long_options[] = {
    {"copyright",   no_argument,        0, 'c'},
    {"help",        no_argument,        0, 'h'},
    {"version",     no_argument,        0, 'v'},
    {"warranty",    no_argument,        0, 'w'},
    {"avail",       no_argument,        0, 'a'},
    {"function",    required_argument,  0, 'i'},
    {"threads",     required_argument,  0, 'n'},
    {"timeout",     required_argument,  0, 't'},
    {0,             0,                  0,  0 }
  };

  while(1){
    c = getopt_long(argc, argv, "chvwai:n:t:", long_options, NULL);
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
      case 'n':
        nr_threads=(unsigned int)strtol(optarg,NULL,10);
        break;
      case 't':
        time=(unsigned int)strtol(optarg,NULL,10);
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
    if ((nr_cpu > 0) && (nr_core >0)){
      threads_per_core = nr_cpu/nr_core;
      if (nr_threads == -1) nr_threads = nr_cpu; // use all logical processors if not specified otherwise
    }
  }

  if (VERBOSE) {
    printf("\nhardware information:\n");
    printf(" - number of processors: %lu\n",nr_pkg);
    printf(" - number of physical processor cores: %lu\n",nr_core);
    printf(" - number of logical processors: %lu\n",nr_cpu);
    printf(" - number of cores per processor: %lu\n",cores_per_package);
    printf(" - number of threads per core: %lu\n",threads_per_core);
    printf(" - number of NUMA nodes: %lu\n",nr_numa_node);
    printf("\n");
  }

  //TODO integrate cpuid based ISA detection for automatic function selection 

  printf("FIRESTARTER - A Processor Stress Test Utility\n");
  printf("Copyright (C) %i TU Dresden, Center for Information Services and High Performance Computing\n\n",COPYRIGHT_YEAR);
  printf("This program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\n");
  printf("You should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n");

  if (nr_threads == -1){
    printf("\nNumber of threads not specified on command line (-n) and autodetection of available CPUs failed. Please specify the number of threads:\n");
    scanf("%d",&nr_threads);
  }
  if (time == 0){ 
    printf("\nNo timeout specified on command line (-t). Please specify the runtime in seconds:\n");
    scanf("%d",&time);
  }
  if (func == FUNC_NOT_DEFINED){
    printf("\nNo function specified on command line (-i).\n");
    list_functions();
    printf("\nPlease enter function ID:\n");

    scanf("%d",&func);
  }

  printf("\nRunning FIRESTARTER with %i threads for %i seconds",nr_threads,time);
  switch (func) {
$$ print information about selected function
$TEMPLATE main_win64_c.main_function_info(dest,architectures,templates)
  }

  DWORD threadDescriptor;
  HANDLE * threads=malloc(nr_threads*sizeof(HANDLE));

  /* start worker threads */
  for (i=0;i<nr_threads;i++){
    threads[i]=CreateThread(
      NULL,
      0,
      WorkerThread,
      &func,
      0,
      &threadDescriptor);
  }

  Sleep(time*1000);
  for (i=0;i<nr_threads;i++)
    TerminateThread(threads[i],1);
}

