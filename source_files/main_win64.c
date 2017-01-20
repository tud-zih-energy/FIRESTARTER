/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2016 TU Dresden, Center for Information Services and High
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
#include <windows.h>

/* current version */
$TEMPLATE firestarter_global_h.version_info(dest,version.major,version.minor,version.info,version.date)
#define COPYRIGHT_YEAR 2016

unsigned long long HIGH=1;  /* shared variable that specifies load level */

int exitCondition;

struct threadParams{
    int param1;
    int param2;
};
static DWORD WINAPI fmaThread(void* threadParams)
{
  void * p= _mm_malloc(6703104*8,4096);
  init_hsw_corei_fma_2t((unsigned long long)p);
  asm_work_hsw_corei_fma_2t((unsigned long long)p,(unsigned long long)&HIGH);
  return 0;
}
static DWORD WINAPI sseThread(void* threadParams)
{
  void * p= _mm_malloc(6703104*8,4096);
  init_nhm_xeonep_sse2_2t((unsigned long long)p);
  asm_work_nhm_xeonep_sse2_2t((unsigned long long)p,(unsigned long long)&HIGH);
  return 0;
}
static DWORD WINAPI avxThread(void* threadParams)
{
  void * p= _mm_malloc(6703104*8,4096);
  init_snb_corei_avx_2t((unsigned long long)p);
  asm_work_snb_corei_avx_2t((unsigned long long)p,(unsigned long long)&HIGH);
  return 0;
}

int main(int argc, char *argv[]){
  int nr_threads=1,i,time;
  char sse=1,avx=0,fma=0;
  char sse_input[256];

  memset(sse_input,0,256);
  printf("FIRESTARTER - A Processor Stress Test Utility\n");
  printf("Copyright (C) %i TU Dresden, Center for Information Services and High Performance Computing\n\n",COPYRIGHT_YEAR);
  printf("This program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\n");
  printf("You should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n");

  printf("Please specify the number of threads:\n");
  scanf("%d",&nr_threads);
  printf("\nPlease specify the runtime in seconds:\n");
  scanf("%d",&time);
  printf("\nDo you want to use AVX Instructions?(y/n):\n");
  scanf("%1s",sse_input);
  printf("\n\n");
  if (sse_input[0]=='y' || sse_input[0] == 'Y') {
    avx=1;
    sse=0;
  }
  printf("\nDo you want to use FMA Instructions?(y/n):\n");
  scanf("%1s",sse_input);
  printf("\n\n");
  if (sse_input[0]=='y' || sse_input[0] == 'Y') {
    fma=1;
    avx=0; 
    sse=0;
  }
  if (sse)
    printf("Running FIRESTARTER - %s Version\n"
      "With %i Threads for %i Seconds\n"
      ,"SSE2",nr_threads,time);
  else if (avx)
    printf("Running FIRESTARTER - %s Version\n"
      "With %i Threads for %i Seconds\n"
      ,"AVX",nr_threads,time);
  else if (fma)
    printf("Running FIRESTARTER - %s Version\n"
      "With %i Threads for %i Seconds\n"
      ,"FMA",nr_threads,time);

  DWORD threadDescriptor;
  HANDLE * threads=malloc(nr_threads*sizeof(HANDLE));
  if (sse) for (i=0;i<nr_threads;i++){
      threads[i]=CreateThread(
          NULL,
          0,
          sseThread,
          NULL,
          0,
          &threadDescriptor);
  }
  if (avx) for (i=0;i<nr_threads;i++){
      threads[i]=CreateThread(
          NULL,
          0,
          avxThread,
          NULL,
          0,
          &threadDescriptor);
  }
  if (fma) for (i=0;i<nr_threads;i++){
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

