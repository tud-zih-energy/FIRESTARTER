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

unsigned long long HIGH=1;  /* shared variable that specifies load level */

int exitCondition;

struct threadParams{
    int param1;
    int param2;
};

static void list_functions(){

  show_version();

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
         fprintf(stderr, "\nError: unknown function id: %s, see --avail for available ids\n\n", optarg);
    }

    return func;
}

$$ thread definitions
$TEMPLATE main_win64_c.thread_definitions(dest,architectures,templates)

int main(int argc, char *argv[]){
  int nr_threads=0,i,time=0;
  int func = FUNC_NOT_DEFINED;
  int c;

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
    printf("\nNo function specified on command line (-i).\n");
    printf("\n available load-functions:\n");
    printf("  ID   | NAME                           \n");
    printf("  --------------------------------------\n");
$$ list available functions with their respective id
$TEMPLATE main_win64_c.list_functions(dest,architectures,templates)
    printf("\nPlease enter function ID:\n");

    scanf("%d",&func);
  }

  printf("\nRunning FIRESTARTER with %i threads for %i seconds",nr_threads,time);

  DWORD threadDescriptor;
  HANDLE * threads=malloc(nr_threads*sizeof(HANDLE));

  switch (func) {
$$ start threads according to selected function
$TEMPLATE main_win64_c.start_threads(dest,architectures,templates)
  }

  Sleep(time*1000);
  for (i=0;i<nr_threads;i++)
    TerminateThread(threads[i],1);
}

