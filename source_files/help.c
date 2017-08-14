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

#define _GNU_SOURCE

#include "firestarter_global.h"
#include "help.h"
#include <stdio.h>

void show_help(void)
{
    show_version();
    printf("This program comes with ABSOLUTELY NO WARRANTY; for details run `FIRESTARTER -w'.\n");
    printf("This is free software, and you are welcome to redistribute it\nunder certain conditions; run `FIRESTARTER -c' for details.\n");
$CUDA #ifdef CUDA
$CUDA     printf("\nUsage: FIRESTARTER_CUDA [Options]\n"
$CUDA #else
    printf("\nUsage: FIRESTARTER [Options]\n"
$CUDA #endif
           "\nOptions:\n\n"
           " -h         | --help             display usage information\n"
           " -v         | --version          display version information\n"
           " -c         | --copyright        display copyright information\n"
           " -w         | --warranty         display warranty information\n"
           " -q         | --quiet            disable output to stdout\n"
           " -r         | --report           display additional information (overridden by -q)\n"
           " -a         | --avail            list available functions\n"
           " -i ID      | --function=ID      specify integer ID of the load-function to be\n"
           "                                 used (as listed by --avail)\n"
$CUDA #ifdef CUDA
$CUDA            " -f         | --usegpufloat      use single precision matrix multiplications instead of double\n"
$CUDA            " -g         | --gpus             number of gpus to use (default: all)\n"
$CUDA            " -m         | --matrixsize       size of the matrix to calculate (default: 12288)\n"
$CUDA #endif
           " -t TIMEOUT | --timeout=TIMEOUT  set timeout (seconds) after which FIRESTARTER\n"
           "                                 terminates itself, default: no timeout\n"
           " -l LOAD    | --load=LOAD        set the percentage of high CPU load to LOAD (%%),\n"
           "                                 default 100, valid values: 0 <= LOAD <= 100,\n"
           "                                 threads will be idle in the remaining time,\n"
           "                                 frequency of load changes is determined by -p\n"
$CUDA #ifdef CUDA
$CUDA            "            |                    This option does NOT influence the GPU workload!\n"
$CUDA #endif
           " -p PERIOD  | --period=PERIOD    set the interval length for CPUs to PERIOD (usec),\n"
           "                                 default: 100000, each interval contains a high\n"
           "                                 load and an idle phase, the percentage of \n"
           "                                 high load is defined by -l\n"
$CUDA #ifdef CUDA
$CUDA            "            |                    This option does NOT influence the GPU workload!\n"
$CUDA #endif
           " -n COUNT   | --threads=COUNT    specify the number of threads\n"
           "                                 cannot be combined with -b | --bind, which\n"
           "                                 implicitly specifies the number of threads\n"
#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
           " -b CPULIST | --bind=CPULIST     select certain CPUs\n"
           "                                 CPULIST format: \"x,y,z\", \"x-y\", \"x-y/step\",\n"
           "                                 and any combination of the above\n"
           "                                 cannot be combined with -n | --threads\n"
#endif
           "\n"
           "\nExamples:\n\n"
           "./FIRESTARTER                    - starts FIRESTARTER without timeout\n"
           "./FIRESTARTER -t 300             - starts a 5 minute run of FIRESTARTER\n"
           "./FIRESTARTER -l 50 -t 600       - starts a 10 minute run of FIRESTARTER with\n"
$CUDA #ifdef CUDA
$CUDA      "                                   50%% high load and 50%% idle time on CPUs and full load on GPUs\n"
$CUDA      "./FIRESTARTER -l 75 -p 20000000  - starts FIRESTARTER with an interval length\n"
$CUDA      "                                   of 2 sec, 1.5s high load and 0.5s idle on CPUs and full load on GPUs\n"
$CUDA #else
           "                                   50%% high load and 50%% idle time\n"
           "./FIRESTARTER -l 75 -p 20000000  - starts FIRESTARTER with an interval length\n"
           "                                   of 2 sec, 1.5s high load and 0.5s idle\n"
$CUDA #endif
           "\n");
}

$WIN64 void show_help_win64(void)
$WIN64 {
$WIN64     show_version();
$WIN64     printf("This program comes with ABSOLUTELY NO WARRANTY; for details run `FIRESTARTER -w'.\n");
$WIN64     printf("This is free software, and you are welcome to redistribute it\nunder certain conditions; run `FIRESTARTER -c' for details.\n");
$WIN64     printf("\nUsage: FIRESTARTER [Options]\n"
$WIN64            "\nOptions:\n\n"
$WIN64            " -h         | --help             display usage information\n"
$WIN64            " -v         | --version          display version information\n"
$WIN64            " -c         | --copyright        display copyright information\n"
$WIN64            " -w         | --warranty         display warranty information\n"
$WIN64            " -r         | --report           display additional information\n"
$WIN64            " -a         | --avail            list available functions\n"
$WIN64            " -i ID      | --function=ID      specify integer ID of the load-function to be\n"
$WIN64            "                                 used (as listed by --avail)\n"
$WIN64            " -t TIMEOUT | --timeout=TIMEOUT  set timeout (seconds) after which FIRESTARTER\n"
$WIN64            "                                 terminates itself, default: no timeout\n"
$WIN64            " -l LOAD    | --load=LOAD        set the percentage of high load to LOAD (%%),\n"
$WIN64            "                                 default 100, valid values: 0 <= LOAD <= 100,\n"
$WIN64            "                                 threads will be idle in the remaining time,\n"
$WIN64            "                                 frequency of load changes is determined by -p\n"
$WIN64            " -p PERIOD  | --period=PERIOD    set the interval length to PERIOD (usec),\n"
$WIN64            "                                 default: 250000, each interval contains a high\n"
$WIN64            "                                 load and an idle phase, the percentage of \n"
$WIN64            "                                 high load is defined by -l\n"
$WIN64            " -n COUNT   | --threads=COUNT    specify the number of threads\n"
$WIN64            "\n"
$WIN64            "\nExamples:\n\n"
$WIN64            "./FIRESTARTER                    - starts FIRESTARTER without timeout\n"
$WIN64            "./FIRESTARTER -t 300             - starts a 5 minute run of FIRESTARTER\n"
$WIN64            "./FIRESTARTER -l 50 -t 600       - starts a 10 minute run of FIRESTARTER with\n"
$WIN64            "                                   50%% high load and 50%% idle time\n"
$WIN64            "./FIRESTARTER -l 75 -p 20000000  - starts FIRESTARTER with an interval length\n"
$WIN64            "                                   of 2 sec, 1.5s high load and 0.5s idle\n"  
$WIN64            "\n");
$WIN64 }
$WIN64 
void show_warranty(void)
{
    show_version();
    printf("\nThis program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\n");
    printf("You should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n");
}

void show_copyright(void)
{
    show_version();
    printf("\nThis program is free software: you can redistribute it and/or modify\nit under the terms of the GNU General Public License as published by\nthe Free Software Foundation, either version 3 of the License, or\n(at your option) any later version.\n\n");
    printf("You should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n\n");
}

void show_version(void)
{

$CUDA #ifdef CUDA
$CUDA     printf("FIRESTARTER_CUDA - A Processor/GPU Stress Test Utility, Version %i.%i%s, build: %s\n",VERSION_MAJOR, VERSION_MINOR, VERSION_INFO, BUILDDATE);
$CUDA #else
    printf("FIRESTARTER - A Processor Stress Test Utility, Version %i.%i%s, build: %s\n",VERSION_MAJOR, VERSION_MINOR, VERSION_INFO, BUILDDATE);
$CUDA #endif
    printf("Copyright (C) %i TU Dresden, Center for Information Services and High Performance Computing\n",COPYRIGHT_YEAR);
}

