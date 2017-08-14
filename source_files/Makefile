###############################################################################
# FIRESTARTER - A Processor Stress Test Utility
# Copyright (C) 2017 TU Dresden, Center for Information Services and High
# Performance Computing
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contact: daniel.hackenberg@tu-dresden.de
###############################################################################

# linux compiler and flags
LINUX_CC?=gcc
# tracing options (VampirTrace/Score-P):
#CC=vtcc -vt:cc gcc -vt:inst manual -DVTRACE -DENABLE_VTRACING
#CC=scorep --user --nocompiler gcc -DENABLE_SCOREP

LINUX_C_FLAGS=-fomit-frame-pointer -Wall -std=c99 -I. -DAFFINITY
OPT_STD=-O2
OPT_ASM=-O0
LINUX_L_FLAGS=-lpthread -lrt -lm

# source and object files of assembler routines
$TEMPLATE Makefile.src_obj_files(dest,templates,version)

$CUDA # CUDA flags
$CUDA LINUX_CUDA_PATH=/opt/cuda
$CUDA LINUX_CUDA_C_FLAGS=${LINUX_C_FLAGS} -I${LINUX_CUDA_PATH}/include -DCUDA
$CUDA LINUX_CUDA_L_FLAGS=${LINUX_L_FLAGS} -lcuda -lcublas -lcudart -L${LINUX_CUDA_PATH}/lib64 -L${LINUX_CUDA_PATH}/lib -Wl,-rpath=${LINUX_CUDA_PATH}/lib64 -Wl,-rpath=${LINUX_CUDA_PATH}/lib
$CUDA
$WIN64 # windows compiler and flags
$WIN64 WIN64_CC=x86_64-w64-mingw32-gcc
$WIN64 WIN64_C_FLAGS=-Wall -std=c99
$WIN64 WIN64_L_FLAGS=-static
$WIN64
# targets

default: version linux

version:
$TAB ${LINUX_CC} --version

linux: FIRESTARTER

$CUDA cuda: FIRESTARTER_CUDA
$CUDA
$WIN64 win64: FIRESTARTER_win64.exe
$WIN64
$ALL $TEMPLATE Makefile.target_all(dest,version.targets)
$ALL
FIRESTARTER: generic.o x86.o main.o init_functions.o work.o x86.o watchdog.o help.o ${ASM_FUNCTION_OBJ_FILES}
$TAB ${LINUX_CC} -o FIRESTARTER  generic.o  main.o  init_functions.o work.o x86.o watchdog.o help.o ${ASM_FUNCTION_OBJ_FILES} ${LINUX_L_FLAGS}

$CUDA FIRESTARTER_CUDA: generic.o  x86.o work.o init_functions.o x86.o watchdog.o gpu.o main_cuda.o help_cuda.o ${ASM_FUNCTION_OBJ_FILES}
$CUDA $TAB ${LINUX_CC} -o FIRESTARTER_CUDA generic.o main_cuda.o init_functions.o work.o x86.o watchdog.o help_cuda.o ${ASM_FUNCTION_OBJ_FILES} gpu.o ${LINUX_CUDA_L_FLAGS}
$CUDA
$WIN64 FIRESTARTER_win64.exe: main_win64.o x86_win64.o init_functions_win64.o help_win64.o ${ASM_FUNCTION_OBJ_FILES_WIN}
$WIN64 $TAB ${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -o FIRESTARTER_win64.exe main_win64.o x86_win64.o init_functions_win64.o help_win64.o ${ASM_FUNCTION_OBJ_FILES_WIN} ${WIN64_L_FLAGS}
$WIN64
generic.o: generic.c cpu.h
$TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c generic.c

x86.o: x86.c cpu.h
$TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c x86.c

main.o: main.c work.h cpu.h
$TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c main.c

init_functions.o: init_functions.c work.h cpu.h
$TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c init_functions.c

work.o: work.c work.h cpu.h
$TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c work.c

watchdog.o: watchdog.h
$TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c watchdog.c -lrt -lm

help.o: help.c help.h
$TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c help.c

$WIN64 main_win64.o: main_win64.c work.h cpu.h
$WIN64 $TAB ${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c main_win64.c
$WIN64
$WIN64 x86_win64.o: x86_win64.c cpu.h
$WIN64 $TAB ${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c x86_win64.c
$WIN64
$WIN64 init_functions_win64.o: init_functions.c work.h cpu.h
$WIN64 $TAB ${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c init_functions.c -o init_functions_win64.o
$WIN64
$WIN64 help_win64.o: help.c help.h
$WIN64 $TAB ${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c help.c -o help_win64.o
$WIN64
$CUDA gpu.o: gpu.c gpu.h
$CUDA $TAB ${LINUX_CC} ${OPT_STD} ${LINUX_CUDA_C_FLAGS} -c gpu.c
$CUDA
$CUDA main_cuda.o: main.c work.h cpu.h
$CUDA $TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -o main_cuda.o -c main.c -DCUDA
$CUDA
$CUDA help_cuda.o: help.c help.h
$CUDA $TAB ${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -o help_cuda.o -c help.c -DCUDA
$CUDA
$TEMPLATE Makefile.template_rules(dest,templates,version)
clean:
$TAB rm -f *.a
$TAB rm -f *.o
$TAB rm -f FIRESTARTER
$CUDA $TAB rm -f FIRESTARTER_CUDA
$WIN64 $TAB rm -f FIRESTARTER_win64.exe
