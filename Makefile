###############################################################################
# FIRESTARTER - A Processor Stress Test Utility
# Copyright (C) 2019 TU Dresden, Center for Information Services and High
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

LINUX_C_FLAGS=-fomit-frame-pointer -Wall -std=gnu99 -I. -DAFFINITY
OPT_STD=-O2
OPT_ASM=-O0
LINUX_L_FLAGS=-lpthread -lm
LINUX_L_FLAGS_STATIC =
LINUX_L_FLAGS_STATIC = -static

# source and object files of assembler routines
ASM_FUNCTION_SRC_FILES=sse2_functions.c avx_functions.c zen_fma_functions.c fma_functions.c fma4_functions.c avx512_functions.c 
ASM_FUNCTION_OBJ_FILES=sse2_functions.o avx_functions.o zen_fma_functions.o fma_functions.o fma4_functions.o avx512_functions.o 
ASM_FUNCTION_SRC_FILES_WIN=sse2_functions.c avx_functions.c zen_fma_functions.c fma_functions.c fma4_functions.c avx512_functions.c 
ASM_FUNCTION_OBJ_FILES_WIN=sse2_functions_win64.o avx_functions_win64.o zen_fma_functions_win64.o fma_functions_win64.o fma4_functions_win64.o avx512_functions_win64.o 

# CUDA flags
LINUX_CUDA_PATH=/opt/cuda
LINUX_CUDA_C_FLAGS=${LINUX_C_FLAGS} -I${LINUX_CUDA_PATH}/include -DCUDA
LINUX_CUDA_L_FLAGS=${LINUX_L_FLAGS} -lrt -lcuda -lcublas -lcudart -L${LINUX_CUDA_PATH}/lib64 -L${LINUX_CUDA_PATH}/lib -Wl,-rpath=${LINUX_CUDA_PATH}/lib64 -Wl,-rpath=${LINUX_CUDA_PATH}/lib

# windows compiler and flags
WIN64_CC=x86_64-w64-mingw32-gcc
WIN64_C_FLAGS=-Wall -std=c99
WIN64_L_FLAGS=-static

# targets

default: version linux

version:
	${LINUX_CC} --version

linux: FIRESTARTER

cuda: FIRESTARTER_CUDA

win64: FIRESTARTER_win64.exe

all: linux cuda win64

FIRESTARTER: generic.o x86.o batch.o main.o init_functions.o work.o x86.o watchdog.o help.o ${ASM_FUNCTION_OBJ_FILES}
	${LINUX_CC} -o FIRESTARTER  batch.o generic.o  main.o  init_functions.o work.o x86.o watchdog.o help.o ${ASM_FUNCTION_OBJ_FILES} ${LINUX_L_FLAGS} ${LINUX_L_FLAGS_STATIC}

FIRESTARTER_CUDA: generic.o  x86.o work.o init_functions.o x86.o watchdog.o gpu.o main_cuda.o help_cuda.o ${ASM_FUNCTION_OBJ_FILES}
	${LINUX_CC} -o FIRESTARTER_CUDA generic.o main_cuda.o init_functions.o work.o x86.o watchdog.o help_cuda.o ${ASM_FUNCTION_OBJ_FILES} gpu.o ${LINUX_CUDA_L_FLAGS}

FIRESTARTER_win64.exe: main_win64.o x86_win64.o init_functions_win64.o help_win64.o ${ASM_FUNCTION_OBJ_FILES_WIN}
	${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -o FIRESTARTER_win64.exe main_win64.o x86_win64.o init_functions_win64.o help_win64.o ${ASM_FUNCTION_OBJ_FILES_WIN} ${WIN64_L_FLAGS}

generic.o: generic.c cpu.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c generic.c

x86.o: x86.c cpu.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c x86.c

main.o: main.c work.h cpu.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c main.c

batch.o: batch.c msr_safe.h batch.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c batch.c

init_functions.o: init_functions.c work.h cpu.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c init_functions.c

work.o: work.c work.h cpu.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c work.c

watchdog.o: watchdog.c watchdog.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c watchdog.c

help.o: help.c help.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -c help.c

main_win64.o: main_win64.c work.h cpu.h
	${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c main_win64.c

x86_win64.o: x86_win64.c cpu.h
	${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c x86_win64.c

init_functions_win64.o: init_functions.c work.h cpu.h
	${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c init_functions.c -o init_functions_win64.o

help_win64.o: help.c help.h
	${WIN64_CC} ${OPT_STD} ${WIN64_C_FLAGS} -c help.c -o help_win64.o

gpu.o: gpu.c gpu.h
	${LINUX_CC} ${OPT_STD} ${LINUX_CUDA_C_FLAGS} -c gpu.c

main_cuda.o: main.c work.h cpu.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -o main_cuda.o -c main.c -DCUDA

help_cuda.o: help.c help.h
	${LINUX_CC} ${OPT_STD} ${LINUX_C_FLAGS} -o help_cuda.o -c help.c -DCUDA

avx512_functions.o: avx512_functions.c
	${LINUX_CC} ${OPT_ASM} ${LINUX_C_FLAGS} -mavx512f  -c avx512_functions.c

avx512_functions_win64.o: avx512_functions.c
	${WIN64_CC} ${OPT_ASM} ${WIN64_C_FLAGS} -mavx512f  -c avx512_functions.c -o avx512_functions_win64.o

fma4_functions.o: fma4_functions.c
	${LINUX_CC} ${OPT_ASM} ${LINUX_C_FLAGS} -mfma4 -mavx  -c fma4_functions.c

fma4_functions_win64.o: fma4_functions.c
	${WIN64_CC} ${OPT_ASM} ${WIN64_C_FLAGS} -mfma4 -mavx  -c fma4_functions.c -o fma4_functions_win64.o

fma_functions.o: fma_functions.c
	${LINUX_CC} ${OPT_ASM} ${LINUX_C_FLAGS} -mfma -mavx  -c fma_functions.c

fma_functions_win64.o: fma_functions.c
	${WIN64_CC} ${OPT_ASM} ${WIN64_C_FLAGS} -mfma -mavx  -c fma_functions.c -o fma_functions_win64.o

zen_fma_functions.o: zen_fma_functions.c
	${LINUX_CC} ${OPT_ASM} ${LINUX_C_FLAGS} -mfma -mavx  -c zen_fma_functions.c

zen_fma_functions_win64.o: zen_fma_functions.c
	${WIN64_CC} ${OPT_ASM} ${WIN64_C_FLAGS} -mfma -mavx  -c zen_fma_functions.c -o zen_fma_functions_win64.o

avx_functions.o: avx_functions.c
	${LINUX_CC} ${OPT_ASM} ${LINUX_C_FLAGS} -mavx  -c avx_functions.c

avx_functions_win64.o: avx_functions.c
	${WIN64_CC} ${OPT_ASM} ${WIN64_C_FLAGS} -mavx  -c avx_functions.c -o avx_functions_win64.o

sse2_functions.o: sse2_functions.c
	${LINUX_CC} ${OPT_ASM} ${LINUX_C_FLAGS} -msse2  -c sse2_functions.c

sse2_functions_win64.o: sse2_functions.c
	${WIN64_CC} ${OPT_ASM} ${WIN64_C_FLAGS} -msse2  -c sse2_functions.c -o sse2_functions_win64.o

clean:
	rm -f *.a
	rm -f *.o
	rm -f FIRESTARTER
	rm -f FIRESTARTER_CUDA
	rm -f FIRESTARTER_win64.exe
