###############################################################################
# FIRESTARTER - A Processor Stress Test Utility
# Copyright (C) 2016 TU Dresden, Center for Information Services and High
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
LINUX_CC=gcc
# tracing options (VampirTrace/Score-P):
#LINUX_CC=vtcc -vt:cc gcc -vt:inst manual -DVTRACE -DENABLE_VTRACING
#LINUX_CC=scorep --user --nocompiler gcc -DENABLE_SCOREP

LINUX_C_FLAGS=-fomit-frame-pointer -Wall -std=c99 -I. -DAFFINITY
OPT=-O0
LINUX_L_FLAGS=-lpthread -lrt -lm

# CUDA flags
LINUX_CUDA_PATH=/opt/cuda
LINUX_CUDA_C_FLAGS=${LINUX_C_FLAGS} -I${LINUX_CUDA_PATH}/include -DCUDA
LINUX_CUDA_L_FLAGS=${LINUX_L_FLAGS} -lcuda -lcublas -lcudart -L${LINUX_CUDA_PATH}/lib64 -L${LINUX_CUDA_PATH}/lib -Wl,-rpath=${LINUX_CUDA_PATH}/lib64 -Wl,-rpath=${LINUX_CUDA_PATH}/lib
OPT_GPU=-O2

# targets
default: linux

linux: FIRESTARTER

cuda: FIRESTARTER_CUDA

all: linux cuda

FIRESTARTER: generic.o x86.o main.o work.o x86.o watchdog.o help.o sse2_functions.o avx_functions.o fma_functions.o fma4_functions.o avx512_functions.o 
	${LINUX_CC} -o FIRESTARTER  generic.o  main.o  work.o x86.o watchdog.o help.o sse2_functions.o avx_functions.o fma_functions.o fma4_functions.o avx512_functions.o  ${LINUX_L_FLAGS}

FIRESTARTER_CUDA: generic.o  x86.o work.o x86.o watchdog.o sse2_functions.o avx_functions.o fma_functions.o fma4_functions.o avx512_functions.o  gpu.o main_cuda.o help_cuda.o
	${LINUX_CC} -o FIRESTARTER_CUDA generic.o main_cuda.o work.o x86.o watchdog.o help_cuda.o sse2_functions.o avx_functions.o fma_functions.o fma4_functions.o avx512_functions.o  gpu.o ${LINUX_CUDA_L_FLAGS}

generic.o: generic.c cpu.h
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -c generic.c

x86.o: x86.c cpu.h
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -c x86.c

main.o: main.c work.h cpu.h
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -c main.c
	
work.o: work.c work.h cpu.h
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -c work.c

watchdog.o: watchdog.h
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -c watchdog.c -lrt -lm

help.o: help.c help.h 
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -c help.c

gpu.o:	gpu.c gpu.h
	${LINUX_CC} ${OPT_GPU} ${LINUX_CUDA_C_FLAGS} -c gpu.c

main_cuda.o: main.c work.h cpu.h
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -o main_cuda.o -c main.c -DCUDA

help_cuda.o: help.c help.h 
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -o help_cuda.o -c help.c -DCUDA

avx512_functions.o: avx512_functions.c
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -mavx512f  -c avx512_functions.c

fma4_functions.o: fma4_functions.c
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -mfma4 -mavx  -c fma4_functions.c

fma_functions.o: fma_functions.c
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -mfma -mavx  -c fma_functions.c

avx_functions.o: avx_functions.c
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -mavx  -c avx_functions.c

sse2_functions.o: sse2_functions.c
	${LINUX_CC} ${OPT} ${LINUX_C_FLAGS} -msse2  -c sse2_functions.c


clean:
	rm -f *.a
	rm -f *.o
	rm -f FIRESTARTER
	rm -f FIRESTARTER_CUDA

