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

/**
 * @file cpu.h
 *  interface definition of hardware detection routines
 */

#ifndef __FIRESTARTER_CPU_H
#define __FIRESTARTER_CPU_H

#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * definitions for cache properties
 */
#define INSTRUCTION_CACHE        0x01
#define DATA_CACHE               0x02
#define UNIFIED_CACHE            0x03
#define INSTRUCTION_TRACE_CACHE  0x04
#define FULLY_ASSOCIATIVE        0x10
#define DIRECT_MAPPED            0x20

/*
 * size of buffer for functions that read data from files
 */
#define _HW_DETECT_MAX_OUTPUT 512

/**
 * check the basic architecture of the machine, each architecture needs its own implementation
 * e.g. the implementation for __ARCH_X86 is in the file x86.c
 */
#if ((defined (__x86_64__))||(defined (__x86_64))||(defined (x86_64))||(defined (__i386__))||(defined (__i386))||(defined (i386))||(defined (__i486__))||(defined (__i486))||(defined (i486))||(defined (__i586__))||(defined (__i586))||(defined (i586))||(defined (__i686__))||(defined (__i686))||(defined (i686)))
 /* see x86.c */
 #define __ARCH_X86
#else
 /* see generic.c */
 #define __ARCH_UNKNOWN
#endif

/* cpu-feature definition */
#define X86_64       0x00000001
#define FPU          0x00000002
#define SMT          0x00000004
#define MMX          0x00000008
#define MMX_EXT      0x00000010
#define SSE          0x00000020
#define SSE2         0x00000040
#define SSE3         0x00000080
#define SSSE3        0x00000100
#define SSE4_1       0x00000200
#define SSE4_2       0x00000400
#define SSE4A        0x00000800
#define ABM          0x00001000
#define POPCNT       0x00002000
#define AVX          0x00004000
#define AVX2         0x00008000
#define FMA4         0x00010000
#define FMA          0x00020000
#define AES          0x00040000
#define AVX512       0x00080000

#define MAX_CACHELEVELS 3

typedef struct cpu_info
{
  unsigned long long clockrate;
  unsigned int family,model,stepping;
  unsigned int features;
  unsigned int num_cpus;
  unsigned int num_packages;
  unsigned int num_cores_per_package;
  unsigned int num_threads_per_core;
  unsigned int Cachelevels;
  unsigned int Cache_unified[MAX_CACHELEVELS];
  unsigned int Cache_shared[MAX_CACHELEVELS];
  unsigned int Cacheline_size[MAX_CACHELEVELS];
  unsigned int I_Cache_Size[MAX_CACHELEVELS];
  unsigned int I_Cache_Sets[MAX_CACHELEVELS];
  unsigned int D_Cache_Size[MAX_CACHELEVELS];
  unsigned int D_Cache_Sets[MAX_CACHELEVELS];
  unsigned int U_Cache_Size[MAX_CACHELEVELS];
  unsigned int U_Cache_Sets[MAX_CACHELEVELS];
  char vendor[13];
  char model_str[48];
  char architecture[10];
} cpu_info_t;

/******************************************************************************
 * The following functions use architecture independent information provided by 
 * the OS 
 ******************************************************************************/

 /**
  * Determine number of CPUs in System
  * @return number of CPUs in the System
  */
 extern int num_cpus();

 /**
  * try to estimate ISA using compiler macros
  */
 extern void get_architecture(char* arch, size_t len);

 /** 
  * tries to determine the physical package, a cpu belongs to
  * @param cpu number of the cpu, -1 -> cpu the program runs on
  * @return -1 in case of an error, else physical package ID
  */
 extern int get_pkg(int cpu);

 /** 
  * tries to determine the core ID, a cpu belongs to
  * @param cpu number of the cpu, -1 -> cpu the program runs on
  * @return -1 in case of an error, else core ID
  */
 extern int get_core_id(int cpu);

 /**
  * determines how many NUMA Nodes are in the system
  * @return -1 in case of errors, 1 -> UMA, >1 -> NUMA
  */
 extern int num_numa_nodes();

 /** 
  * tries to determine the NUMA Node, a cpu belongs to
  * @param cpu number of the cpu, -1 -> cpu the program runs on
  * @return -1 in case of an error, else NUMA Node
  */
 extern int get_numa_node(int cpu);

 extern void init_cpuinfo(cpu_info_t *cpuinfo, int print);
 #ifdef AFFINITY
 extern int cpu_set(int id);
 extern int cpu_allowed(int id);
 #endif

/****************************************************************************** 
 * auxiliary functions
 ******************************************************************************/
 extern int scaling_governor(int cpu, char* output, size_t len);

/****************************************************************************** 
 * architecture specific functions
 ******************************************************************************/

 /**
  * basic information about cpus
  */
 extern int get_cpu_vendor(char* vendor, size_t len);
 extern int get_cpu_name(char* name, size_t len);
 extern int get_cpu_family();
 extern int get_cpu_model();
 extern int get_cpu_stepping();

 /**
  * additional features (e.g. SSE)
  */
 extern int get_cpu_isa_extensions(char* features, size_t len);

 /**
  * tests if a certain feature is supported
  */
 extern int feature_available(char* feature);
 
 /**
  * measures clockrate using cpu-internal counters (if available)
  * @param check if set to 1 additional checks are performed if the result is reliable
  *              see implementations in the architecture specific and generic parts for mor details
  * @param cpu the cpu that should be used, cpu affinity has to be set to the desired cpu before calling this function
  *            used to determine which cpu should be checked (e.g. relevant for finding the appropriate directory in sysfs)
  */
 extern unsigned long long get_cpu_clockrate(int check,int cpu);

 /**
  * returns a timestamp from cpu-internal counters (if available)
  */
 extern unsigned long long timestamp();

 /**
  * number of caches (of one cpu). Not equivalent to the number of cachelevels, as Inst and Data Caches for the same level
  * are counted as 2 individual cache!
  * @param cpu the cpu that should be used, cpu affinity has to be set to the desired cpu before calling this function
  *            used to determine which cpu should be checked (e.g. relevant for finding the appropriate directory in sysfs)
  */
 extern int num_caches(int cpu);

 /**
  * information about the cache: level, associativity...
  * @param cpu the cpu that should be used, cpu affinity has to be set to the desired cpu before calling this function
  *            used to determine which cpu should be checked (e.g. relevant for finding the appropriate directory in sysfs)
  * @param id id of the cache 0 <= id <= num_caches()-1
  * @param output preallocated buffer for the result string
  */
 extern int cache_info(int cpu, int id, char* output, size_t len);

 /* additional functions to query certain information about a cache */
 extern int cache_level(int cpu, int id);
 extern unsigned long long cache_size(int cpu, int id);
 extern unsigned int cache_assoc(int cpu, int id);
 extern int cache_type(int cpu, int id);
 extern int cache_shared(int cpu, int id);
 extern int cacheline_length(int cpu, int id);

 /**
  * the following four functions estimate how the CPUs are distributed among packages
  * num_cpus() = num_packages() * num_threads_per_package()
  * num_threads_per_package() = num_cores_per_package() * num_threads_per_core()
  */
 extern int num_packages();
 extern int num_cores_per_package();   /* >1 -> Multicore */
 extern int num_threads_per_core();    /* >1 -> SMT support */
 extern int num_threads_per_package(); /* >1 Multicore or SMT or both */

/****************************************************************************** 
 * architecture independent fallback functions used for unsupported architectures 
 * and in case of errors or unavailable information in the architecture dependent 
 * detection routines (see generic.c).
 ******************************************************************************/
 extern int generic_get_cpu_vendor(char* vendor);
 extern int generic_get_cpu_name(char* name);
 extern int generic_get_cpu_family();
 extern int generic_get_cpu_model();
 extern int generic_get_cpu_stepping();
 extern int generic_get_cpu_isa_extensions();
 extern unsigned long long generic_get_cpu_clockrate(int cpu);
 extern unsigned long long generic_timestamp();
 extern int generic_num_caches(int cpu);
 extern int generic_cache_info(int cpu, int id, char* output, size_t len);
 extern int generic_cache_level(int cpu, int id);
 extern unsigned long long generic_cache_size(int cpu, int id);
 extern unsigned int generic_cache_assoc(int cpu, int id);
 extern int generic_cache_type(int cpu, int id);
 extern int generic_cache_shared(int cpu, int id);
 extern int generic_cacheline_length(int cpu, int id);
 extern int generic_num_packages();
 extern int generic_num_cores_per_package();
 extern int generic_num_threads_per_core();
 extern int generic_num_threads_per_package();

#endif

