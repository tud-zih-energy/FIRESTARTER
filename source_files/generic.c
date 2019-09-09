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
 * @file generic.c
 *  fallback implementations of the hardware detection using information provided by the opperating system
 *  used for unsupported architectures, in case of errors in the architecture specific detection, and if
 *  there is no architecture specific method to implement a function.
 */

#define _GNU_SOURCE

/* needed for CPU_SET macros and sched_{set|get}affinity() functions (is not available with older glibc versions) */
/* TODO check availability in MAC OS, AIX */
#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
#include <sched.h>
#endif
/* needed for sched_getcpu() (is not available with older glibc versions) */
/* TODO check availability in MAC OS, AIX */
#if (defined(linux) || defined(__linux__)) && defined(SCHED_GETCPU)
#include <utmpx.h>
#endif

#include "cpu.h"
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

/* buffer for some generic implementations */
// TODO remove global variables to allow thread safe execution of detection
static char output[_HW_DETECT_MAX_OUTPUT];
static char path[_HW_DETECT_MAX_OUTPUT];

/* avoid multiple executions of the corresponding functions */
static int num_packages_sav = 0, num_cores_per_package_sav = 0, num_threads_per_core_sav = 0, num_threads_per_package_sav = 0;

/**
 * list element for counting unique package_ids, core_ids etc.
 */
typedef struct id_element {
    int id, count;
    struct id_element *next;
} id_le;

/******************************************************************************
 * internally used routines
 ******************************************************************************/

static void insert_element(int id, id_le **list)
{
    id_le *new_element, *ptr;

    new_element = malloc(sizeof(id_le));
    new_element->id = id;
    new_element->count = 1;
    new_element->next = NULL;

    if(*list == NULL) {
        *list = new_element;
    }
    else {
        ptr = *list;
        while(ptr->next != NULL) {
            ptr = ptr->next;
        }
        ptr->next = new_element;
    }
}

static void free_id_list(id_le **list)
{
    id_le * ptr, *tofree;
    ptr = *list;

    if(ptr != NULL)
    {
        while(ptr != NULL) {
            tofree = ptr;
            ptr = ptr->next;
            free(tofree);
        }
    }
    *list = NULL;
}

static int id_total_count(id_le *list)
{
    int c = 0;
    if(list != NULL) {
        c++;
        while(list->next != NULL) {
            c++;
            list = list->next;
        }
    }
    return c;
}

static void inc_id_count(int id, id_le **list)
{
    id_le *ptr;
    ptr = *list;

    if(ptr == NULL)
        insert_element(id, list);
    else
    {
        while(ptr->next != NULL && ptr->id != id) ptr = ptr->next;
        if(ptr->id == id)
            ptr->count++;
        else
            insert_element(id, list);
    }
}

/**
 * reads a certain data element from /proc/cpuinfo
 */
static int get_proc_cpuinfo_data(char *element, char *result, int proc)
{
    FILE *f;
    char buffer[_HW_DETECT_MAX_OUTPUT];
    char* ret;
    int h, cur_proc = -1;

    if(!element || !result) return -1;

    if((f=fopen("/proc/cpuinfo", "r")) != NULL) {
        while(!feof(f)) {
            ret = fgets(buffer, sizeof(buffer), f);
            if (ret == NULL) return -1;
            if(!strncmp(buffer, "processor", 9)) {
                cur_proc = atoi(strstr(buffer, ":")+2);
            }
            if(cur_proc == proc && !strncmp(buffer, element, strlen(element))) {
                strncpy(result, strstr(buffer, ":")+2,_HW_DETECT_MAX_OUTPUT);
                h=strlen(result)-1;
                if(result[h] == '\n') result[h] = '\0';
                fclose(f);
                return 0;
            }
        }
        fclose(f);
    }
    return -1;
}

static int match_str(char * mstr, char * pattern, int * n)
{
    char * pend;
    int l;
    if (mstr == NULL || n == NULL || pattern == NULL) {
        return 0;
    }
    l = strlen(pattern);
    if(!strncmp(mstr, pattern, l)) {
        *n = strtol(mstr+l, &pend, 10);
        if(pend == NULL) {
            return 0;
        }
        //check if there are any following non-number characters:
        if(strlen(pend) > 0) {
            return 0;
        }
    }
    else { return 0; }

    return -1;
}

/**
 * reads the file from path into buffer
 */
static int read_file(char * path, char * buffer, int bsize)
{
    FILE * f;
    long size;
    size_t read;

    if((path == NULL) || (buffer == NULL)) return 0;
    memset(buffer, 0, bsize);
    bsize--;
    if((f=fopen(path, "rb")) != NULL)
    {
        fseek(f, 0, SEEK_END);
        size=ftell(f);
        rewind(f);
        read = fread(buffer, 1, (bsize < size) ? bsize : size, f);
        if (!read) return -1;
        fclose(f);
    }
    else return 0;

    while(*buffer++) if(*buffer == '\n') *buffer = 0;

    return -1;
}

/**
 * tries to determine on which cpu the program is being run
 */
static int get_cpu()
{
    int cpu=-1;
    #if (defined(linux) || defined(__linux__)) && defined (SCHED_GETCPU)
        cpu = sched_getcpu();
    #endif
    return cpu;
}

/******************************************************************************
 * auxiliary functions
 ******************************************************************************/

/**
 * Determines scaling governor, which influences how processor clockspeed is estimated
 */
int scaling_governor(int cpu, char* output, size_t len)
{
    if (cpu==-1) { cpu = get_cpu(); }
    if (cpu!=-1)
    {
        snprintf(path,sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu);
        if(!read_file(path, output, len)) { return -1; }

        return 0;
    }
    else { return -1; }
}

/******************************************************************************
 * architecture independent fallback implementations
 ******************************************************************************/

/**
 * Determine number of CPUs in System
 */
int num_cpus()
{
    struct dirent **namelist;
    int ndir, c, num=0, m;
    char tmppath[_HW_DETECT_MAX_OUTPUT];
    char buf[20];

    /* try sysconf online cpus */
    num = sysconf(_SC_NPROCESSORS_ONLN);

    /* extract information from sysfs */
    if (num <= 0) {
        strcpy(path, "/sys/devices/system/cpu/");
        ndir = scandir(path, &namelist, 0, 0);
        if(ndir >= 0)
        {
            c = 0;
            while(ndir--) {
                if(match_str(namelist[ndir]->d_name, "cpu", &m)) {
                    strncpy(tmppath, path, sizeof(tmppath));
                    snprintf(buf, sizeof(buf), "cpu%i", m);
                    strcat(tmppath, buf);
                    strcat(tmppath, "/online");
                    read_file(tmppath, output, sizeof(output));
                    if (strncmp(output,"0",1) != 0) {
                        c++;
                    }
                }
                free(namelist[ndir]);
            }
            free(namelist);
            if(c > 0) num = c;
            else num = -1;
        }
    }

    /*TODO proc/cpuinfo*/

    /* try sysconf configured cpus */
    if (num <= 0) { num = sysconf(_SC_NPROCESSORS_CONF); }

    /*assume 1 if all detection methods fail*/
    if (num < 1) { num = 1; }

    return num;
}

void generic_get_architecture(char* arch)
{
    get_proc_cpuinfo_data("arch", arch, 0);
}

/**
 * tries to determine the physical package, a cpu belongs to
 */
int get_pkg(int cpu)
{
    int pkg=-1;
    char buffer[10];

    if (cpu == -1) { cpu = get_cpu(); }
    if (cpu != -1)
    {
        sprintf(path, "/sys/devices/system/cpu/cpu%i/topology/physical_package_id", cpu);
        if( read_file(path, buffer, sizeof(buffer)) ) pkg = atoi(buffer);

        /* fallbacks if sysfs is not working */
        if (pkg == -1)
        {
            /* assume 0 if there is only one CPU or only one package */
            if ((num_cpus() == 1) || (num_packages() == 1)) { pkg = 0; }
            /* get the physical package id from /proc/cpuinfo */
            else if(!get_proc_cpuinfo_data("physical id", buffer, cpu)) { pkg = atoi(buffer); }
            /* if the number of cpus equals the number of packages assume pkg_id = cpu_id*/
            else if (num_cpus() == num_packages()) { pkg = cpu; }
            /* if there is only one core per package assume pkg_id = core_id */
            else if (num_cores_per_package() == 1) { pkg = get_core_id(cpu); }
            /* if the number of packages equals the number of numa nodes assume pkg_id = numa node */
            else if (num_numa_nodes() == num_packages()) { pkg = get_numa_node(cpu); }

            /* NOTE pkg_id in UMA Systems with multiple sockets and more than 1 Core per socket can't be determined
            without correct topology information in sysfs*/
        }
    }

    return pkg;
}

/**
 * tries to determine the core ID, a cpu belongs to
 */
int get_core_id(int cpu)
{
    int core=-1;
    char buffer[10];

    if (cpu == -1) { cpu = get_cpu(); }
    if (cpu != -1)
    {
        sprintf(path, "/sys/devices/system/cpu/cpu%i/topology/core_id", cpu);
        if(read_file(path, buffer, sizeof(buffer))) core = atoi(buffer);

        /* fallbacks if sysfs is not working */
        if (core == -1)
        {
            /* assume 0 if there is only one CPU */
            if (num_cpus() == 1) { core = 0; }
            /* if each package contains only one cpu assume core_id = package_id = cpu_id */
            else if (num_cores_per_package() == 1) { core = 0; }

            /* NOTE core_id can't be determined without correct topology information in sysfs if there are multiple cores per package
               TODO /proc/cpuinfo */
        }
    }

    return core;
}

/**
 * determines how many NUMA Nodes are in the system
 */
int num_numa_nodes()
{
    struct dirent **namelist;
    int ndir, c, m;

    strcpy(path, "/sys/devices/system/node/");

    ndir = scandir(path, &namelist, 0, 0);
    if(ndir >= 0)
    {
        c = 0;
        while(ndir--) {
            if(match_str(namelist[ndir]->d_name, "node", &m)) { c++; }
            free(namelist[ndir]);
        }
        free(namelist);
        if(c > 0) { return c; }
    }
    return -1;
}

/**
 * tries to determine the NUMA Node, a cpu belongs to
 */
int get_numa_node(int cpu)
{
    int node = -1, ndir, m;
    struct dirent **namelist;
    struct stat statbuf;

    if (cpu == -1) { cpu = get_cpu(); }
    if (cpu != -1)
    {
        strcpy(path, "/sys/devices/system/node/");
        ndir = scandir(path, &namelist, 0, 0);
        if (ndir >= 0)
        {
            while(ndir--)
            {
                if(match_str(namelist[ndir]->d_name, "node", &m))
                {
                    sprintf(path, "/sys/devices/system/node/node%i/cpu%i", m, cpu);
                    if(!stat(path, &statbuf)) {
                        node = m;
                    }
                }
                free(namelist[ndir]);
            }
            free(namelist);
        }
    }

    return node;
}

/**
 * initializes cpuinfo-struct
 * @param print detection-summary is written to stdout when !=0
 */
void init_cpuinfo(cpu_info_t *cpuinfo,int print)
{
    unsigned int i;
    char output[_HW_DETECT_MAX_OUTPUT];

    /* initialize data structure */
    memset(cpuinfo,0,sizeof(cpu_info_t));
    strcpy(cpuinfo->architecture,"unknown\0");
    strcpy(cpuinfo->vendor,"unknown\0");
    strcpy(cpuinfo->model_str,"unknown\0");

    cpuinfo->num_cpus               = num_cpus();
    get_architecture(cpuinfo->architecture, sizeof(cpuinfo->architecture));
    get_cpu_vendor(cpuinfo->vendor, sizeof(cpuinfo->vendor));
    get_cpu_name(cpuinfo->model_str, sizeof(cpuinfo->model_str));
    cpuinfo->family                 = get_cpu_family();
    cpuinfo->model                  = get_cpu_model();
    cpuinfo->stepping               = get_cpu_stepping();
    cpuinfo->num_cores_per_package  = num_cores_per_package();
    cpuinfo->num_threads_per_core   = num_threads_per_core();
    cpuinfo->num_packages           = num_packages();
    cpuinfo->clockrate              = get_cpu_clockrate(1, 0);

    /* setup supported feature list*/
    if(!strcmp(cpuinfo->architecture,"x86_64")) cpuinfo->features   |= X86_64;
    if (feature_available("SMT")) cpuinfo->features                 |= SMT;
    if (feature_available("FPU")) cpuinfo->features                 |= FPU;
    if (feature_available("MMX")) cpuinfo->features                 |= MMX;
    if (feature_available("MMX_EXT")) cpuinfo->features             |= MMX_EXT;
    if (feature_available("SSE")) cpuinfo->features                 |= SSE;
    if (feature_available("SSE2")) cpuinfo->features                |= SSE2;
    if (feature_available("SSE3")) cpuinfo->features                |= SSE3;
    if (feature_available("SSSE3")) cpuinfo->features               |= SSSE3;
    if (feature_available("SSE4.1")) cpuinfo->features              |= SSE4_1;
    if (feature_available("SSE4.2")) cpuinfo->features              |= SSE4_2;
    if (feature_available("SSE4A")) cpuinfo->features               |= SSE4A;
    if (feature_available("ABM")) cpuinfo->features                 |= ABM;
    if (feature_available("POPCNT")) cpuinfo->features              |= POPCNT;
    if (feature_available("AVX")) cpuinfo->features                 |= AVX;
    if (feature_available("AVX2")) cpuinfo->features                |= AVX2;
    if (feature_available("FMA")) cpuinfo->features                 |= FMA;
    if (feature_available("FMA4")) cpuinfo->features                |= FMA4;
    if (feature_available("AES")) cpuinfo->features                 |= AES;
    if (feature_available("AVX512")) cpuinfo->features              |= AVX512;

    /* determine cache details */
    for (i=0; i<(unsigned int)num_caches(0); i++)
    {
        cpuinfo->Cache_shared[cache_level(0,i)-1]=cache_shared(0,i);
        cpuinfo->Cacheline_size[cache_level(0,i)-1]=cacheline_length(0,i);
        if (cpuinfo->Cachelevels < (unsigned int)cache_level(0,i)) { cpuinfo->Cachelevels = cache_level(0,i); }
        switch (cache_type(0,i))
        {
        case UNIFIED_CACHE: {
            cpuinfo->Cache_unified[cache_level(0,i)-1]=1;
            cpuinfo->U_Cache_Size[cache_level(0,i)-1]=cache_size(0,i);
            cpuinfo->U_Cache_Sets[cache_level(0,i)-1]=cache_assoc(0,i);
            break;            
        }
        case DATA_CACHE: {
            cpuinfo->Cache_unified[cache_level(0,i)-1]=0;
            cpuinfo->D_Cache_Size[cache_level(0,i)-1]=cache_size(0,i);
            cpuinfo->D_Cache_Sets[cache_level(0,i)-1]=cache_assoc(0,i);
            break;
        }
        case INSTRUCTION_CACHE: {
            cpuinfo->Cache_unified[cache_level(0,i)-1]=0;
            cpuinfo->I_Cache_Size[cache_level(0,i)-1]=cache_size(0,i);
            cpuinfo->I_Cache_Sets[cache_level(0,i)-1]=cache_assoc(0,i);
            break;
        }
        default:
            break;
        }
    }

    /* print a summary */
    if (print)
    {
        fflush(stdout);
        printf("\n  system summary:\n");
        if(cpuinfo->num_packages) printf("    number of processors: %i\n",cpuinfo->num_packages);
        if(cpuinfo->num_cores_per_package) printf("    number of cores per package: %i\n",cpuinfo->num_cores_per_package);
        if(cpuinfo->num_threads_per_core) printf("    number of threads per core: %i\n",cpuinfo->num_threads_per_core);
        if(cpuinfo->num_cpus) printf("    total number of threads: %i\n",cpuinfo->num_cpus);
        printf("\n  processor characteristics:\n");
        printf("    architecture:   %s\n",cpuinfo->architecture);
        printf("    vendor:         %s\n",cpuinfo->vendor);
        printf("    processor-name: %s\n",cpuinfo->model_str);
        printf("    model:          Family %i, Model %i, Stepping %i\n",cpuinfo->family,cpuinfo->model,cpuinfo->stepping);
        printf("    frequency:      %llu MHz\n",cpuinfo->clockrate/1000000);
        fflush(stdout);
        printf("    supported features:\n      -");
        if(cpuinfo->features&X86_64)    printf(" X86_64");
        if(cpuinfo->features&FPU)       printf(" FPU");
        if(cpuinfo->features&MMX)       printf(" MMX");
        if(cpuinfo->features&MMX_EXT)   printf(" MMX_EXT");
        if(cpuinfo->features&SSE)       printf(" SSE");
        if(cpuinfo->features&SSE2)      printf(" SSE2");
        if(cpuinfo->features&SSE3)      printf(" SSE3");
        if(cpuinfo->features&SSSE3)     printf(" SSSE3");
        if(cpuinfo->features&SSE4_1)    printf(" SSE4.1");
        if(cpuinfo->features&SSE4_2)    printf(" SSE4.2");
        if(cpuinfo->features&SSE4A)     printf(" SSE4A");
        if(cpuinfo->features&POPCNT)    printf(" POPCNT");
        if(cpuinfo->features&AVX)       printf(" AVX");
        if(cpuinfo->features&AVX2)      printf(" AVX2");
        if(cpuinfo->features&AVX512)    printf(" AVX512");
        if(cpuinfo->features&FMA)       printf(" FMA");
        if(cpuinfo->features&FMA4)      printf(" FMA4");
        if(cpuinfo->features&AES)       printf(" AES");
        if(cpuinfo->features&SMT)       printf(" SMT");
        printf("    \n");
        if(cpuinfo->Cachelevels)
        {
            printf("    Caches:\n");
            for(i = 0; i < (unsigned int)num_caches(0); i++)
            {
                snprintf(output,sizeof(output),"n/a");
                if (cache_info(0, i, output, sizeof(output)) != -1) printf("      - %s\n",output);
            }
        }
    }
    fflush(stdout);
}

#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)

/**
 * pin process to a cpu
 */
int cpu_set(int id)
{
    cpu_set_t  mask;

    CPU_ZERO( &mask );
    CPU_SET( id , &mask );
    return sched_setaffinity(0, sizeof(cpu_set_t), &mask);
}

/**
 * check if a cpu is allowed to be used
 */
int cpu_allowed(int id)
{
    cpu_set_t  mask;

    CPU_ZERO( &mask );
    if (!sched_getaffinity(0, sizeof(cpu_set_t), &mask))
    {
        return CPU_ISSET( id, &mask );
    }
    return 0;
}

#endif

/*
 * generic implementations for architecture dependent functions
 */
int generic_get_cpu_vendor(char* vendor)
{
    return get_proc_cpuinfo_data("vendor", vendor, 0);
}

int generic_get_cpu_name(char* name)
{
    return get_proc_cpuinfo_data("model name", name, 0);
}

int generic_get_cpu_family()
{
    char buffer[_HW_DETECT_MAX_OUTPUT];
    if(!get_proc_cpuinfo_data("cpu family", buffer, 0))
        return atoi(buffer);
    else if(!get_proc_cpuinfo_data("family", buffer, 0))
        return atoi(buffer);
    else
        return -1;
}

int generic_get_cpu_model()
{
    char buffer[_HW_DETECT_MAX_OUTPUT];
    if(!get_proc_cpuinfo_data("model", buffer, 0))
        return atoi(buffer);
    else
        return -1;
}

int generic_get_cpu_stepping()
{
    char buffer[_HW_DETECT_MAX_OUTPUT];
    if(!get_proc_cpuinfo_data("stepping", buffer, 0))
        return atoi(buffer);
    else if(!get_proc_cpuinfo_data("revision", buffer, 0))
        return atoi(buffer);
    else
        return -1;
}

int feature_available(char* feature)
{
    char buffer[_HW_DETECT_MAX_OUTPUT];
    get_cpu_isa_extensions(buffer,sizeof(buffer));

    if (strstr(buffer,feature)!=NULL) return 1;
    else return 0;
}

/**
 * additional features (e.g. SSE)
 */
int generic_get_cpu_isa_extensions() {
    return -1;   /*TODO parse /proc/cpuinfo */
}


unsigned long long generic_get_cpu_clockrate_proccpuinfo_fallback(int cpu)
{
    char buffer[_HW_DETECT_MAX_OUTPUT];
    if(!get_proc_cpuinfo_data("cpu MHz", buffer, cpu)) {
        return atoll(buffer)*1000000;
    }
    else { return 0; }
}

/**
 * read clockrate from sysfs
 * @param check ignored
 * @param cpu used to find accosiated directory in sysfs
 */
unsigned long long generic_get_cpu_clockrate(int cpu)
{
    char tmp[_HW_DETECT_MAX_OUTPUT];
    unsigned long long in;

    if (cpu == -1) { cpu = get_cpu(); }
    if (cpu == -1) { return 0; }

    memset(tmp, 0, sizeof(tmp));
    scaling_governor(cpu, tmp, sizeof(tmp));

    sprintf(path, "/sys/devices/system/cpu/cpu%i/cpufreq/", cpu);

    if ( (!strcmp(tmp,"performance")) || (!strcmp(tmp,"powersave")) )
    {
        strcpy(tmp, path);
        strcat(tmp, "scaling_cur_freq");
        if (!read_file(tmp, output, _HW_DETECT_MAX_OUTPUT)) {
            strcpy(tmp, path);
            strcat(tmp, "cpuinfo_cur_freq");
            if (!read_file(tmp, output, _HW_DETECT_MAX_OUTPUT)) {
                return generic_get_cpu_clockrate_proccpuinfo_fallback(cpu);
            }
        }
    }
    else
    {
        strcpy(tmp, path);
        strcat(tmp, "scaling_max_freq");
        if (!read_file(tmp, output, _HW_DETECT_MAX_OUTPUT)) {
            strcpy(tmp, path);
            strcat(tmp, "cpuinfo_max_freq");
            if (!read_file(tmp, output, _HW_DETECT_MAX_OUTPUT)) {
                return generic_get_cpu_clockrate_proccpuinfo_fallback(cpu);
            }
        }
    }
    in = atoll(output);
    in *= 1000;

    return in;
}

/**
 * returns a timestamp from cpu-internal counters (if available)
 */
unsigned long long generic_timestamp()
{
    struct timeval tv;

    if (gettimeofday(&tv,NULL) == 0) { return ((unsigned long long)tv.tv_sec)*1000000 + tv.tv_usec; }
    else { return 0; }
}

/**
 * number of caches (of one cpu)
 */
int generic_num_caches(int cpu)
{
    struct dirent **namelist;
    int ndir, c, m;

    if (cpu==-1) { cpu = get_cpu(); }
    if (cpu==-1) { return -1; }

    sprintf(path, "/sys/devices/system/cpu/cpu%i/cache/", cpu);
    ndir = scandir(path, &namelist, 0, 0);
    if(ndir >= 0)
    {
        c = 0;
        while(ndir--) {
            if(match_str(namelist[ndir]->d_name, "index", &m)) c++;
            free(namelist[ndir]);
        }
        free(namelist);
        if(c > 0) return c;
    }
    return -1;
}


int cpu_map_to_list(char * input, char * buffer, int bsize)
{
    int pos = 0;
    char *current;
    int cur_hex;
    char *tmp;
    char buf[20];

    if(input == NULL || buffer == NULL || bsize <= 0) return 0;

    tmp = malloc((strlen(input)+1) * sizeof(char));
    memcpy(tmp, input, strlen(input)+1);
    memset(buffer, 0, bsize);

    while(strlen(tmp))
    {
        current = &(tmp[strlen(tmp)-1]);
        if(*current != ',')
        {
            cur_hex = (int)strtol(current, NULL, 16);
            if (cur_hex&0x1) {
                sprintf(buf, "cpu%i ", pos);
                strcat(buffer, buf);
            }
            if (cur_hex&0x2) {
                sprintf(buf, "cpu%i ", pos+1);
                strcat(buffer, buf);
            }
            if (cur_hex&0x4) {
                sprintf(buf, "cpu%i ", pos+2);
                strcat(buffer, buf);
            }
            if (cur_hex&0x8) {
                sprintf(buf, "cpu%i ", pos+3);
                strcat(buffer, buf);
            }
            pos += 4;
        }
        *current = '\0';
    }

    return -1;
}


/**
 * information about the cache: level, associativity...
 */
int generic_cache_info(int cpu, int id, char* output, size_t len)
{
    char tmp[_HW_DETECT_MAX_OUTPUT], tmp2[_HW_DETECT_MAX_OUTPUT];
    char tmppath[_HW_DETECT_MAX_OUTPUT];
    struct stat statbuf;

    if (cpu == -1) cpu = get_cpu();
    if (cpu == -1) return -1;

    snprintf(path,sizeof(path), "/sys/devices/system/cpu/cpu%i/cache/index%i/", cpu, id);
    memset(output, 0, len);
    if(stat(path, &statbuf)) //path doesn't exist
        return -1;

    strncpy(tmppath, path, _HW_DETECT_MAX_OUTPUT);
    strncat(tmppath, "level", (_HW_DETECT_MAX_OUTPUT-strlen(tmppath))-1);

    if(read_file(tmppath, tmp, _HW_DETECT_MAX_OUTPUT)) {
        snprintf(tmp2,_HW_DETECT_MAX_OUTPUT-1, "Level %s", tmp);
        strncat(output, tmp2, (len-strlen(output))-1);
    }

    strncpy(tmppath, path, _HW_DETECT_MAX_OUTPUT);
    strncat(tmppath, "type", (_HW_DETECT_MAX_OUTPUT-strlen(tmppath))-1);
    if(read_file(tmppath, tmp, _HW_DETECT_MAX_OUTPUT)) {
        if(!strcmp(tmp, "Unified")) {
            strncpy(tmp2, output,_HW_DETECT_MAX_OUTPUT-1);
            snprintf(output, len, "%s ", tmp);
            strncat(output, tmp2, (len-strlen(output))-1);
        }
        else {
            strncat(output, " ", (len-strlen(output))-1);
            strncat(output, tmp, (len-strlen(output))-1);
        }
    }
    strncat(output, " Cache,", (len-strlen(output))-1);

    strncpy(tmppath, path, _HW_DETECT_MAX_OUTPUT);
    strncat(tmppath, "size", (_HW_DETECT_MAX_OUTPUT-strlen(tmppath))-1);
    if(read_file(tmppath, tmp, _HW_DETECT_MAX_OUTPUT)) {
        strncat(output, " ", (len-strlen(output))-1);
        strncat(output, tmp, (len-strlen(output))-1);
    }

    strncpy(tmppath, path, _HW_DETECT_MAX_OUTPUT);
    strncat(tmppath, "ways_of_associativity", (_HW_DETECT_MAX_OUTPUT-strlen(tmppath))-1);
    if(read_file(tmppath, tmp, _HW_DETECT_MAX_OUTPUT)) {
        strncat(output, ", ", (len-strlen(output))-1);
        strncat(output, tmp, (len-strlen(output))-1);
        strncat(output, "-way set associative", (len-strlen(output))-1);
    }

    strncpy(tmppath, path,_HW_DETECT_MAX_OUTPUT);
    strncat(tmppath, "coherency_line_size", (_HW_DETECT_MAX_OUTPUT-strlen(tmppath))-1);
    if(read_file(tmppath, tmp, _HW_DETECT_MAX_OUTPUT)) {
        strncat(output, ", ", (len-strlen(output))-1);
        strncat(output, tmp, (len-strlen(output))-1);
        strncat(output, " Byte cachelines", (len-strlen(output))-1);
    }

    strncpy(tmppath, path,_HW_DETECT_MAX_OUTPUT);
    strncat(tmppath, "shared_cpu_map",(_HW_DETECT_MAX_OUTPUT-strlen(tmppath))-1);
    if(read_file(tmppath, tmp, _HW_DETECT_MAX_OUTPUT)) {
        cpu_map_to_list(tmp, tmp2, _HW_DETECT_MAX_OUTPUT);
        snprintf(tmppath,_HW_DETECT_MAX_OUTPUT, "cpu%i ", cpu);
        if(!strcmp(tmp2, tmppath))
        {
            strncat(output, ", exclusive for ", (len-strlen(output))-1);
            strncat(output, tmppath, (len-strlen(output))-1);
        }
        else
        {
            strncat(output, ", shared among ", (len-strlen(output))-1);
            strncat(output, tmp2, (len-strlen(output))-1);
        }
    }
    return 0;
}
/* additional functions to query certain information about the cache */
int generic_cache_level(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    generic_cache_info(cpu, id, tmp,sizeof(tmp));
    beg = strstr(tmp, "Level");
    if (beg == NULL) return -1;
    else beg += 6;
    end = strstr(beg," ");
    if (end != NULL) { *end = '\0'; }

    return atoi(beg);
}
unsigned long long generic_cache_size(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    generic_cache_info(cpu,id,tmp,sizeof(tmp));
    beg=strstr(tmp,",");
    if (beg==NULL) return -1;
    else beg+=2;
    end=strstr(beg,",");
    if (end!=NULL) *end='\0';
    end=strstr(beg,"K");
    if (end!=NULL)
    {
        end--;
        *end = '\0';
        return atoi(beg)*1024;
    }
    end=strstr(beg,"M");
    if (end!=NULL)
    {
        end--;
        *end='\0';
        return atoi(beg)*1024*1024;
    }

    return -1;
}
unsigned int generic_cache_assoc(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg,*end;

    generic_cache_info(cpu,id,tmp,sizeof(tmp));
    beg = strstr(tmp,",")+1;
    if (beg == NULL) return -1;
    else beg++;
    end = strstr(beg,",")+1;
    if (end == NULL) return -1;
    else end++;
    beg = end;
    end = strstr(beg,",");
    if (end != NULL) *end='\0';
    end = strstr(tmp,"-way");
    if (end != NULL) {
        *end = '\0';
        return atoi(beg);
    }
    end = strstr(tmp,"fully");
    if (end != NULL) {
        *end = '\0';
        return FULLY_ASSOCIATIVE;
    }
    return -1;
}
int generic_cache_type(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg, *end;

    generic_cache_info(cpu, id, tmp, sizeof(tmp));
    beg = tmp;
    end = strstr(beg, ",");
    if (end != NULL) { *end = '\0'; }
    else return -1;

    if (strstr(beg,"Unified") != NULL) return UNIFIED_CACHE;
    if (strstr(beg,"Trace") != NULL) return INSTRUCTION_TRACE_CACHE;
    if (strstr(beg,"Data") != NULL) return DATA_CACHE;
    if (strstr(beg,"Instruction") != NULL) return INSTRUCTION_CACHE;

    return -1;
}
int generic_cache_shared(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg, *end;
    int num = 0;

    /* re-use num for checking return value */
    num = generic_cache_info(cpu, id, tmp, sizeof(tmp));
    if ( num == -1 )
        return -1;
    num = 0;

    beg = strstr(tmp,",")+1;
    if (beg == NULL) return -1;
    else beg++;
    end = strstr(beg,",")+1;
    if (end == NULL) return -1;
    else end++;
    beg = end;
    end = strstr(beg,",")+1;
    if (end == NULL) return -1;
    else end++;
    beg = end;
    end = strstr(beg,",")+1;
    if (end == NULL) return -1;
    else end++;
    beg = end;

    while (strstr(beg,"cpu")!=NULL)
    {
        end = strstr(beg,"cpu");
        beg = end+1;
        num++;
    }

    if (num!=0) return num;
    else return -1;
}
int generic_cacheline_length(int cpu, int id) {
    char tmp[_HW_DETECT_MAX_OUTPUT];
    char *beg, *end;

    generic_cache_info(cpu,id,tmp,sizeof(tmp));
    beg = strstr(tmp,",")+1;
    if (beg == NULL) return -1;
    else beg++;
    end = strstr(beg,",")+1;
    if (end == NULL) return -1;
    else end++;
    beg = end;
    end = strstr(beg,",")+1;
    if (end == NULL) return -1;
    else end++;
    beg = end;
    end = strstr(beg,"Byte cachelines");
    if (end != NULL) { *(end--) = '\0'; }

    return atoi(beg);
}

/**
* the following four functions describe how the CPUs are distributed among packages
* num_cpus() = num_packages() * num_threads_per_package()
* num_threads_per_package() = num_cores_per_package() * num_threads_per_core()
*/
int generic_num_packages()
{
    struct dirent **namelist;
    int ndir, m;
    char tmppath[_HW_DETECT_MAX_OUTPUT];
    char buf[20];
    id_le * pkg_id_list = NULL;

    if (num_packages_sav != 0) return num_packages_sav;
    num_packages_sav = -1;

    strcpy(path, "/sys/devices/system/cpu/");
    ndir = scandir(path, &namelist, 0, 0);
    if(ndir >= 0)
    {
        while(ndir--) {
            if(match_str(namelist[ndir]->d_name, "cpu", &m)) {
                strncpy(tmppath, path, sizeof(tmppath));
                snprintf(buf, sizeof(buf), "cpu%i", m);
                strcat(tmppath, buf);
                strcat(tmppath, "/online");
                read_file(tmppath, output, sizeof(output));
                if (strncmp(output,"0",1) != 0) {
                    strncpy(tmppath, path, sizeof(tmppath));
                    snprintf(buf, sizeof(buf), "cpu%i", m);
                    strcat(tmppath, buf);
                    strcat(tmppath, "/topology/physical_package_id");
                    if(read_file(tmppath, output, sizeof(output)))
                        inc_id_count(atoi(output), &pkg_id_list);
                }
            }
            free(namelist[ndir]);
        }
        free(namelist);
        num_packages_sav = id_total_count(pkg_id_list);
        free_id_list(&pkg_id_list);
    }
    return num_packages_sav;
}

int generic_num_cores_per_package()
{
    struct dirent **namelist;
    int ndir, m, n, pkg_id_tocount = -1;
    char tmppath[_HW_DETECT_MAX_OUTPUT];
    char buf[20];
    id_le *core_id_list = NULL;

    if (num_cores_per_package_sav != 0) return num_cores_per_package_sav;
    num_cores_per_package_sav=-1;

    strcpy(path, "/sys/devices/system/cpu/");
    ndir = scandir(path, &namelist, 0, 0);
    if(ndir >= 0)
    {
        while(ndir--) {
            if(match_str(namelist[ndir]->d_name, "cpu", &m)) {
                strncpy(tmppath, path,sizeof(tmppath));
                snprintf(buf, sizeof(buf), "cpu%i", m);
                strcat(tmppath, buf);
                strcat(tmppath, "/online");
                read_file(tmppath, output, sizeof(output));
                if (strncmp(output, "0", 1)!=0) {
                    strcpy(tmppath, path);
                    sprintf(buf, "cpu%i", m);
                    strcat(tmppath, buf);
                    strcat(tmppath, "/topology/physical_package_id");
                    read_file(tmppath, output, _HW_DETECT_MAX_OUTPUT);
                    m = atoi(output);
                    if(pkg_id_tocount == -1) pkg_id_tocount = m;

                    strcpy(tmppath, path);
                    strcat(tmppath, buf);
                    strcat(tmppath, "/topology/core_id");
                    read_file(tmppath, output, _HW_DETECT_MAX_OUTPUT);
                    n = atoi(output);

                    if(m == pkg_id_tocount) /*FIXME: only counts cores from first package_id that is found, assumes that every package has the same amount of cores*/
                    {
                        //if (num<n+1) num=n+1; //doesn't work if there is a gap in between the ids
                        inc_id_count(n, &core_id_list);
                    }
                }
            }
            free(namelist[ndir]);
        }
        free(namelist);
        num_cores_per_package_sav = id_total_count(core_id_list);
        free_id_list(&core_id_list);
    }
    else num_cores_per_package_sav = -1;

    if (num_cores_per_package_sav == 0) num_cores_per_package_sav = -1;

    return num_cores_per_package_sav;
}

int generic_num_threads_per_core()
{
    struct dirent **namelist;
    int ndir, m, n, pkg_id_tocount = -1, core_id_tocount = -1;
    char tmppath[_HW_DETECT_MAX_OUTPUT];
    char buf[20];

    if (num_threads_per_core_sav != 0) return num_threads_per_core_sav;

    strcpy(path, "/sys/devices/system/cpu/");
    ndir = scandir(path, &namelist, 0, 0);
    if(ndir >= 0)
    {
        while(ndir--) {
            if(match_str(namelist[ndir]->d_name, "cpu", &m)) {
                strncpy(tmppath, path, sizeof(tmppath));
                snprintf(buf, sizeof(buf), "cpu%i", m);
                strcat(tmppath, buf);
                strcat(tmppath, "/online");
                read_file(tmppath, output, sizeof(output));
                if (strncmp(output,"0",1) != 0) {
                    strcpy(tmppath, path);
                    sprintf(buf, "cpu%i", m);
                    strcat(tmppath, buf);
                    strcat(tmppath, "/topology/core_id");
                    read_file(tmppath, output, _HW_DETECT_MAX_OUTPUT);
                    m = atoi(output);
                    if(core_id_tocount == -1) core_id_tocount = m;

                    strcpy(tmppath, path);
                    strcat(tmppath, buf);
                    strcat(tmppath, "/topology/physical_package_id");
                    read_file(tmppath, output, _HW_DETECT_MAX_OUTPUT);
                    n = atoi(output);
                    if(pkg_id_tocount == -1) pkg_id_tocount = n;

                    if(m == core_id_tocount && n == pkg_id_tocount) /*FIXME: only counts threads from the first core_id and package_id that are found, assumes that every core has the same amount of threads*/
                    {
                        num_threads_per_core_sav++;
                    }
                }
            }
            free(namelist[ndir]);
        }
        free(namelist);
    }
    else num_threads_per_core_sav = -1;

    if (num_threads_per_core_sav == 0) num_threads_per_core_sav = generic_num_threads_per_package() / generic_num_cores_per_package();
    if (num_threads_per_core_sav != generic_num_threads_per_package() / generic_num_cores_per_package()) num_threads_per_core_sav = -1;

    return num_threads_per_core_sav;
}

int generic_num_threads_per_package()
{

    struct dirent **namelist;
    int ndir, m, pkg_id_tocount = -1;
    char tmppath[_HW_DETECT_MAX_OUTPUT];
    char buf[20];

    if (num_threads_per_package_sav != 0) return num_threads_per_package_sav;

    strcpy(path, "/sys/devices/system/cpu/");
    ndir = scandir(path, &namelist, 0, 0);
    if(ndir >= 0)
    {
        while(ndir--) {
            if(match_str(namelist[ndir]->d_name, "cpu", &m)) {
                strncpy(tmppath, path, sizeof(tmppath));
                snprintf(buf, sizeof(buf), "cpu%i", m);
                strcat(tmppath, buf);
                strcat(tmppath, "/online");
                read_file(tmppath, output, sizeof(output));
                if (strncmp(output, "0", 1) != 0) {
                    strcpy(tmppath, path);
                    sprintf(buf, "cpu%i", m);
                    strcat(tmppath, buf);
                    strcat(tmppath, "/topology/physical_package_id");
                    read_file(tmppath, output, _HW_DETECT_MAX_OUTPUT);
                    m = atoi(output);
                    if(pkg_id_tocount == -1) { pkg_id_tocount = m; }

                    if(m == pkg_id_tocount) /*FIXME: only counts threads from first package_id that is found and assumes that every package has the same amount of threads*/
                    {
                        num_threads_per_package_sav++;
                    }
                }
            }
            free(namelist[ndir]);
        }
        free(namelist);
    }
    else num_threads_per_package_sav = -1;

    if (num_threads_per_package_sav == 0) num_threads_per_package_sav = -1;

    return num_threads_per_package_sav;
}

/* see cpu.h */
#if defined (__ARCH_UNKNOWN)

/*
 * use generic implementations for unknown architectures
 */

void get_architecture(char * arch) {
    generic_get_architecture(arch);
}

int get_cpu_vendor(char* vendor) {
    return generic_get_cpu_vendor(vendor);
}

int get_cpu_name(char* name) {
    return generic_get_cpu_name(name);
}

int get_cpu_family() {
    return generic_get_cpu_family();
}

int get_cpu_model() {
    return generic_get_cpu_model();
}

int get_cpu_stepping() {
    return generic_get_cpu_stepping();
}

int get_cpu_isa_extensions(char* features) {
    return generic_get_cpu_isa_extensions(features);
}

unsigned long long get_cpu_clockrate(int check, int cpu) {
    return generic_get_cpu_clockrate(check,cpu);
}

unsigned long long timestamp() {
    return generic_timestamp();
}

int num_caches(int cpu) {
    return generic_num_caches(cpu);
}

int cache_info(int cpu,int id, char* output) {
    return generic_cache_info(cpu,id,output);
}

int cache_level(int cpu, int id) {
    return generic_cache_level(cpu,id);
}

unsigned long long cache_size(int cpu, int id) {
    return generic_cache_size(cpu,id);
}

unsigned int cache_assoc(int cpu, int id) {
    return generic_cache_assoc(cpu,id);
}

int cache_type(int cpu, int id) {
    return generic_cache_type(cpu,id);
}

int cache_shared(int cpu, int id) {
    return generic_cache_shared(cpu,id);
}

int cacheline_length(int cpu, int id) {
    return generic_cacheline_length(cpu,id);
}

int num_packages() {
    return generic_num_packages();
}

int num_cores_per_package() {
    return generic_num_cores_per_package();
}

int num_threads_per_core() {
    return generic_num_threads_per_core();
}

int num_threads_per_package() {
    return generic_num_threads_per_package();
}

#endif

