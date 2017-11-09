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

#if (defined(linux) || defined(__linux__))
#define _GNU_SOURCE
#ifdef AFFINITY
#include <sched.h>
#endif
#endif

#include <signal.h>
#include <getopt.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>
#include <sys/time.h>
#include "firestarter_global.h"
#include "watchdog.h"
#include "help.h"

$MAC /* Mac OS compatibility */
$MAC #ifdef __MACH__
$MAC #include <mach/mach_time.h>
$MAC #define CLOCK_REALTIME 0
$MAC #endif
$MAC
/*
 * Header for local functions
 */
#include "work.h"
#include "cpu.h"
$CUDA #ifdef CUDA
$CUDA #include "gpu.h"
$CUDA #endif

/*
 * used for --bind option
 */
#define ADD_CPU_SET(cpu,cpuset) \
do { \
  if (cpu_allowed(cpu)) { \
    CPU_SET(cpu, &cpuset); \
  } else { \
    if (cpu >= num_cpus() ) { \
      fprintf( stderr, "Error: The given bind argument (-b/--bind) includes CPU %d that is not available on this system.\n",cpu ); \
    } \
    else { \
      fprintf( stderr, "Error: The given bind argument (-b/--bind) cannot be implemented with the cpuset given from the OS\n" ); \
      fprintf( stderr, "This can be caused by the taskset tool, cgroups, the batch system, or similar mechanisms.\n" ); \
      fprintf( stderr, "Please fix the argument to match the restrictions.\n" ); \
    } \
    exit( EACCES ); \
  } \
} while (0)

mydata_t *mdp;                          /* global data structure */
cpu_info_t *cpuinfo = NULL;             /* data structure for hardware detection */
unsigned long long LOADVAR = LOAD_HIGH; /* shared variable that specifies load level */
int ALIGNMENT = 64;                     /* alignment of buffers and data structures */
unsigned int verbose = 1;               /* enable/disable output to stdout */

/*
 * FIRESTARTER configuration, determined by evaluate_environment function
 */
unsigned long long BUFFERSIZEMEM, RAMBUFFERSIZE;
unsigned int BUFFERSIZE[3];
unsigned int NUM_THREADS = 0;
int FUNCTION = FUNC_NOT_DEFINED;

/*
 * timeout and load characteristics as defind by -t, -p, and -l
 */
long TIMEOUT = 0, PERIOD = 100000, LOAD = 100;

/*
 * pointer for CPU bind argument (-b | --bind)
 */
char *fsbind = NULL;

/*
 * temporary variables
 */
int tmp1,tmp2;

$MAC /* Mac OS compatibility */
$MAC #ifdef __MACH__
$MAC mach_timebase_info_data_t info;
$MAC double ns_per_tick;
$MAC #endif
$MAC
/*
 * worker threads
 */
pthread_t *threads;

/*
 * CPU bindings of threads
 */
#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
cpu_set_t cpuset;
#endif
unsigned long long *cpu_bind;

$MAC /* Mac OS compatibility - clock_gettime is not implemented on OSX */
$MAC #ifdef __MACH__
$MAC static int clock_gettime(int id, struct timespec *t)
$MAC {
$MAC     uint64_t ts, sec, nsec;
$MAC
$MAC     ts = (uint64_t)((double)mach_absolute_time() * ns_per_tick);
$MAC     sec = ts / 1000000000;
$MAC     nsec = ts - sec * 1000000000;
$MAC     t->tv_sec = (time_t)sec;
$MAC     t->tv_nsec = (long)nsec;
$MAC     return id;
$MAC }
$MAC #endif
$MAC 
/*
 * initialize data structures
 */
static void *init()
{
    unsigned int i, t;

#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
    cpu_set(cpu_bind[0]);
#endif
    mdp->cpuinfo = cpuinfo;

    BUFFERSIZEMEM = sizeof(char) * (2 * BUFFERSIZE[0] + BUFFERSIZE[1] + BUFFERSIZE[2] + RAMBUFFERSIZE +
                                    ALIGNMENT + 2 * sizeof(unsigned long long));

    if(BUFFERSIZEMEM <= 0){
        fprintf(stderr, "Error: Determine BUFFERSIZEMEM failed\n");
        fflush(stderr);
        exit(127);
    }

    if((NUM_THREADS > mdp->cpuinfo->num_cpus) || (NUM_THREADS == 0)){
        NUM_THREADS = mdp->cpuinfo->num_cpus;
    }

    threads = _mm_malloc(NUM_THREADS * sizeof(pthread_t), ALIGNMENT);
    mdp->thread_comm = _mm_malloc(NUM_THREADS * sizeof(int), ALIGNMENT);
    mdp->threaddata = _mm_malloc(NUM_THREADS * sizeof(threaddata_t), ALIGNMENT);
    mdp->num_threads = NUM_THREADS;
    if((threads == NULL) || (mdp->thread_comm == NULL) || (mdp->threaddata == NULL)){
        fprintf(stderr, "Error: Allocation of structure mydata_t failed\n");
        fflush(stderr);
        exit(127);
    }

    if (verbose) {
$$ TODO list core ID to package mapping
$$        if ((num_packages() != -1) && (num_cores_per_package() != -1) && (num_threads_per_core() != -1)){
$$            printf ("  num_packages: %i, %i cores per package, %i threads per core\n",
$$                    num_packages(), num_cores_per_package(),num_threads_per_core());
$$        }
        printf("  using %i threads\n", NUM_THREADS);
        #if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
            int print_core_id_info = 0;
            for (i = 0; i < NUM_THREADS; i++){
                /* avoid multiple sysfs accesses */
                tmp1=get_core_id(cpu_bind[i]);
                tmp2=get_pkg(cpu_bind[i]);
                if ((tmp1 != -1) && (tmp2 != -1)){
                    printf("    - Thread %i runs on CPU %llu, core %i in package: %i\n",
                           i, cpu_bind[i], tmp1, tmp2);
                    print_core_id_info = 1;
                }
            }
            if (print_core_id_info){ 
              printf("  The cores are numbered using the IDs from sysfs (see sys/devices/system/cpu/\n");
              printf("  cpu<no.>/topology/) or /proc/cpuinfo. These IDs do not have to be consecutive.\n");
            }
        #endif
        printf("\n");
        fflush(stdout);
    }

    // create worker threads
    for (t = 0; t < NUM_THREADS; t++) {
        mdp->ack = 0;
        mdp->threaddata[t].thread_id = t;
        mdp->threaddata[t].cpu_id = cpu_bind[t];
        mdp->threaddata[t].data = mdp;
        mdp->threaddata[t].buffersizeMem = BUFFERSIZEMEM;
        mdp->threaddata[t].iterations = 0;
        mdp->threaddata[t].flops = 0;
$$        for (i=0;i<5;i++) mdp->threaddata[t].bytes[i] = 0;
        mdp->threaddata[t].bytes = 0;
        mdp->threaddata[t].alignment = ALIGNMENT;
        mdp->threaddata[t].FUNCTION = FUNCTION;
        mdp->threaddata[t].period = PERIOD;
        mdp->thread_comm[t] = THREAD_INIT;
        i=pthread_create(&(threads[t]), NULL, thread,(void *) (&(mdp->threaddata[t])));
        while (!mdp->ack); // wait for this thread's memory allocation
        if (mdp->ack == THREAD_INIT_FAILURE) {
            fprintf(stderr,"Error: Initialization of threads failed\n");
            fflush(stderr);
            exit(127);
        }
    }
    mdp->ack = 0;

#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
    cpu_set(cpu_bind[0]);
#endif

    /* wait for threads to complete their initialization */
    for (t = 0; t < NUM_THREADS; t++) {
        mdp->ack = 0;
        mdp->thread_comm[t] = THREAD_WAIT;
        while (!mdp->ack);
    }
    mdp->ack = 0;

    return (void *) mdp;
}

static void list_functions(){

  show_version();

  printf("\n available load-functions:\n");
  printf("  ID   | NAME                           | available on this system\n");
  printf("  ----------------------------------------------------------------\n");
$$ list available functions with their respective id
$TEMPLATE main_c.list_functions(dest,architectures)

  return;
}

static int get_function(unsigned int id){
    int func=FUNC_UNKNOWN;

    switch(id){
$$ select function based on specified id
$TEMPLATE main_c.get_function_cases(dest,architectures)
       default:
         fprintf(stderr, "\nError: unknown function id: %s, see --avail for available ids\n\n", optarg);
    }

    return func;
}

/*
 * detect hardware configuration and setup FIRESTARTER accordingly
 */
static void evaluate_environment()
{
    unsigned int i;
#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
    unsigned int j = 0;
#endif
    char arch[16];

    cpuinfo = (cpu_info_t *) _mm_malloc(sizeof(cpu_info_t), 64);
    init_cpuinfo(cpuinfo, verbose);

    if((NUM_THREADS>0) && (NUM_THREADS > cpuinfo->num_cpus)){
        fprintf(stderr, "\nWarning: not enough CPUs for requested number of threads\n");
    }

    if (fsbind==NULL) { // no cpu binding defined
#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
        CPU_ZERO(&cpuset);
        if (NUM_THREADS==0){ // use all CPUs if not defined otherwise
          for (i = 0; i < cpuinfo->num_cpus; i++) {
            if (cpu_allowed(i)) {
                CPU_SET(i, &cpuset);
                NUM_THREADS++;
            }
          }
        }
        else{ // if -n / --threads is set
          int current_cpu=0;
          for (i = 0; i < NUM_THREADS; i++) {
            /* search for available cpu */
            while(! cpu_allowed(current_cpu) ) {
              current_cpu++;

              /* if reached end of avail cpus or max(int) */
              if (current_cpu >= cpuinfo->num_cpus || current_cpu < 0)
              {
                /* start at beginning */
                fprintf(stderr, "Error: You are requesting more threads than there are CPUs available in the given cpuset.\n");
                fprintf(stderr, "This can be caused by the taskset tool, cgroups, the batch system, or similar mechanisms.\n" ); \
                fprintf(stderr, "Please fix the -n/--threads argument to match the restrictions.\n");
                exit( EACCES );
              }
            }
            ADD_CPU_SET(current_cpu,cpuset);
            /* next cpu for next thread (or one of the following) */
            current_cpu++;
          }
        }
$CUDA         #ifdef CUDA
$CUDA         NUM_THREADS++; //+1 for the CUDA-thread!
$CUDA         #endif
$CUDA #elif defined(CUDA)
$CUDA         if (NUM_THREADS==0) NUM_THREADS = cpuinfo->num_cpus+1; //+1 for the CUDA-thread!
$CUDA         else NUM_THREADS++;
#else
        if (NUM_THREADS==0) NUM_THREADS = cpuinfo->num_cpus;
#endif
    }
#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
    else { // parse CPULIST for binding
        char *p,*q,*r,*s,*t;
        int p_val=0,r_val=0,s_val=0,error=0;

        CPU_ZERO(&cpuset);
        errno=0;
        p=strdup(fsbind);
        while(p!=NULL) {
            q=strstr(p,",");
            if (q) {
                *q='\0';
                q++;
            }
            s=strstr(p,"/");
            if (s) {
                *s='\0';
                s++;
                s_val=(int)strtol(s,&t,10);
                if ((errno) || ((strcmp(t,"\0") && (t[0] !=','))) ) error++;
            }
            r=strstr(p,"-");
            if (r) {
                *r='\0';
                r++;
                r_val=(int)strtol(r,&t,10);
                if ((errno) || ((strcmp(t,"\0") && (t[0] !=',') && (t[0] !='/'))) ) error++;
            }
            p_val=(int)strtol(p,&t,10);
            if ((errno) || (p_val < 0) || (strcmp(t,"\0"))) error++;
            if(error) {
                fprintf(stderr, "Error: invalid symbols in CPU list: %s\n",fsbind);
                fflush(stderr);
                exit(127);
            }
            if ((s) && (s_val<=0)) {
                fprintf(stderr, "Error: s has to be >= 0 in x-y/s expressions of CPU list: %s\n",fsbind);
                fflush(stderr);
                exit(127);
            }
            if ((r) && (r_val < p_val)) {
                fprintf(stderr, "Error: y has to be >= x in x-y expressions of CPU list: %s\n",fsbind);
                fflush(stderr);
                exit(127);
            }
            if ((s)&&(r)) for (i=p_val; (int)i<=r_val; i+=s_val) {
                ADD_CPU_SET(i,cpuset);
                NUM_THREADS++;
            }
            else if (r) for (i=p_val; (int)i<=r_val; i++) {
                ADD_CPU_SET(i,cpuset);
                NUM_THREADS++;
            }
            else {
                ADD_CPU_SET(p_val,cpuset);
                NUM_THREADS++;
            }
            p=q;
        }
        free(p);
    }
#endif

    cpu_bind = (unsigned long long *) calloc((cpuinfo->num_cpus + 1), sizeof(unsigned long long));
    if (NUM_THREADS == 0) {
        fprintf(stderr, "Error: found no useable CPUs!\n");
        fflush(stderr);
        exit(127);
    }
#if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
    else {
        for (i = 0; i < cpuinfo->num_cpus; i++) {
            if (CPU_ISSET(i, &cpuset)) {
                cpu_bind[j] = i;
                j++;
            }
        }
    }
#endif

    mdp = (mydata_t *) _mm_malloc(sizeof(mydata_t), ALIGNMENT);
    if (mdp == 0) {
        fprintf(stderr,"Error: Allocation of structure mydata_t failed (1)\n");
        fflush(stderr);
        exit(127);
    }
    memset(mdp, 0, sizeof(mydata_t));

    get_architecture(arch, sizeof(arch));
    if (strcmp(arch, "x86_64")) {
        fprintf(stderr,"Error: wrong architecture: %s, x86_64 required \n", arch);
        exit(1);
    }

    if (!feature_available("SSE2")) {
        fprintf(stderr, "Error: SSE2 not supported!\n");
        exit(1);
    }

    if (FUNCTION == FUNC_NOT_DEFINED){
     if ((strcmp("GenuineIntel", cpuinfo->vendor) == 0) || (strcmp("AuthenticAMD", cpuinfo->vendor) == 0))
     {
        switch (cpuinfo->family) {
$$ select function according to cpu family and model
$TEMPLATE main_c.evaluate_environment_set_function_cases(dest,architectures,families)
          default:
            fprintf(stderr, "\nWarning: family %i processors are not supported by this version of FIRESTARTER!\n         Check project website for updates.\n",cpuinfo->family);
        }
     }
     else {
        fprintf(stderr, "Warning: %s processors not supported by this version of FIRESTARTER!\n         Check project website for updates.\n",cpuinfo->vendor);
     }
    }

$$ select fallback function based on ISA
$TEMPLATE main_c.evaluate_environment_select_fallback_function(dest,templates)
    if (FUNCTION == FUNC_NOT_DEFINED) {
      fprintf(stderr, "Error: no fallback implementation found for available ISA extensions.\n");
      exit(1);
    }


    switch (FUNCTION) {
$$ set buffersizes according to selected code path and print corresponding debug information
$TEMPLATE main_c.evaluate_environment_set_buffersize(dest,architectures)
      default:
        fprintf(stderr, "Internal Error: missing code-path %i!\n",FUNCTION);
        exit(1);
    }

}

int main(int argc, char *argv[])
{
    int i,c;
    unsigned long long iterations=0;

$CUDA     #ifdef CUDA
$CUDA     gpustruct * structpointer=malloc(sizeof(gpustruct));
$CUDA     structpointer->useDouble=1;     //we want to use Doubles, if no -f Argument is given
$CUDA     structpointer->msize=12288;     //12288 is a good Matrixsize, it produced on nvidia k20x the biggest power-consumption
$CUDA     structpointer->useDevice=-1;    //by default, we use all GPUs with -1 option.
$CUDA     structpointer->verbose=1;       //Verbosity
$CUDA     structpointer->loadingdone=0;
$CUDA     #endif 
$CUDA 
    static struct option long_options[] = {
        {"copyright",   no_argument,        0, 'c'},
        {"help",        no_argument,        0, 'h'},
        {"version",     no_argument,        0, 'v'},
        {"warranty",    no_argument,        0, 'w'},
        {"quiet",       no_argument,        0, 'q'},
        {"report",      no_argument,        0, 'r'},
        {"avail",       no_argument,        0, 'a'},
        {"function",    required_argument,  0, 'i'},
$CUDA         #ifdef CUDA
$CUDA         {"usegpufloat", no_argument,        0, 'f'},
$CUDA         {"gpus",        required_argument,  0, 'g'},
$CUDA         {"matrixsize",  required_argument,  0, 'm'},
$CUDA         #endif
        {"threads",     required_argument,  0, 'n'},
        #if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
        {"bind",        required_argument,  0, 'b'},
        #endif
        {"timeout",     required_argument,  0, 't'},
        {"load",        required_argument,  0, 'l'},
        {"period",      required_argument,  0, 'p'},
        {0,             0,                  0,  0 }
    };

    while(1)
    {
$$ construct getopt_long command
$TEMPLATE main_c.main_getopt(dest,version)

        if(c == -1) break;

        errno = 0;
        switch(c)
        {
        case 'c':
            show_copyright();
            return EXIT_SUCCESS;
        case 'h':
            show_help();
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
            FUNCTION=get_function((unsigned int)strtol(optarg,NULL,10));
            if (FUNCTION==FUNC_UNKNOWN) return EXIT_FAILURE;
            break;
        case 'r':
            if (verbose) verbose = 2;
            break;
        case 'q':
$CUDA             #ifdef CUDA
$CUDA             structpointer->verbose=0;
$CUDA             #endif
            verbose = 0;
            break;
        case 'n':
            if (fsbind!=NULL){
                printf("Error: -b/--bind and -n/--threads cannot be used together\n");
                return EXIT_FAILURE;
            }
            NUM_THREADS=(unsigned int)strtol(optarg,NULL,10);
            break;
        #if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
        case 'b':
            if (NUM_THREADS){
                printf("Error: -b/--bind and -n/--threads cannot be used together\n");
                return EXIT_FAILURE;
            }
            fsbind=strdup(optarg);
            break;
        #endif
$CUDA         #ifdef CUDA
$CUDA         case 'f':
$CUDA             structpointer->useDouble=0; //disabling double-precision
$CUDA             break;
$CUDA         case 'm':
$CUDA             structpointer->msize=strtoull(optarg,NULL,10); //resetting the Matrixsize if the user wants to.
$CUDA             if ((errno != 0) || (structpointer->msize < 64)) {
$CUDA                 printf("Error: matrixsize out of range (<64) or not a number: %s\n",optarg);
$CUDA                 return EXIT_FAILURE;
$CUDA             }
$CUDA             break;
$CUDA         case 'g':
$CUDA             structpointer->useDevice=strtoull(optarg,NULL,10); //setting, how many GPUs the user wants to use.
$CUDA             if ((errno != 0 ) || (structpointer->useDevice <0)){
$CUDA                 printf("Error: Number of GPUs out of range (<0) or not a number: %s\n",optarg);
$CUDA                 return EXIT_FAILURE;
$CUDA             }
$CUDA             break;
$CUDA         #endif
        case 't':
            TIMEOUT = strtol(optarg,NULL,10);
            if ((errno != 0) || (TIMEOUT <= 0)) {
                printf("Error: timeout out of range or not a number: %s\n",optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'l':
            LOAD = strtol(optarg,NULL,10);
            if ((errno != 0) || (LOAD < 0) || (LOAD > 100)) {
                printf("Error: load out of range or not a number: %s\n",optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'p':
            PERIOD = strtol(optarg,NULL,10);
            if ((errno != 0) || (PERIOD <= 0)) {
                printf("Error: period out of range or not a number: %s\n",optarg);
                return EXIT_FAILURE;
            }
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

    LOAD = ( PERIOD * LOAD ) / 100;
    if ((LOAD == PERIOD) || (LOAD == 0)) PERIOD = 0;    // disable interupts for 100% and 0% load case
    if (LOAD == 0) LOADVAR = LOAD_LOW;                  // use low load routine
    watchdog_arg.timeout = (unsigned int) TIMEOUT;
    watchdog_arg.period  = (useconds_t) PERIOD;
    watchdog_arg.load    = (useconds_t) LOAD;
    watchdog_arg.loadvar = &LOADVAR;

    if(verbose){
       show_version();
       printf("This program comes with ABSOLUTELY NO WARRANTY; for details run `FIRESTARTER -w'.\n");
       printf("This is free software, and you are welcome to redistribute it\nunder certain conditions; run `FIRESTARTER -c' for details.\n");
    }

    signal(SIGTERM, sigterm_handler);
    signal(SIGINT, sigterm_handler);
$CUDA
$CUDA     #ifdef CUDA
$CUDA     pthread_t gpu_thread;
$CUDA     pthread_create(&gpu_thread,NULL,initgpu,(void*)structpointer);
$CUDA     while(structpointer->loadingdone!=1){
$CUDA       usleep(10000);
$CUDA     }
$CUDA     #endif

    evaluate_environment();
    init();

    //start worker threads
    _work(mdp, &LOADVAR);

    //start watchdog
    watchdog_arg.pid = getpid();
    watchdog_timer(&watchdog_arg);

    /* wait for threads after watchdog has requested termination */
    for(i = 0; i < mdp->num_threads; i++) pthread_join(threads[i], NULL);

    if (verbose == 2){
       unsigned long long start_tsc,stop_tsc;
       double runtime;
  
       printf("\nperformance report:\n\n");

       start_tsc=mdp->threaddata[0].start_tsc;
       stop_tsc=mdp->threaddata[0].stop_tsc;
       for(i = 0; i < mdp->num_threads; i++){
          printf("Thread %i: %llu iterations, tsc_delta: %llu\n",i,mdp->threaddata[i].iterations, mdp->threaddata[i].stop_tsc - mdp->threaddata[i].start_tsc );
          iterations+=mdp->threaddata[i].iterations;
          if (start_tsc > mdp->threaddata[i].start_tsc) start_tsc = mdp->threaddata[i].start_tsc;
          if (stop_tsc < mdp->threaddata[i].stop_tsc) stop_tsc = mdp->threaddata[i].stop_tsc;
       }
       printf("\ntotal iterations: %llu\n",iterations);
       runtime=(double)(stop_tsc - start_tsc) / (double)cpuinfo->clockrate;
       printf("runtime: %.2f seconds (%llu cycles)\n\n",runtime, stop_tsc - start_tsc);

       printf("estimated floating point performance: %.2f GFLOPS\n", (double)mdp->threaddata[0].flops*0.000000001*(double)iterations/runtime);
       printf("estimated memory bandwidth*: %.2f GB/s\n", (double)mdp->threaddata[0].bytes*0.000000001*(double)iterations/runtime);
       printf("\n* this estimate is highly unreliable if --function is used in order to select\n");
       printf("  a function that is not optimized for your architecture, or if FIRESTARTER is\n");
       printf("  executed on an unsupported architecture!\n");

       printf("\n");
    }

$CUDA     #ifdef CUDA
$CUDA     free(structpointer);
$CUDA     #endif
$CUDA
    return EXIT_SUCCESS;
}

