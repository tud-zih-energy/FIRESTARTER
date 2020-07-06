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

#define _GNU_SOURCE

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#include <pthread.h>

#ifdef ENABLE_VTRACING
#include <vt_user.h>
#endif
#ifdef ENABLE_SCOREP
#include <SCOREP_User.h>
#endif

/*
 * Header for local functions
 */
#include "work.h"
#include "cpu.h"

/*
 * low load function
 */
int low_load_function(volatile unsigned long long addrHigh, unsigned int period) __attribute__((noinline));
int low_load_function(volatile unsigned long long addrHigh, unsigned int period)
{
    int nap;

    nap = period / 100;
    __asm__ __volatile__ ("mfence;"
                  "cpuid;" ::: "eax", "ebx", "ecx", "edx");
    while(*((volatile unsigned long long *)addrHigh) == LOAD_LOW){
        __asm__ __volatile__ ("mfence;"
                      "cpuid;" ::: "eax", "ebx", "ecx", "edx");
        usleep(nap);
        __asm__ __volatile__ ("mfence;"
                      "cpuid;" ::: "eax", "ebx", "ecx", "edx");
    }

    return 0;
}

/*
 * function that performs the stress test
 */
inline void _work(volatile mydata_t *data, unsigned long long *high)
{
    unsigned int i;

    //start worker threads
    for(i = 0; i < data->num_threads; i++){
        data->ack = 0;
        data->threaddata[i].addrHigh = (unsigned long long)high;
        data->thread_comm[i] = THREAD_WORK;
        while(!data->ack); // wait for acknowledgment
    }
    data->ack = 0;
}

/*
 * loop for additional worker threads
 * communicating with master thread using shared variables
 */
void *thread(void *threaddata)
{
    int id = ((threaddata_t *)threaddata)->thread_id;
    volatile mydata_t *global_data = ((threaddata_t *)threaddata)->data; //communication with master thread
    threaddata_t *mydata = (threaddata_t *)threaddata;
    unsigned int tmp = 0;
    unsigned long long old = THREAD_STOP;

    /* wait untill master thread starts initialization */
    while(global_data->thread_comm[id] != THREAD_INIT);

    while(1){
        switch(global_data->thread_comm[id]){
            case THREAD_INIT: // allocate and initialize memory
                if(old != THREAD_INIT){
                    old = THREAD_INIT;
                    
                    /* set affinity  */
                    #if (defined(linux) || defined(__linux__)) && defined (AFFINITY)
                    cpu_set(((threaddata_t *) threaddata)->cpu_id);
                    #endif

                    /* allocate memory */
                    if(mydata->buffersizeMem){
                        mydata->bufferMem = _mm_malloc(mydata->buffersizeMem, mydata->alignment);
                        mydata->addrMem = (unsigned long long)(mydata->bufferMem);
                    }
                    if(mydata->bufferMem == NULL){
                        global_data->ack = THREAD_INIT_FAILURE;
                    }
                    else{ 
                        global_data->ack = id + 1; 
                    }

                    /* call init function */
                    switch (mydata->FUNCTION)
                    {
                        case FUNC_KNL_XEONPHI_AVX512_4T:
                            tmp = init_knl_xeonphi_avx512_4t(mydata);
                            break;
                        case FUNC_SKL_COREI_FMA_1T:
                            tmp = init_skl_corei_fma_1t(mydata);
                            break;
                        case FUNC_SKL_COREI_FMA_2T:
                            tmp = init_skl_corei_fma_2t(mydata);
                            break;
                        case FUNC_SKL_XEONEP_AVX512_1T:
                            tmp = init_skl_xeonep_avx512_1t(mydata);
                            break;
                        case FUNC_SKL_XEONEP_AVX512_2T:
                            tmp = init_skl_xeonep_avx512_2t(mydata);
                            break;
                        case FUNC_HSW_COREI_FMA_1T:
                            tmp = init_hsw_corei_fma_1t(mydata);
                            break;
                        case FUNC_HSW_COREI_FMA_2T:
                            tmp = init_hsw_corei_fma_2t(mydata);
                            break;
                        case FUNC_HSW_XEONEP_FMA_1T:
                            tmp = init_hsw_xeonep_fma_1t(mydata);
                            break;
                        case FUNC_HSW_XEONEP_FMA_2T:
                            tmp = init_hsw_xeonep_fma_2t(mydata);
                            break;
                        case FUNC_SNB_COREI_AVX_1T:
                            tmp = init_snb_corei_avx_1t(mydata);
                            break;
                        case FUNC_SNB_COREI_AVX_2T:
                            tmp = init_snb_corei_avx_2t(mydata);
                            break;
                        case FUNC_SNB_XEONEP_AVX_1T:
                            tmp = init_snb_xeonep_avx_1t(mydata);
                            break;
                        case FUNC_SNB_XEONEP_AVX_2T:
                            tmp = init_snb_xeonep_avx_2t(mydata);
                            break;
                        case FUNC_NHM_COREI_SSE2_1T:
                            tmp = init_nhm_corei_sse2_1t(mydata);
                            break;
                        case FUNC_NHM_COREI_SSE2_2T:
                            tmp = init_nhm_corei_sse2_2t(mydata);
                            break;
                        case FUNC_NHM_XEONEP_SSE2_1T:
                            tmp = init_nhm_xeonep_sse2_1t(mydata);
                            break;
                        case FUNC_NHM_XEONEP_SSE2_2T:
                            tmp = init_nhm_xeonep_sse2_2t(mydata);
                            break;
                        case FUNC_BLD_OPTERON_FMA4_1T:
                            tmp = init_bld_opteron_fma4_1t(mydata);
                            break;
                        case FUNC_ZEN_EPYC_ZEN_FMA_1T:
                            tmp = init_zen_epyc_zen_fma_1t(mydata);
                            break;
                        case FUNC_ZEN_EPYC_ZEN_FMA_2T:
                            tmp = init_zen_epyc_zen_fma_2t(mydata);
                            break;
                        default:
                            fprintf(stderr, "Error: unknown function %i\n", mydata->FUNCTION);
                            pthread_exit(NULL);
                    }
                    if (tmp != EXIT_SUCCESS){
                        fprintf(stderr, "Error in function %i\n", mydata->FUNCTION);
                        pthread_exit(NULL);
                    } 

                }
                else{
                    tmp = 100;
                    while(tmp > 0) tmp--;
                }
                break; // end case THREAD_INIT
            case THREAD_WORK: // perform stress test
                if (old != THREAD_WORK){
                    old = THREAD_WORK;
                    global_data->ack = id + 1;

                   /* record thread's start timestamp */
                   ((threaddata_t *)threaddata)->start_tsc = timestamp();

                    /* will be terminated by watchdog 
                     * watchdog also alters mydata->addrHigh to switch between high and low load function
                     */
                    while(1){
                        /* call high load function */
                        #ifdef ENABLE_VTRACING
                        VT_USER_START("HIGH_LOAD_FUNC");
                        #endif
                        #ifdef ENABLE_SCOREP
                        SCOREP_USER_REGION_BY_NAME_BEGIN("HIGH", SCOREP_USER_REGION_TYPE_COMMON);
                        #endif
                        switch (mydata->FUNCTION)
                        {
                            case FUNC_KNL_XEONPHI_AVX512_4T:
                                tmp = asm_work_knl_xeonphi_avx512_4t(mydata);
                                break;
                            case FUNC_SKL_COREI_FMA_1T:
                                tmp = asm_work_skl_corei_fma_1t(mydata);
                                break;
                            case FUNC_SKL_COREI_FMA_2T:
                                tmp = asm_work_skl_corei_fma_2t(mydata);
                                break;
                            case FUNC_SKL_XEONEP_AVX512_1T:
                                tmp = asm_work_skl_xeonep_avx512_1t(mydata);
                                break;
                            case FUNC_SKL_XEONEP_AVX512_2T:
                                tmp = asm_work_skl_xeonep_avx512_2t(mydata);
                                break;
                            case FUNC_HSW_COREI_FMA_1T:
                                tmp = asm_work_hsw_corei_fma_1t(mydata);
                                break;
                            case FUNC_HSW_COREI_FMA_2T:
                                tmp = asm_work_hsw_corei_fma_2t(mydata);
                                break;
                            case FUNC_HSW_XEONEP_FMA_1T:
                                tmp = asm_work_hsw_xeonep_fma_1t(mydata);
                                break;
                            case FUNC_HSW_XEONEP_FMA_2T:
                                tmp = asm_work_hsw_xeonep_fma_2t(mydata);
                                break;
                            case FUNC_SNB_COREI_AVX_1T:
                                tmp = asm_work_snb_corei_avx_1t(mydata);
                                break;
                            case FUNC_SNB_COREI_AVX_2T:
                                tmp = asm_work_snb_corei_avx_2t(mydata);
                                break;
                            case FUNC_SNB_XEONEP_AVX_1T:
                                tmp = asm_work_snb_xeonep_avx_1t(mydata);
                                break;
                            case FUNC_SNB_XEONEP_AVX_2T:
                                tmp = asm_work_snb_xeonep_avx_2t(mydata);
                                break;
                            case FUNC_NHM_COREI_SSE2_1T:
                                tmp = asm_work_nhm_corei_sse2_1t(mydata);
                                break;
                            case FUNC_NHM_COREI_SSE2_2T:
                                tmp = asm_work_nhm_corei_sse2_2t(mydata);
                                break;
                            case FUNC_NHM_XEONEP_SSE2_1T:
                                tmp = asm_work_nhm_xeonep_sse2_1t(mydata);
                                break;
                            case FUNC_NHM_XEONEP_SSE2_2T:
                                tmp = asm_work_nhm_xeonep_sse2_2t(mydata);
                                break;
                            case FUNC_BLD_OPTERON_FMA4_1T:
                                tmp = asm_work_bld_opteron_fma4_1t(mydata);
                                break;
                            case FUNC_ZEN_EPYC_ZEN_FMA_1T:
                                tmp = asm_work_zen_epyc_zen_fma_1t(mydata);
                                break;
                            case FUNC_ZEN_EPYC_ZEN_FMA_2T:
                                tmp = asm_work_zen_epyc_zen_fma_2t(mydata);
                                break;
                            default:
                                fprintf(stderr,"Error: unknown function %i\n",mydata->FUNCTION);
                                pthread_exit(NULL);
                        }
                        if(tmp != EXIT_SUCCESS){
                            fprintf(stderr, "Error in function %i\n", mydata->FUNCTION);
                            pthread_exit(NULL);
                        }

                        /* call low load function */
                        #ifdef ENABLE_VTRACING
                        VT_USER_END("HIGH_LOAD_FUNC");
                        VT_USER_START("LOW_LOAD_FUNC");
                        #endif
                        #ifdef ENABLE_SCOREP
                        SCOREP_USER_REGION_BY_NAME_END("HIGH");
                        SCOREP_USER_REGION_BY_NAME_BEGIN("LOW", SCOREP_USER_REGION_TYPE_COMMON);
                        #endif
                        low_load_function(mydata->addrHigh, mydata->period);
                        #ifdef ENABLE_VTRACING
                        VT_USER_END("LOW_LOAD_FUNC");
                        #endif
                        #ifdef ENABLE_SCOREP
                        SCOREP_USER_REGION_BY_NAME_END("LOW");
                        #endif

                        /* terminate if master signals end of run */
                        if(*((volatile unsigned long long *)(mydata->addrHigh)) == LOAD_STOP) {
                            ((threaddata_t *)threaddata) -> stop_tsc = timestamp();

                            pthread_exit(NULL);
                        }
                    } // end while
                }
                else{
                    tmp = 100;
                    while(tmp > 0) tmp--;
                }
                break; //end case THREAD_WORK
            case THREAD_WAIT:
                if(old != THREAD_WAIT){
                    old = THREAD_WAIT;
                    global_data->ack = id + 1;
                }
                else { 
                    tmp = 100;
                    while(tmp > 0) tmp--;
                }
                break;
            case THREAD_STOP: // exit
            default:
                pthread_exit(0);
        } 
    }
}

