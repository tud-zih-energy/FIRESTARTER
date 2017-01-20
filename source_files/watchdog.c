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

#include "firestarter_global.h"
#include "watchdog.h"

extern unsigned long long LOADVAR;
int TERMINATE = 0;

/* signal load changes to workers */
static void set_load(unsigned long long *loadvar, unsigned long long value)
{
$$    /* one can argue if the mfence and cpuid is necessary, but we have certain
$$     * measurements which suggest that the workers are reacting quicker on the
$$     * shared variable change if one uses these two instructions
$$     */
$$    __asm__ __volatile__ ("mfence;"
$$                  "cpuid;" ::: "eax", "ebx", "ecx", "edx");
$$    *loadvar = value;
$$    __asm__ __volatile__ ("mfence;"
$$                  "cpuid;" ::: "eax", "ebx", "ecx", "edx");
$$    
$$    DM: removed cpuid as mfence is sufficient to ensure global visibility
$$    
     *loadvar = value;
     __asm__ __volatile__ ("mfence;");
}


/* exit with zero returncode on sigterm */
void sigterm_handler()
{
    fprintf(stderr, "Caught shutdown signal, ending now ...\n");
    LOADVAR = LOAD_STOP; // required for the cases load = 100 and load = 0, which do not enter the while loop
    TERMINATE = 1;       // exit while loop used in case of 0 < load < 100
    
    //exit(EXIT_SUCCESS);
}


/* coordinates high load and low load phases
 * stops FIRESTARTER when timeout is reached
 * SPECIAL MPI Version
 */
void *watchdog_timer(watchdog_arg_t *arg)
{
    sigset_t signal_mask;
    long long timeout, time, period, load, idle, advance, load_reduction, idle_reduction;
    unsigned long long *loadvar;
    struct timespec start_ts, current;
    int sleepret;

$MAC     /* Mac OS compatibility */
$MAC     #ifdef __MACH__
$MAC       mach_timebase_info(&info);
$MAC       ns_per_tick = (double)info.numer / (double)info.denom;
$MAC     #endif
$MAC
    sigemptyset(&signal_mask);
    sigaddset(&signal_mask, SIGINT);
    sigaddset(&signal_mask, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signal_mask, NULL);

    period = arg->period;
    load = arg->load;
    idle = period - load;
    timeout = arg->timeout;
    loadvar = arg->loadvar;


    clock_gettime(CLOCK_REALTIME, &start_ts);
    time = 0;
    /* TODO: I don't like that the control flow depends on the period variable,
     * it is not wrong but confusing
     */
    while(period > 0){
        // cycles++;

        clock_gettime(CLOCK_REALTIME, &current);
        advance = ((current.tv_sec - start_ts.tv_sec) * 1000000 + (current.tv_nsec - start_ts.tv_nsec) / 1000) % period;
        load_reduction = (load * advance) / period;
        idle_reduction = advance - load_reduction;

#ifdef ENABLE_VTRACING
        VT_USER_START("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("WD_HIGH", SCOREP_USER_REGION_TYPE_COMMON);
#endif

        sleepret = usleep(load - load_reduction);
        while(sleepret != 0){ /* sometimes usleep fails, this is to be very sure it works */
            sleepret = usleep(load - load_reduction);
        }

#ifdef ENABLE_VTRACING
        VT_USER_END("WD_HIGH");
        VT_USER_START("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("WD_HIGH");
        SCOREP_USER_REGION_BY_NAME_BEGIN("WD_LOW", SCOREP_USER_REGION_TYPE_COMMON);
#endif

        /* signal low load */
        set_load(loadvar, LOAD_LOW);
        
        sleepret = usleep(idle - idle_reduction);
        while(sleepret != 0) { /* sometimes usleep fails, this is to be very sure it works */
            sleepret = usleep(idle - idle_reduction);
        }

#ifdef ENABLE_VTRACING
        VT_USER_END("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("WD_LOW");
#endif

        /* signal high load */
        set_load(loadvar, LOAD_HIGH);
        
        time += period;

        /* exit when termination signal is received or timeout is reached */
        if( (TERMINATE) || ((timeout > 0) && (time / 1000000 >= timeout)) ){
            /* signal that the workers shall shout down */
            set_load(loadvar, LOAD_STOP);
            return 0;
        }
    }
    
    if(timeout > 0){
        sleep(timeout);
        /* signal that the workers shall shout down */
        set_load(loadvar, LOAD_STOP);
    }
    return 0;
}

