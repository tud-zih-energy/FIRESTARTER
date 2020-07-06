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

#include <signal.h>

#include "firestarter_global.h"
#include "watchdog.h"

#define NSEC_PER_SEC 1000000000

extern unsigned long long LOADVAR;
int TERMINATE = 0;


#define FIRESTARTER_gettime(timer) \
do { \
    clock_gettime(CLOCK_REALTIME, &timer); \
} \
while ( 0 )


/* signal load changes to workers */
static void set_load(unsigned long long *loadvar, unsigned long long value)
{
     *loadvar = value;
     __asm__ __volatile__ ("mfence;");
}

static pthread_t watchdog_thread;

/* exit with zero returncode on sigterm */
static void sigterm_handler()
{
    LOADVAR = LOAD_STOP; // required for the cases load = 100 and load = 0, which do not enter the while loop
    TERMINATE = 1;       // exit while loop used in case of 0 < load < 100

    pthread_kill(watchdog_thread,SIGALRM);
    //exit(EXIT_SUCCESS);
}

void sigalrm_handler()
{
}

#define DO_SLEEP(sleepret,target,remaining) \
do \
{ \
    if ( TERMINATE ) \
    { \
        fprintf(stderr, "Caught shutdown signal, ending now ...\n"); \
        return EINTR; \
    } \
    sleepret = nanosleep(&target,&remaining); \
    while ( sleepret == -1 && errno == EINTR && ! TERMINATE ) \
    { \
        sleepret = nanosleep(&remaining,&remaining); \
    } \
    if ( sleepret == -1 ) \
    { \
        switch (errno) \
        { \
        case EFAULT: \
            fprintf(stderr,"Found a bug in FIRESTARTER, error on copying for nanosleep\n"); \
            break; \
        case EINVAL: \
            fprintf(stderr,"Found a bug in FIRESTARTER, invalid setting for nanosleep\n"); \
            break; \
        case EINTR: \
            fprintf(stderr, "Caught shutdown signal, ending now ...\n"); \
            break; \
        default: \
            fprintf(stderr,"Error calling nanosleep: %d\n",errno); \
            break; \
        } \
        set_load(loadvar, LOAD_STOP); \
        return errno; \
    } \
}\
while (0)

/* coordinates high load and low load phases
 * stops FIRESTARTER when timeout is reached
 * SPECIAL MPI Version
 */
int watchdog_timer(watchdog_arg_t *arg)
{
    sigset_t signal_mask;
    long long timeout, time, period, load, idle, advance, load_reduction, idle_reduction;
    unsigned long long *loadvar;
    int sleepret;
    struct timespec target, remaining;

    /* Definition of timestamps */
    struct timespec start_ts, current;

    sigemptyset(&signal_mask);
    sigaddset(&signal_mask, SIGINT);
    sigaddset(&signal_mask, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signal_mask, NULL);


    watchdog_thread = pthread_self();

    signal(SIGALRM, sigalrm_handler);

    signal(SIGTERM, sigterm_handler);
    signal(SIGINT, sigterm_handler);

    period = arg->period*1000;
    load = arg->load*1000;
    idle = period - load;
    timeout = arg->timeout;
    loadvar = arg->loadvar;

    FIRESTARTER_gettime(start_ts);

    time = 0;
    /* TODO: I don't like that the control flow depends on the period variable,
     * it is not wrong but confusing
     */

    while(period > 0){
        // cycles++;

        FIRESTARTER_gettime(current);

        /* compute advance and align with generally planned period */
             advance = ((current.tv_sec - start_ts.tv_sec) * NSEC_PER_SEC + (current.tv_nsec - start_ts.tv_nsec) ) % period;

        load_reduction = (load * advance) / period;
        idle_reduction = advance - load_reduction;

        /* signal high load */
        set_load(loadvar, LOAD_HIGH);
        target.tv_nsec = (load - load_reduction) % NSEC_PER_SEC;
        target.tv_sec  = (load - load_reduction) / NSEC_PER_SEC;
#ifdef ENABLE_VTRACING
        VT_USER_START("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("WD_HIGH", SCOREP_USER_REGION_TYPE_COMMON);
#endif

        DO_SLEEP(sleepret,target,remaining);

#ifdef ENABLE_VTRACING
        VT_USER_END("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("WD_HIGH");
#endif
       /* wil terminate for SIGINT/SIGTERM, but not due to timeout */
       if ( TERMINATE )
       {
            set_load(loadvar, LOAD_STOP);
            return 0;         
       }

#ifdef ENABLE_VTRACING
        VT_USER_START("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("WD_LOW", SCOREP_USER_REGION_TYPE_COMMON);
#endif

        /* signal low load */
        set_load(loadvar, LOAD_LOW);

        target.tv_nsec = (idle - idle_reduction) % NSEC_PER_SEC;
        target.tv_sec  = (idle - idle_reduction) / NSEC_PER_SEC;

        DO_SLEEP(sleepret,target,remaining);

#ifdef ENABLE_VTRACING
        VT_USER_END("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("WD_LOW");
#endif        
        time += period;

        /* exit when termination signal is received or timeout is reached */
        if( (TERMINATE) || ((timeout > 0) && (time / NSEC_PER_SEC >= timeout)) ){
            /* signal that the workers shall shout down */
            set_load(loadvar, LOAD_STOP);
            return 0;
        }
    }
    
    if(timeout > 0){
        target.tv_nsec = 0;
        target.tv_sec = timeout;

        DO_SLEEP(sleepret,target,remaining);

        /* signal that the workers shall shut down */
        set_load(loadvar, LOAD_STOP);
        return 0;
    }
    return 0;
}

