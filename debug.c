#include <fcntl.h>
#include <assert.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <inttypes.h>
#include <sys/ioctl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "msr_safe.h" 
#define NUM_CPUS (16)
#define OPS_PER_CPU (2)
#define OPS_PER_PKG (1)
struct msr_batch_op clear_array[NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG];
struct msr_batch_array clear_batch;
struct msr_batch_op read_array[NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG]; //reads the msr 0xe8
struct msr_batch_array read_batch;

int fd, rc;

void zeroOut(){
    int array_index, cpu_count;
    clear_batch.numops = NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG;
    clear_batch.ops = &clear_array[0];

    clear_array[0].cpu = 0;
    clear_array[0].isrdmsr = 0;
    clear_array[0].err = 0;
    clear_array[0].msrdata = 0;
    clear_array[0].msr = 0x611;   // energy counter

    for(cpu_count = 0, array_index = 1; cpu_count < NUM_CPUS; cpu_count++){
        clear_array[array_index].cpu = cpu_count;
        clear_array[array_index].isrdmsr = 0;
        clear_array[array_index].err = 0;
        clear_array[array_index].msrdata = 0;
        clear_array[array_index].msr = 0xe8;   // APERF

        clear_array[++array_index].cpu = cpu_count;
        clear_array[array_index].isrdmsr = 0;
        clear_array[array_index].err = 0;
        clear_array[array_index].msrdata = 0;
        clear_array[array_index].msr = 0xe7;   // MPERF

        ++array_index;
    }

    if (ioctl(fd, X86_IOC_MSR_BATCH, &clear_batch) < 0 ){
        printf("ioctl failed\n");
        printf("%s::%d Got here....\n", __FILE__, __LINE__);
    }
}

int cpu_index, arr_index;

void timer_handler(int signum){
    int arr_idx;

    if (ioctl(fd, X86_IOC_MSR_BATCH, &read_batch) < 0 ){
       // printf("ERROR! rc = %i, msr addr = %x, error = %i\n", rc, read_array[arr_index].msr, read_array[arr_index].err);
        printf("ioctl failed\n");
    }

    // Modify this so you are printing out QQQ and then the data specified below.
    
    printf("QQQ %llu ", read_array[0].msrdata);
    for( arr_idx=OPS_PER_PKG; arr_idx < OPS_PER_CPU * NUM_CPUS + OPS_PER_PKG; arr_idx+=OPS_PER_CPU ){
        printf( "%llu %llu ",
               read_array[arr_idx].msrdata,
               read_array[arr_idx+1].msrdata);
    }
    printf("\n");
}
//main is just setup_timer the compiler whined at me because there was no main.
int main(){
    struct sigaction sa;
    struct itimerval timer;
    int cpu_idx;
    read_batch.numops =  NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG;
    read_batch.ops = &read_array[0];
    zeroOut();
    fd = open("/dev/cpu/msr_batch", O_RDONLY);
    //fprintf(stdout, "QQQ CPU_INDEX APERF0 ENERGY MPERF0 \n");
    // Programming problem:  print a header of the form
    // APERFxx JOULESxx MPERFxx
    // For xx values 00 through NUM_CPUS.
    
    printf( "QQQ ENERGY " );   // tag for grepping results.
    for( cpu_idx=0; cpu_idx < NUM_CPUS; cpu_idx++ ){
        printf( "APERF%02d MPERF%02d ", cpu_idx, cpu_idx );
    }
    printf( "\n" );


   // printf("%s::%d Got here....\n", __FILE__, __LINE__);
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = &timer_handler;
    sigaction(SIGVTALRM, &sa, NULL);

    read_array[0].cpu = 0;
    read_array[0].isrdmsr = 1;
    read_array[0].err = 0;
    read_array[0].msrdata = 0;
    read_array[0].msr = 0x611;   // energy counter

    for(cpu_index=0, arr_index=1; cpu_index < NUM_CPUS; cpu_index++){
        read_array[arr_index].cpu = cpu_index;
        read_array[arr_index].isrdmsr = 1;
        read_array[arr_index].err = 0;
        read_array[arr_index].msrdata = 0;
        read_array[arr_index].msr = 0xe8;   // APERF

        read_array[++arr_index].cpu = cpu_index;
        read_array[arr_index].isrdmsr = 1;
        read_array[arr_index].err = 0;
        read_array[arr_index].msrdata = 0;
        read_array[arr_index].msr = 0xe7;   // MPERF
        arr_index++;
    }

   // printf("%s::%d Got here....\n", __FILE__, __LINE__);
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = &timer_handler;
    sigaction(SIGVTALRM, &sa, NULL);

    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = 100000;
    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = 100000;

    setitimer(ITIMER_VIRTUAL, &timer, NULL);
    return 0;
}

