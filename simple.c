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
#include <errno.h>
#include <stdlib.h>
#include "msr_safe.h" 
#define NUM_CPUS (16)
#define OPS_PER_CPU (2)
#define OPS_PER_PKG (1)

struct msr_batch_op clear_array[NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG];
struct msr_batch_array clear_batch;
struct msr_batch_op read_array[NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG]; //reads the msr 0xe8
struct msr_batch_array read_batch;

int fd, rc;
// Make a copy of this file.  Create a "print_op(struct msr_batch_op *)" function that prints out the op fields
// given a pointer to the op.
//
// Then create a print_my_batch(struct msr_batch_array *) function that prints out all of
// the ops in that batch.  It should call print_op() for each individual op.
//
// Then rewrite the list_first_batch_error() function below to use print_op().
int print_op(struct msr_batch_array *clear_array){
    int i;
    for(i=9; i < clear_array->numops; i++){
        printf("OP: %d, cpu core: %" PRIu16 ", isrdmsr: " %PRId32 "err: %" PRId32 ", msr: 0x%" PRIx32 ", msrdata: 0x" PRIx64", wmask: 0x%" PRIx64 "\n"
            i,
            clear_array->ops[i]

            clear
int list_first_batch_error( struct msr_batch_array *a ){
    // a->ops is equivalent to (*a).ops
    //
    int i;
    for( i=0; i<a->numops; i++ ){
        if (a->ops[i].err){
            fprintf(stdout, "op=%d, cpu=%" PRIu16 ", isrdmsr=%" PRId32 ", err=%" PRId32 ", msr=0x%" PRIx32 ", msrdata=0x%" PRIx64 ", wmask=0x%" PRIx64 "\n",
                    i,
                    a->ops[i].cpu,
                    a->ops[i].isrdmsr,
                    a->ops[i].err,
                    a->ops[i].msr,
                    (uint64_t)(a->ops[i].msrdata),
                    (uint64_t)(a->ops[i].wmask));
            return i;
        }
    }
    return 0;
}

int main(){
    int arr_idx, cpu_idx;

    fd = open("/dev/cpu/msr_batch", O_RDWR);
    if( fd==-1 ){
        fprintf(stderr, "%s::%d Error opening /dev/cpu/msr_batch.\n", __FILE__, __LINE__);
        perror("Bye!");
        exit(-1);
    }

    clear_batch.numops = NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG;
    clear_batch.ops = &clear_array[0];

    clear_array[0].cpu = 0;
    clear_array[0].isrdmsr = 1;     // This register is read-only!
    clear_array[0].err = 0;
    clear_array[0].msrdata = 0;
    clear_array[0].msr = 0x611;   // energy counter
    clear_array[0].wmask = 0xFFFFFFFF;

    for(cpu_idx = 0, arr_idx = 1; cpu_idx < NUM_CPUS; cpu_idx++){
        clear_array[arr_idx].cpu = cpu_idx;
        clear_array[arr_idx].isrdmsr = 0;
        clear_array[arr_idx].err = 0;
        clear_array[arr_idx].msrdata = 0;
        clear_array[arr_idx].msr = 0xe8;   // APERF

        clear_array[++arr_idx].cpu = cpu_idx;
        clear_array[arr_idx].isrdmsr = 0;
        clear_array[arr_idx].err = 0;
        clear_array[arr_idx].msrdata = 0;
        clear_array[arr_idx].msr = 0xe7;   // MPERF

        ++arr_idx;
    }

    read_batch.numops = NUM_CPUS * OPS_PER_CPU + OPS_PER_PKG;
    read_batch.ops = &read_array[0];

    read_array[0].cpu = 0;
    read_array[0].isrdmsr = 1;
    read_array[0].err = 0;
    read_array[0].msrdata = 0;
    read_array[0].msr = 0x611;   // energy counter

    for(cpu_idx=0, arr_idx=1; cpu_idx < NUM_CPUS; cpu_idx++){
        read_array[arr_idx].cpu = cpu_idx;
        read_array[arr_idx].isrdmsr = 1;
        read_array[arr_idx].err = 0;
        read_array[arr_idx].msrdata = 0;
        read_array[arr_idx].msr = 0xe8;   // APERF

        read_array[++arr_idx].cpu = cpu_idx;
        read_array[arr_idx].isrdmsr = 1;
        read_array[arr_idx].err = 0;
        read_array[arr_idx].msrdata = 0;
        read_array[arr_idx].msr = 0xe7;   // MPERF
        arr_idx++;
    }

    errno = ioctl(fd, X86_IOC_MSR_BATCH, &clear_batch);
    if (errno < 0 ){
        errno = errno * -1;
        perror("ioctl failed");
        fprintf(stderr, "%s::%d errno=%d\n", __FILE__, __LINE__, errno);
        if( list_first_batch_error(&clear_batch) == 0){
            fprintf(stderr, "No ops reported an error code\n");
        }
        exit(-1);
    }

    errno = ioctl(fd, X86_IOC_MSR_BATCH, &read_batch);
    if (errno < 0 ){
        errno = errno * -1;
        perror("ioctl failed");
        fprintf(stderr, "%s::%d errno=%d\n", __FILE__, __LINE__, errno);
        if( list_first_batch_error(&read_batch) == 0){
            fprintf(stderr, "No ops reported an error code\n");
        }
        exit(-1);
    }


    printf("QQQ %llu ", read_array[0].msrdata);
    for( arr_idx=OPS_PER_PKG; arr_idx < OPS_PER_CPU * NUM_CPUS + OPS_PER_PKG; arr_idx+=OPS_PER_CPU ){
        printf( "%llu %llu ",
               read_array[arr_idx].msrdata,
               read_array[arr_idx+1].msrdata);
    }

    printf("\n");
    return 0;
}

