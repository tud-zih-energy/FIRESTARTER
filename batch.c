#include "./msr_safe.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>
#define max_ops (10000)
/* This is what is included from msr_safe.h.
 * Included here for reference.
 
struct msr_batch_op
{
    __u16 cpu;     // In: CPU to execute {rd/wr}msr instruction
    __u16 isrdmsr; // In: 0=wrmsr, non-zero=rdmsr
    __s32 err;     // Out: set if error occurred with this operation
    __u32 msr;     // In: MSR Address to perform operation
    __u64 msrdata; // In/Out: Input/Result to/from operation
    __u64 wmask;   // Out: Write mask applied to wrmsr
};

struct msr_batch_array
{
    __u32 numops;             // In: # of operations in operations array
    struct msr_batch_op *ops; // In: Array[numops] of operations
};

#define X86_IOC_MSR_BATCH   _IOWR('c', 0xA2, struct msr_batch_array)

*/

// How to staticly allocate memory.
//struct msr_batch_op mybatch[5][10000];

// Determine size of an array:
// sizeof( struct msr_batch_op )

//
// Function declarations (fill these in for each funciton below)
//
int print_op( struct msr_batch_op *op );
//
// Function defintions
//

// Adds a read operation to the batch.  Returns the index of the new
// op or -1 on error.  Duplicate this for write ops.
/*
int add_readop_to_batch(struct msr_batch_array *batch, __u16 cpu, __u32 msr){
    batch->numops = batch->numops+1;  // Adding one more op.
    batch->ops = realloc( batch->ops, sizeof(struct msr_batch_op) * batch->numops );
    batch->ops[numops-1].cpu = cpu;
    batch->ops[numops-1].isrdmsr = 1;
    batch->ops[numops-1].err = 0;
    batch->ops[numops-1].msr = msr;
    batch->ops[numops-1].msrdata = 0;
    batch->ops[numops-1].wmask = 0;

    printf("MSR value: %" PRIx64 "\n" "CPU core: %" PRIu16, 
            batch->ops[numops-1].msrdata,
            batch->ops[numops-1].cpu);

return 0;


}

*/

/*
    first call to add_readop_to_batch using malloc
    malloc returns 0xFFFFFFFFF3847100, the first byte of the 32 bytes that were allocated.
    That value is saved in batch->ops

    second call 
    malloc returns 0xFAAAAFFFF3847100, the first byte of the 32 bytes that were allocated.
    That value is saved in batch->ops
 
    Alternatively
    first call to add_readop_to_batch using realloc
    batch-ops was initialized to NULL by the caller.
    realloc returns 0xFFFFFFFFF3847100, the first byte of the 32 bytes that were allocated.
    That value is saved in batch->ops

    second call 
    realloc called with 64 bytes and pointer 0xFFFFFFFFF3847100
    realloc returns 0xFAAAFFFFF3847100.  It has copied 64 bytes from 0xFFFFFFFFF3847100
      and then freed that memory.  In short, it has appened 32 bytes to the previous 32
      bytes, and if needed, found a new home for that in memory where it will fit.
    That value is saved in batch->ops

*/


// Adds multiple read operations to the batch, each to be run on
// a different cpu.  Returns the index of the first op or -1 on error.
// Duplicate this for write ops.
int add_readops_to_batch(struct msr_batch_array *batch, __u16 firstcpu, __u16 lastcpu, __u32 msr){
    int i;
    //make an if statement that nakes sure tha firstcpu < lastcpu
    //
    if(lastcpu < firstcpu){
	    printf("arg should be in form (firstcpu, lastcpu) | lastcpu < firstcpu");
    	exit(-1);
    }
    batch->numops = batch->numops+(lastcpu-firstcpu)+1;
    batch->ops = realloc( batch->ops, sizeof(struct msr_batch_op) * batch->numops );
    for(i=firstcpu; i < lastcpu; i++){
        batch->ops[i].cpu = i;
        batch->ops[i].isrdmsr = 1;
        batch->ops[i].err = 0;
        batch->ops[i].msr = msr;
        batch->ops[i].msrdata = 0;
        batch->ops[i].wmask = 0;

        printf("MSR value: %llu "  "\n" "CPU core:% " PRIu16, 
            batch->ops[i].msrdata,
            batch->ops[i].cpu);
    }

    return 0;

}



/*
int add_writeop_to_batch(struct msr_batch_array *batch, __u16 cpu, __u32 msr, __u64 writemask){

        batch->numops = batch->numops+1;  // Adding one more op.
        batch->ops = realloc( batch->ops, sizeof(struct msr_batch_op) * batch->numops );
        batch->ops[numops-1].cpu = cpu;
        batch->ops[numops-1].isrdmsr = 0;
        batch->ops[numops-1].err = 0;
        batch->ops[numops-1].msr = msr;
        batch->ops[numops-1].msrdata = 0;
        batch->ops[numops-1].wmask = writemask;

        printf("MSR value: %" PRIx64 "\n" "CPU core: %" PRIu16, 
            batch->ops[numops-1].msrdata,
            batch->ops[numops-1].cpu);
        return 0;
}

int add_writeops_to_batch(struct msr_batch_array *batch, __u16 first_cpu, __u16 last_cpu, __u32 msr, __u64 writemask){
}
*/
// Print an indivdiual op via the msr_batch_op pointer.
// DO THIS FIRST.
// 
int print_op( struct msr_batch_op *op ){
    printf("cpu: %" PRIu16 "  isrdmsr: %" PRIu16  " err: %" PRId32 "  msraddr: %" PRIx32 "  msrdata: %" PRIx64  "   wmask: %" PRIx64 " \n", 
    (uint16_t)op->cpu,
    (uint16_t)op->isrdmsr,
    (int32_t)op->err,
    (uint32_t)op->msr,
    (uint64_t)op->msrdata,
    (uint64_t)op->wmask);

    return 0;
}

// Print a full batch by the msr_batch_array pointer
// DO THIS SECOND.
int print_batch( struct msr_batch_array *batch ){
    //This function prints the number of operations in msr_batch_array.numops
    //and then prints each of the ops contains in msr_batch_array.ops.
   int i;
    printf("numops: %" PRIu32 "\n", (uint32_t)batch->numops);
    printf("operations in batch " PRIu32 "\n");
    for(i=0; i < batch->numops; i++){
        print_op(&(batch->ops[i]));

        }
    return 0;
}

// Print any ops that have errors by pointer.
/*
int print_error_ops( struct msr_batch_array *batch ){
}

// Actually run the batch.
int run_batch( struct msr_batch_array *batch ){
}
*/
int main(){
    struct msr_batch_array my_batch;
    my_batch.numops = 0;
    my_batch.ops = NULL;
    
    //add_readops_to_batch( &my_batch, 0, 0x7e );
    //add_readops_to_batch( &my_batch, 0, 0x8e );
    add_readops_to_batch( &my_batch, 0, 8, 0x611);

    /*
    struct msr_batch_op op[3];

    op[0].cpu=17;
    op[0].isrdmsr=1;
    op[0].err=38;
    op[0].msr=0x611;
    op[0].msrdata=0;
    op[0].wmask=0;

    op[1].cpu=18;
    op[1].isrdmsr=1;
    op[1].err=38;
    op[1].msr=0xe7;
    op[1].msrdata=0;
    op[1].wmask=0;

    op[2].cpu=19;
    op[2].isrdmsr=1;
    op[2].err=38;
    op[2].msr=0xe8;
    op[2].msrdata=0;
    op[2].wmask=0;

    my_batch.numops = 3;
    my_batch.ops = &(op[0]); // Or, colloqually, op.

    print_batch( &my_batch );


.....

    // I'm going to give you a msr_batch_array* called p.
    int the_number_of_ops_in_p_is = p->numops;
    struct msr_batch_op* the_first_op_in_p_is = p->ops;
    struct msr_batch_op* the_second_op_in_p_is = p->ops[1];
    struct msr_batch_op* the_ith_op_in_p_is = p->ops[i];

    print_op( &(op[2]) );


    printf("__u64 %zd\n", sizeof(__u64));
    printf("__u32 %zd\n", sizeof(__u32));
    printf("__u16 %zd\n", sizeof(__u16));
    printf(" sizeof(struct msr_batch_op) = %zd\n", sizeof(struct msr_batch_op) );
    printf(" sizeof(struct msr_batch_op*) = %zd\n", sizeof(struct msr_batch_op*) );
    */
    return 0;

}


/*
 * msr_batch_op many_ops[50];
 *
 * many_ops[13].cpu
 *
 * &(many_ops[13])
 *
 * struct msr_batch_op* many_ops_ptr = (struct msr_batch_op*) malloc( sizeof(struct msr_batch_op) * how_many_ops_I_need) );
 *
 * many_op_ptr[13].msr
 */

