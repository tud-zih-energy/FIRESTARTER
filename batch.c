#include "msr_safe.h"
#include<stdio.h>
#include<stdlib.h>
#include<inttypes.h>
#include<stdint.h>
#include<sys/ioctl.h>
#include<fcntl.h>
#include<errno.h>
#define max_ops (10000)

/* FIXME list
 *
 * Errors should print to stderr and use the format
 *  fprintf(stderr, "%s:%d Error message.\n", __FILE__, __LINE__);
 *
 * Clean up indentation.
 *
 * Create a header file.
 *
 * Move this code into msr-safe along with a makefile that create a library and a toy application that uses it.
 *
 * Man pages?
 *
 * Clean up spacing.
 *
 * Remove unneeded commments and commented-out code.
 */

int print_op( struct msr_batch_op *op );
int add_readops_to_batch(struct msr_batch_array *batch, __u16 firstcpu, __u16 lastcpu, __u32 msr){
    int i;
    //make an if statement that nakes sure tha firstcpu < lastcpu
    if(firstcpu > lastcpu){
        //FIXME "lastcpu is less than firstcpu"
	 printf("arg should be in form (first cpu, last cpu | first cpu < last cpu.");
	 exit(-1);
    }



    batch->numops = batch->numops+(lastcpu-firstcpu)+1;
    batch->ops = realloc( batch->ops, sizeof(struct msr_batch_op) * batch->numops );
    for(i = firstcpu; i <= lastcpu; i++){
        batch->ops[i].cpu = i;
        batch->ops[i].isrdmsr = 1;
        batch->ops[i].err = 0;
        batch->ops[i].msr = msr;
        batch->ops[i].msrdata = 0;
        batch->ops[i].wmask = 0;

        // FIXME wrap this is an "#if DEBUG".
    	printf("MSR Add: %" PRIx32 " MSR value: %llu"  " CPU core: %" PRIu16 "\n",
    		batch->ops[i].msr,
        	batch->ops[i].msrdata,
        	batch->ops[i].cpu);
    }

    // FIXME does this code need to be here?
   if(batch->ops[i].err != 0){
	perror("Error");
	printf("Errno: %d \n", batch->ops[i].err);
	exit(-1);
    	}

    return 0;

}



int print_op( struct msr_batch_op *op ){
    printf("cpu: %" PRIu16 "  isrdmsr: %" PRIu16  " err: %" PRId32 "  msraddr: %" PRIx32 "  msrdata: %" PRIu64  "   wmask: %" PRIx64 " \n",
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
    printf("operations in batch " PRIu32 "\n");                 //FIXME Does this line need to be here?
    for(i=0; i < batch->numops; i++){
        print_op(&(batch->ops[i]));

        }
    return 0;
}

// FIXME Should there be an init/finalize batch so run_batch can assume everything is already open?
int run_batch( struct msr_batch_array *batch ){
	int i, fd, rc;
	fd = open("/dev/cpu/msr_batch", O_RDONLY);
	printf("%s::%d fd = %d\n",__FILE__, __LINE__, fd);      // FIXME does this need to be here?

	if(fd == -1){                                           // FIXME are these errors informative?
		perror("error!");                                   // Check for cases:  batch file does not
		exit(-1);                                           //   exist because module wasn't loaded,
	}                                                       //   file exists but user doesn't have 
                                                            //   read perms.
	for(i=0; i < batch->numops; i++){
		print_op(&(batch->ops[i])); //FIXME Wrap in #if DEBUG
	}

	rc = ioctl(fd, X86_IOC_MSR_BATCH, batch);

	for(i=0; i < batch->numops; i++){   //FIXME wrap in #if DEBUG
		print_op(&(batch->ops[i]));
	}

    // FIXME I want to see a useful error message here.  Was the error because the allowedlist
    // was empty?  Because a single MSR wasn't included in the allowedlist?  Because while the
    // msr was included, the allowedlist didn't allow writing to those particular bits?  The
    // allowedlist allowed writing to those bits but the underlying MSR is read-only (aka the
    // allowedlist is wrong; you discovered that error).  Or the MSR just doesn't exist.
	if(rc < 0){
		rc = rc * -1;
		perror("ioctl failed");
		fprintf(stderr, "%s::%d rc=%d\n", __FILE__, __LINE__, rc);
		exit(-1);
	}
	return 0;
}

// FIXME move this into a test harness/toy app to allowing building the above as a library.
int main(){
	struct msr_batch_array my_batch;
	my_batch.numops = 0;
	my_batch.ops = NULL;
/*	struct msr_batch_op op;

	op.cpu 		= 0;
	op.isrdmsr	= 1;
	op.err		= 0;
	op.msr		= 0xE7;	// MPERF
	op.msrdata	= 0;
	op.wmask	= 0;
	add_readops_to_batch( &my_batch, 0, 8, 0xE7);
	run_batch(&my_batch);
	return 0;

}
