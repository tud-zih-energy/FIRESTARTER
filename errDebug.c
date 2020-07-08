#include "msr_safe.h"
#include<stdio.h>
#include<stdlib.h>
#include<inttypes.h>
#include<stdint.h>


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

    if(batch->ops[i].err !=0){
	    perror("Error: ");
	    exit(-1);
    }
        printf("MSR: %" PRIx32  " MSR value: %llu"  " CPU core: %" PRIu16 "\n",
	    batch->ops[i].msr,
            batch->ops[i].msrdata,
            batch->ops[i].cpu);
    }
    
    return 0;

}





int main(){
    struct msr_batch_array my_batch;
    my_batch.numops = 0;
    my_batch.ops = NULL;
    
    //add_readops_to_batch( &my_batch, 0, 0x7e );
    //add_readops_to_batch( &my_batch, 0, 0x8e );
    add_readops_to_batch( &my_batch, 0, 8, 0x611);

 }
