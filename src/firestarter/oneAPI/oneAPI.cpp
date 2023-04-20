#include <stdio.h>
#include <stdlib.h>

#include <firestarter/oneAPI/oneAPI.hpp>
#include <firestarter/LoadWorkerData.hpp>
#include <firestarter/Logging/Log.hpp>

#include <omp.h>

#include <mkl.h>
#include <mkl_omp_offload.h>

#include <level_zero/ze_api.h>

#include <algorithm>
#include <atomic>
#include <type_traits>

#include <math.h>

using namespace firestarter::oneapi;

int runLoad(volatile unsigned long long *loadVar, unsigned matrixsize, std::atomic<int> &initCount, std::condition_variable &waitForInitCv, std::mutex &waitForInitCvMutex);
void init_data(float* f1, float* f2, float* f3, int size);
void initACCs(volatile unsigned long long *loadVar, unsigned matrixsize, int gpus, std::condition_variable &cv);
int round_up(int num_to_round, int multiple);

oneAPI::oneAPI(volatile unsigned long long *loadVar, unsigned matrixsize, int gpus){
	printf("Hello Worlf from oneAPI.cpp\n");
	matrixsize = 4000; //Hack to get it running
	printf("This is with icpx and matrixsize is %u\n", matrixsize);
	
	
	std::thread initAccelerators_t(initACCs, loadVar, matrixsize, gpus, std::ref(_waitForInitCv));
	initAccelerators_t.detach();
}

void initACCs(volatile unsigned long long *loadVar, unsigned matrixsize, int gpus, std::condition_variable &cv){
	
	std::condition_variable waitForInitCv;
	std::mutex waitForInitCvMutex;
	
	if(gpus){
		int num_devices = omp_get_num_devices(); // Get number of Accelerators
		
		if(num_devices){
		
			std::atomic<int> initCount = 0;
			std::vector<std::thread> gpuThreads;
			
			// use all GPUs if the user gave no information about use_device
	      		if (gpus < 0) {
				gpus = num_devices;
	      		}

	      		if (gpus > num_devices) {
				firestarter::log::warn()
		    			<< "You requested more oneAPI devices than available. ";
				firestarter::log::warn()
		    			<< "FIRESTARTER will use " << num_devices << " of the requested "
		    			<< gpus << " oneAPI device(s)";
				gpus = num_devices;
	      		}
			//Start a thread for each GPU. Thread allocates necessary memory and starts the load
			{
			std::lock_guard<std::mutex> lk(waitForInitCvMutex);
			for(int i = 0; i < num_devices; i++){
				
				std::thread startLoad_t(runLoad, loadVar, matrixsize, std::ref(initCount), std::ref(waitForInitCv), std::ref(waitForInitCvMutex)); 
				gpuThreads.push_back(std::move(startLoad_t));
			}
			}
			std::unique_lock<std::mutex> lk(waitForInitCvMutex);
        		// wait for all threads to initialize
        		waitForInitCv.wait(lk, [&] { return initCount == gpus; });
        		
        		// notify that init is done
      			cv.notify_all();

      			/* join computation threads */
      			for (auto &t : gpuThreads) {
        			t.join();
      			}
		}
		else{
			firestarter::log::info()
          			<< "    - No oneAPI devices. Just stressing CPU(s). Maybe use "
             			"FIRESTARTER instead of FIRESTARTER_ONEAPI?";
      			cv.notify_all();
		}
	}
	else{
    		firestarter::log::info()
        		<< "--gpus 0 is set. Just stressing CPU(s). Maybe use "
        	   	"FIRESTARTER instead of FIRESTARTER_ONEAPI?";
    		cv.notify_all();
	}
}


int runLoad(volatile unsigned long long *loadVar, unsigned matrixsize, std::atomic<int> &initCount, std::condition_variable &waitForInitCv, std::mutex &waitForInitCvMutex){

	unsigned size=matrixsize;
	//unsigned size = 15000;
	//mem_avail=15000*15000*15000;
	float one=1.0;
	float *f1, *f2, *f3;

	f1=(float*)mkl_malloc(size*size*sizeof(float),64);
	f2=(float*)mkl_malloc(size*size*sizeof(float),64);
	f3=(float*)mkl_malloc(size*size*sizeof(float),64);
	init_data(f1,f2,f3, size);
	
	if (!size || ((size * size * 3 > mem_avail))) {
    		size = round_up((int)(0.8 * sqrt(((mem_avail) / (3)))),1024); // a multiple of 1024 works always well
  	}
  	firestarter::log::trace() << "Set oneAPI matrix size: " << size;

#pragma omp target enter data map(to:f1[0:size],f2[0:size*size],f3[0:size]) device(0)

{
 while(*loadVar != LOAD_STOP){
 #pragma omp target variant dispatch use_device_ptr(f1, f2, f3) device(0)
 {	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, size, size, one, f1, size, f2, size, one, f3, size);
 }
}
}

#pragma omp target exit data map(from:f3[0:size])

	srand(f3[rand()%size]);
	return 0;
}

void init_data(float* f1, float* f2, float* f3, int size) {
	long x, index, max;
  #pragma omp parallel for schedule(static,1) private(x,index,max)
	for(x = 0; x < size; x++) {
		index = x * size;
		max = index + size;
		for(; index < max; index++) {
                        f1[index] = 30.01;
			f2[index] = 0.01;
                        f3[index] = 0.01;
		}
	}
}

int round_up(int num_to_round, int multiple) {
  if (multiple == 0) {
    return num_to_round;
  }

  int remainder = num_to_round % multiple;
  if (remainder == 0) {
    return num_to_round;
  }

  return num_to_round + multiple - remainder;
}
