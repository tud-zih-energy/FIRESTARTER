#!/usr/bin/env bash

export CUDA_ROOT=/usr/local/cuda-${CUDA}

case $CUDA in
	6.5)
		travis_retry wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
		sudo sh cuda_6.5.14_linux_64.run -extract=${CUDA_ROOT}
		sudo sh ${CUDA_ROOT}/cuda-linux64-rel-6.5.14-18749181.run --tar mxvf -C ${CUDA_ROOT}
		;;
	8.0)
		travis_retry wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
		travis_retry wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run
		sudo sh cuda_8.0.61_375.26_linux-run --extract=${CUDA_ROOT}
		sudo sh ${CUDA_ROOT}/cuda-linux64-rel-8.0.61-21551265.run --tar mxvf -C ${CUDA_ROOT}
		sudo sh cuda_8.0.61.2_linux-run --accept-eula --silent
		;;
	11.0)
		travis_retry wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
		sudo sh cuda_11.0.3_450.51.06_linux.run --toolkit --toolkitpath=${CUDA_ROOT} --override --silent
		;;
esac

export CPATH=${CUDA_ROOT}/include:${CPATH}
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib64/stubs:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${CUDA_ROOT}/lib64:${CUDA_ROOT}/lib64/stubs:${LIBRARY_PATH}
export PATH=${CUDA_ROOT}:${PATH}
export CMAKE_PREFIX_PATH=${CUDA_ROOT}:${CMAKE_PREFIX_PATH}
export CUDA_HOME=${CUDA_ROOT}
export CUDA_PATH=${CUDA_ROOT}
