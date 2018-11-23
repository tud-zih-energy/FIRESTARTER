#!/bin/bash

travis_retry wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_${CUDA}_amd64.deb
travis_retry sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
travis_retry sudo dpkg -i cuda-repo-ubuntu1404_${CUDA}_amd64.deb
travis_retry sudo apt-get update -qq

export CUDA_VERSION=$(expr ${CUDA} : '\([0-9]*\.[0-9]*\)')
export CUDA_VERSION_APT=${CUDA_VERSION/./-}

travis_retry sudo apt-get install -y cuda-drivers cuda-core-${CUDA_VERSION_APT} cuda-cudart-dev-${CUDA_VERSION_APT} cuda-cublas-dev-${CUDA_VERSION_APT}
travis_retry sudo apt-get clean

export CUDA_ROOT=/usr/local/cuda-${CUDA_VERSION}
export LD_LIBRARY_PATH=${CUDA_HOME}/nvvm/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_ROOT}/bin:${PATH}

sudo mkdir -p /opt/cuda/
sudo ln -s /usr/local/cuda-${CUDA_VERSION}/include /opt/cuda/
sudo ln -s /usr/local/cuda-${CUDA_VERSION}/lib64 /opt/cuda/lib64
sudo ln -s /usr/local/cuda-${CUDA_VERSION}/lib64 /opt/cuda/lib