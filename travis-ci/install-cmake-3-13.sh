#! /usr/bin/env bash

mkdir -p ${TRAVIS_BUILD_DIR}/deps
cd ${TRAVIS_BUILD_DIR}/deps

travis_retry wget https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz
echo "1c6612f3c6dd62959ceaa96c4b64ba7785132de0b9cbc719eea6fe1365cc8d94  cmake-3.13.0-Linux-x86_64.tar.gz" > SHA-256.txt
sha256sum -c SHA-256.txt

export CMAKE_ROOT=${TRAVIS_BUILD_DIR}/deps/cmake
export PATH=${CMAKE_ROOT}:${CMAKE_ROOT}/bin:$PATH

tar -xf cmake-3.13.0-Linux-x86_64.tar.gz
mv cmake-3.13.0-Linux-x86_64 ${CMAKE_ROOT}

cd ${TRAVIS_BUILD_DIR}
