cmake_minimum_required(VERSION 3.13)
project(FIRESTARTER)

# enable c++11
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")
# enable static linking
#if(DEFINED ${BUILD_STATIC})
#	SET(CMAKE_EXE_LINKER_FLAGS "-static")
#endif()

find_package(LLVM REQUIRED CONFIG)

include_directories(include)

add_subdirectory(src)
