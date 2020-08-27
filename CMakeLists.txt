cmake_minimum_required(VERSION 3.13)
project(FIRESTARTER)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

# set FIRESTARTER version
find_package(Git)
if(Git_FOUND)
	execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --tags --dirty
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		OUTPUT_VARIABLE _FIRESTARTER_VERSION_STRING
		)
	string(STRIP ${_FIRESTARTER_VERSION_STRING} _FIRESTARTER_VERSION_STRING)
else()
	message(AUTHOR_WARNING "Cannot detect FIRESTARTER version.")
	set(_FIRESTARTER_VERSION_STRING "unknown")
endif()
add_compile_definitions(_FIRESTARTER_VERSION_STRING="${_FIRESTARTER_VERSION_STRING}")

string(TIMESTAMP _FIRESTARTER_BUILD_YEAR "%Y")
add_compile_definitions(_FIRESTARTER_BUILD_YEAR="${_FIRESTARTER_BUILD_YEAR}")

# init git submodules
include(cmake/GitSubmoduleUpdate.cmake)
git_submodule_update()

# enable c++17
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -fdata-sections -ffunction-sections")

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -Wl,--gc-sections -Wl,--strip-all")

if(NOT DEFINED ASMJIT_STATIC)
	set(ASMJIT_STATIC TRUE)
endif()

include("lib/asmjit/CMakeLists.txt")

include_directories(include)
include_directories(lib/nitro/include)
include_directories(lib/cxxopts/include)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(cmake/InstallLLVM.cmake)
include(cmake/InstallHwloc.cmake)

#include(FindPkgConfig)
#pkg_check_modules(HWLOC REQUIRED hwloc>=2.0)

add_subdirectory(src)
