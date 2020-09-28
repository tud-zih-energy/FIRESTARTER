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

# static linking is not supported, see Apple Technical QA1118
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
else()
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--strip-all")
endif()

# enable debug features on linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG_FEATURES")
endif()

if(NOT DEFINED ASMJIT_STATIC)
	set(ASMJIT_STATIC TRUE)
endif()

include("lib/asmjit/CMakeLists.txt")

include_directories(include)
include_directories(lib/nitro/include)
include_directories(lib/cxxopts/include)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(cmake/InstallHwloc.cmake)

add_subdirectory(src)
