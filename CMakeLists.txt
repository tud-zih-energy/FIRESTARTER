cmake_minimum_required(VERSION 3.22)
project(FIRESTARTER)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/GitSubmoduleUpdate.cmake)

# set FIRESTARTER version
find_package(Git)
if(Git_FOUND)
	_is_git(${CMAKE_CURRENT_SOURCE_DIR} IN_GIT)
endif()

SET (_FIRESTARTER_VERSION_TEMPERED OFF)

SET(_FIRESTARTER_VERSION_STRING "default" CACHE STRING "_FIRESTARTER_VERSION_STRING should not be used if you do not know what to do.")
if("${_FIRESTARTER_VERSION_STRING}" STREQUAL "default")
	if (IN_GIT)
		execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --tags --dirty
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			OUTPUT_VARIABLE _FIRESTARTER_VERSION_STRING
			)
		string(STRIP ${_FIRESTARTER_VERSION_STRING} _FIRESTARTER_VERSION_STRING)
	else()
		message(AUTHOR_WARNING "Cannot detect FIRESTARTER version.")
		SET(_FIRESTARTER_VERSION_STRING "unknown")
	endif()
else()
	SET (_FIRESTARTER_VERSION_TEMPERED ON)
endif()
add_compile_definitions(_FIRESTARTER_VERSION_STRING="${_FIRESTARTER_VERSION_STRING}")

SET(_FIRESTARTER_BUILD_YEAR "default" CACHE STRING "_FIRESTARTER_BUILD_YEAR should not be used if you do not know what to do.")
if("${_FIRESTARTER_BUILD_YEAR}" STREQUAL "default")
	if(IN_GIT)
		execute_process(COMMAND ${GIT_EXECUTABLE} show -s --format=%ci HEAD
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			OUTPUT_VARIABLE _FIRESTARTER_BUILD_YEAR
			)
		string(SUBSTRING ${_FIRESTARTER_BUILD_YEAR} 0 4 _FIRESTARTER_BUILD_YEAR)
	else()
		message(AUTHOR_WARNING "Cannot detect FIRESTARTER build year.")
		SET(_FIRESTARTER_BUILD_YEAR "unknown")
	endif()
else()
	SET (_FIRESTARTER_VERSION_TEMPERED ON)
endif()
add_compile_definitions(_FIRESTARTER_BUILD_YEAR="${_FIRESTARTER_BUILD_YEAR}")

if(_FIRESTARTER_VERSION_TEMPERED)
	add_compile_definitions(_FIRESTARTER_VERSION_TEMPERED)
endif()

# init git submodules
include(cmake/GitSubmoduleUpdate.cmake)
git_submodule_update()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
else()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -fdata-sections -ffunction-sections")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
else()
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif()

# enable debug features on linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	option(FIRESTARTER_DEBUG_FEATURES "Enable debug features" ON)
	if (FIRESTARTER_DEBUG_FEATURES)
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_DEBUG_FEATURES")
	endif()
endif()

set(FIRESTARTER_BUILD_TYPE "FIRESTARTER" CACHE STRING "FIRESTARTER_BUILD_TYPE can be any of FIRESTARTER, FIRESTARTER_CUDA, FIRESTARTER_ONEAPI, or FIRESTARTER_HIP.")
set_property(CACHE FIRESTARTER_BUILD_TYPE PROPERTY STRINGS FIRESTARTER FIRESTARTER_CUDA FIRESTARTER_ONEAPI)
if (${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER")
	option(FIRESTARTER_LINK_STATIC "Link FIRESTARTER as a static binary. Note, dlopen is not supported in static binaries. This option is not available on macOS or with CUDA or OneAPI enabled." ON)
endif()
if (${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER")
	option(FIRESTARTER_BUILD_HWLOC "Build hwloc dependency." ON)
elseif(${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER_CUDA")
	option(FIRESTARTER_BUILD_HWLOC "Build hwloc dependency." ON)
elseif(${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER_ONEAPI")
	option(FIRESTARTER_BUILD_HWLOC "Build hwloc dependency." ON)
elseif(${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER_HIP")
	option(FIRESTARTER_BUILD_HWLOC "Build hwloc dependency." ON)
endif()
option(FIRESTARTER_THREAD_AFFINITY "Enable FIRESTARTER to set affinity to hardware threads." ON)

if(NOT DEFINED ASMJIT_STATIC)
	set(ASMJIT_STATIC TRUE)
endif()

add_subdirectory(lib/asmjit)
add_subdirectory(lib/nitro)

include_directories(include)
include_directories(lib/cxxopts/include)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(lib/json)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(cmake/InstallHwloc.cmake)

add_subdirectory(src)
