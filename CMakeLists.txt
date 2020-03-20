cmake_minimum_required(VERSION 3.13)
project(FIRESTARTER)

# enable static linking
#if(DEFINED ${BUILD_STATIC})
#	SET(CMAKE_EXE_LINKER_FLAGS "-static")
#endif()

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

# enable c++11
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra")

find_package(LLVM REQUIRED CONFIG)

include_directories(include)

include(lib/nitro/Nitro.cmake)

add_subdirectory(src)
