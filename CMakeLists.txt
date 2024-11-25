cmake_minimum_required(VERSION 3.22)
project(FIRESTARTER)
enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

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

include(cmake/BuildOptions.cmake)
include(cmake/BuildSettings.cmake)

include(cmake/DarwinBuild.cmake)

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

add_subdirectory(test)