# Dependent Linux features
if(FIRESTARTER_LINK_STATIC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_LINK_STATIC")
endif()

if (FIRESTARTER_DEBUG_FEATURES)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_DEBUG_FEATURES")
endif()

if (FIRESTARTER_THREAD_AFFINITY)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_THREAD_AFFINITY")
endif()


# Not MSVC
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -fdata-sections -ffunction-sections")
endif()


# Darwin
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
endif()


# Not (Darwin or MSVC)
# equivalent to Linux and Windows with mingw
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC"))
	SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif()


# Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	# enable position independant code on linux
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()


# Find packages, set the compiler and compile flags specific to the selected FIRESTARTER build.
if(${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER")
	# No specific compiler selected
elseif ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_CUDA")
	find_package(CUDAToolkit REQUIRED)
	include_directories(${CUDAToolkit_INCLUDE_DIRS})

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_BUILD_CUDA")
elseif ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_ONEAPI")
	find_program(ICX_PATH icx REQUIRED)

	message(STATUS "Path of icx executable is: ${ICX_PATH}")

	SET(CMAKE_CXX_COMPILER ${ICX_PATH})
	SET(CMAKE_C_COMPILER ${ICX_PATH})

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -DFIRESTARTER_BUILD_ONEAPI")
elseif("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_HIP")
	if (NOT DEFINED ROCM_PATH )
		set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
	endif ()

	# Search for rocm in common locations
	list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH}/lib ${ROCM_PATH})
	find_package(HIP REQUIRED)
	find_package(rocblas REQUIRED)
	find_package(rocrand REQUIRED)
	find_package(hiprand REQUIRED)
	find_package(hipblas REQUIRED)

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_BUILD_HIP")
endif()