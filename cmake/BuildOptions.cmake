include(CMakeDependentOption)

# Set the different available FIRESTARTER builds.
set(FIRESTARTER_BUILD_TYPE "FIRESTARTER" CACHE STRING "FIRESTARTER_BUILD_TYPE can be any of FIRESTARTER, FIRESTARTER_CUDA, FIRESTARTER_ONEAPI, or FIRESTARTER_HIP.")
set_property(CACHE FIRESTARTER_BUILD_TYPE PROPERTY STRINGS FIRESTARTER FIRESTARTER_CUDA FIRESTARTER_ONEAPI FIRESTARTER_HIP)

# Static linking is not supported with GPU devices or MacOS.
if(${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER" AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
	option(FIRESTARTER_LINK_STATIC "Link FIRESTARTER as a static binary. Note, dlopen is not supported in static binaries. This option is not available on macOS or with CUDA, OneAPI or HIP enabled." ON)
endif()


# We vendor hwloc per default.
option(FIRESTARTER_BUILD_HWLOC "Build hwloc dependency." ON)

# Should we compile the tests
option(FIRESTARTER_BUILD_TESTS "Enable the tests" OFF)

# We fetch google test per default
option(FIRESTARTER_FETCH_GOOGLETEST "Fetch the Google Test dependency." ON)


# Debug feature are enabled on linux per default.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	option(FIRESTARTER_DEBUG_FEATURES "Enable debug features" ON)
endif()


# Feature for different tracing libraries
set(FIRESTARTER_TRACING "None" CACHE STRING "FIRESTARTER_TRACING can be any of None or External.")
set_property(CACHE FIRESTARTER_TRACING PROPERTY STRINGS None External)