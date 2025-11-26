cmake_minimum_required(VERSION 3.22)

project(FIRESTARTER_ScoreP)

include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/scorep_cmake/src")
include(scorep_cmake/src/ScorePUtilities.cmake)

# Include the firestarter options
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/BuildOptions.cmake)

# Force set tracing type to external
set(FIRESTARTER_TRACING "External")
set(FIRESTARTER_LINK_STATIC OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/BuildSettings.cmake)

# Build the string that will be passed as CMake arguments for the firestarter build
get_cmake_property(all_cmake_variables VARIABLES)
string (REGEX MATCHALL "FIRESTARTER_[A-Za-z0-9_]*" firestarter_cmake_variables "${all_cmake_variables}")

foreach (var IN LISTS firestarter_cmake_variables)
    message("${var}=${${var}}")
    list(APPEND FIRESTARTER_CMAKE_ARGUMENTS "-D${var}=${${var}}")
endforeach()

message("Passing the following arguments to the FIRESTARTER build: ${FIRESTARTER_CMAKE_ARGUMENTS}")

# Define the library that contains the instrumentation
find_package(ScoreP REQUIRED COMPONENTS PTHREAD)

add_library(firestarter_scorep SHARED
	ScoreP.c
  ${CMAKE_CURRENT_BINARY_DIR}/ScorePAdapter.c
)
target_include_directories(firestarter_scorep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

get_target_property(scorep ScoreP::ScoreP IMPORTED_LOCATION)

scorep_instrument("firestarter_scorep" LANGS C ARGUMENTS "--nocompiler" "--nokokkos" "--user" "--thread=pthread")

# Create the ScoreP adapter
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ScorePAdapter.c
  COMMAND ${SCOREP_CONFIG_EXECUTABLE} --adapter-init --user --nokokkos --nocompiler --thread=pthread > ${CMAKE_CURRENT_BINARY_DIR}/ScorePAdapter.c
)

list(APPEND FIRESTARTER_CMAKE_ARGUMENTS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")
list(APPEND FIRESTARTER_CMAKE_ARGUMENTS "-DCMAKE_CXX_STANDARD_LIBRARIES=-L${CMAKE_CURRENT_BINARY_DIR} -lfirestarter_scorep")

# Define the build the FIRESTARTER executable
ExternalProject_Add(FIRESTARTER
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/FIRESTARTER
  CMAKE_ARGS "${FIRESTARTER_CMAKE_ARGUMENTS}"
)

add_dependencies(FIRESTARTER firestarter_scorep)