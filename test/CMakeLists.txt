find_package(Python REQUIRED)

# Function to add reference tests
# Argument NAME: The required name of the executable that will be executed in this test. This name is also used for the ref-file if REFNAME is not supplied.
# Argument REFNAME: The optional name of the ref-file that should be used for this test.
function(add_ref_test)
    set(oneValueArgs NAME REFNAME)
    cmake_parse_arguments(TEST "" "${oneValueArgs}"
                          "" ${ARGN} )

    if(NOT DEFINED TEST_NAME)
        message(FATAL_ERROR "Called add_ref_test without a name.")
    endif()

    if(NOT DEFINED TEST_REFNAME)
        set(TEST_REFNAME ${TEST_NAME})
    endif()

    add_test(NAME ${TEST_REFNAME}
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tooling/ref-test.py $<TARGET_FILE:${TEST_NAME}> ${PROJECT_SOURCE_DIR}/test/refs/${TEST_REFNAME}.log
    )
endfunction()

add_subdirectory(DumpCPUTopology)
add_subdirectory(UnitTests)

# The DumpPayloads and X86Functions test executables reference X86-specific
# payload and processor code, so they are only built on x86 targets.
if(FIRESTARTER_ARCH STREQUAL "x86_64")
    add_subdirectory(DumpPayloads)
    add_subdirectory(X86Functions)
endif()