find_package(Python REQUIRED)

# Function to add reference tests
function(add_ref_test)
    set(oneValueArgs NAME)
    cmake_parse_arguments(TEST "" "${oneValueArgs}"
                          "" ${ARGN} )

    add_test(NAME ${TEST_NAME}
        COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tooling/ref-test.py $<TARGET_FILE:${TEST_NAME}> ${PROJECT_SOURCE_DIR}/test/refs/${TEST_NAME}.log
    )
endfunction()

add_subdirectory(DumpPayloads)
add_subdirectory(X86Functions)