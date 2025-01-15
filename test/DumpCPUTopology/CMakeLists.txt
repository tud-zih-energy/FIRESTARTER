add_executable(DumpCPUTopology Main.cpp)
target_link_libraries(DumpCPUTopology firestartercore)
target_link_libraries_darwin(NAME DumpCPUTopology)

file(GLOB TEST_FILES inputs/*.xml)

foreach(testfile ${TEST_FILES})
    get_filename_component(testname ${testfile} NAME_WLE)

    add_ref_test(NAME DumpCPUTopology
                 REFNAME DumpCPUTopology/${testname})

    # Set the correct topology xml file for the ref test
    set_property(TEST DumpCPUTopology/${testname} PROPERTY ENVIRONMENT "HWLOC_XMLFILE=${testfile}")
endforeach()