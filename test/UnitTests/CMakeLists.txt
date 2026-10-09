set(UNIT_TEST_FILES
    CpuBindTest.cpp
    FunctionSelectionTest.cpp
    InstructionGroupsTest.cpp
    MetricNameTest.cpp
    MetricsTest.cpp
)

# The X86-specific unit tests are only built on x86 targets.
if(FIRESTARTER_ARCH STREQUAL "x86_64")
    list(APPEND UNIT_TEST_FILES
        X86CpuFeaturesTest.cpp
        X86CpuModelTest.cpp
        X86PayloadTest.cpp
        X86PlatformConfigTest.cpp
    )
endif()

add_executable(UnitTests
    ${UNIT_TEST_FILES}
)

target_link_libraries(UnitTests
    firestartercore
    GTest::gtest_main
    GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(UnitTests DISCOVERY_MODE PRE_TEST)