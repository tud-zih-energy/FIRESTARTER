add_executable(UnitTests
    CpuBindTest.cpp
    FunctionSelectionTest.cpp
    InstructionGroupsTest.cpp
    MetricNameTest.cpp
    MetricsTest.cpp
    X86CpuFeaturesTest.cpp
    X86CpuModelTest.cpp
    X86PayloadTest.cpp
    X86PlatformConfigTest.cpp
)

target_link_libraries(UnitTests
    firestartercore
    GTest::gtest_main
    GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(UnitTests DISCOVERY_MODE PRE_TEST)