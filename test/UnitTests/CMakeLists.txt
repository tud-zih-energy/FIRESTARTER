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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(PerfMetricTest PerfMetricTest.cpp
        ${PROJECT_SOURCE_DIR}/src/firestarter/Measurement/Metric/Perf.cpp)
    target_link_options(PerfMetricTest PRIVATE
        -Wl,--wrap=access -Wl,--wrap=syscall -Wl,--wrap=ioctl -Wl,--wrap=read -Wl,--wrap=close)
    add_test(NAME PerfMetricTest COMMAND PerfMetricTest)
endif()