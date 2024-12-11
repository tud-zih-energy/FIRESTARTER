add_executable(UnitTests
    CpuBindTest.cpp
)

target_link_libraries(UnitTests
    firestartercore
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(UnitTests)