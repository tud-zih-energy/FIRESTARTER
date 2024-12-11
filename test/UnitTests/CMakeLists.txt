add_executable(UnitTests
    CpuBindTest.cpp
    InstructionGroupsTest.cpp
)

target_link_libraries(UnitTests
    firestartercore
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(UnitTests)