add_executable(UnitTests
    CpuBindTest.cpp
    InstructionGroupsTest.cpp
    X86CpuFeaturesTest.cpp
    X86PayloadTest.cpp
)

target_link_libraries(UnitTests
    firestartercore
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(UnitTests)