add_executable(DumpPayloads Main.cpp)
target_link_libraries(DumpPayloads firestartercore)

add_ref_test(NAME DumpPayloads)