add_executable(X86Functions Main.cpp)
target_link_libraries(X86Functions firestartercore)

add_ref_test(NAME X86Functions)