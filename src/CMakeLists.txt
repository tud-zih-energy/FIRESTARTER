add_executable(firestarter firestarter/main.cpp firestarter/firestarter.cpp firestarter/cpuClockrate.cpp firestarter/x86.c)

target_link_libraries(firestarter LLVMSupport)
target_link_libraries(firestarter hwloc)
