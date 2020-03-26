add_executable(firestarter firestarter/main.cpp firestarter/firestarter.cpp firestarter/cpuClockrate.cpp firestarter/x86.c firestarter/threadWorker.cpp)

target_link_libraries(firestarter
	LLVMSupport
	hwloc
)
