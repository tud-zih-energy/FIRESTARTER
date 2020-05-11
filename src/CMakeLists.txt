add_executable(FIRESTARTER
	firestarter/Main.cpp
	firestarter/Firestarter.cpp
	firestarter/CpuClockrate.cpp
	firestarter/ThreadWorker.cpp
	firestarter/Compat/x86.c
)

target_link_libraries(FIRESTARTER
	LLVMSupport
	hwloc
)
