add_executable(FIRESTARTER
	firestarter/Main.cpp
	firestarter/ThreadWorker.cpp
	firestarter/Compat/x86.c

	firestarter/Environment/CpuClockrate.cpp
	firestarter/Environment/CpuAffinity.cpp
	firestarter/Environment/Environment.cpp

	firestarter/Environment/X86/Functions.cpp
	firestarter/Environment/X86/CpuClockrate.cpp

	firestarter/Environment/X86/Payload/X86Payload.cpp
	firestarter/Environment/X86/Payload/AVX512Payload.cpp
	firestarter/Environment/X86/Payload/FMA4Payload.cpp
	firestarter/Environment/X86/Payload/FMAPayload.cpp
	firestarter/Environment/X86/Payload/AVXPayload.cpp
	firestarter/Environment/X86/Payload/SSE2Payload.cpp
)

target_link_libraries(FIRESTARTER
	LLVMSupport
	hwloc
	AsmJit::AsmJit
)
