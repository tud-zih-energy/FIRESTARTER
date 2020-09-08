SET(FIRESTARTER_FILES
	firestarter/Main.cpp
	firestarter/ThreadWorker.cpp
	firestarter/WatchdogWorker.cpp

	firestarter/Environment/CpuClockrate.cpp
	firestarter/Environment/CpuAffinity.cpp
	firestarter/Environment/Environment.cpp
	firestarter/Environment/Payload/Payload.cpp

	# here starts the x86 specific code
	firestarter/Environment/X86/Functions.cpp
	firestarter/Environment/X86/CpuParameter.cpp
	firestarter/Environment/X86/Timestamp.cpp

	firestarter/Environment/X86/Payload/X86Payload.cpp
	firestarter/Environment/X86/Payload/AVX512Payload.cpp
	firestarter/Environment/X86/Payload/FMA4Payload.cpp
	firestarter/Environment/X86/Payload/FMAPayload.cpp
	firestarter/Environment/X86/Payload/ZENFMAPayload.cpp
	firestarter/Environment/X86/Payload/AVXPayload.cpp
	firestarter/Environment/X86/Payload/SSE2Payload.cpp
	)

if(DEFINED BUILD_CUDA)
	add_executable(FIRESTARTER_CUDA
		${FIRESTARTER_FILES}
		firestarter/Cuda/Cuda.cpp
		)
else()
	add_executable(FIRESTARTER
		${FIRESTARTER_FILES}
		)
endif()

SET(FIRESTARTER_LINK_LIBRARIES
	Threads::Threads
	hwloc
	AsmJit::AsmJit
	)

if(DEFINED BUILD_CUDA)
	find_package(CUDA)

	target_link_libraries(FIRESTARTER_CUDA
		${FIRESTARTER_LINK_LIBRARIES}
		${CUDA_LIBRARIES}
		${CUDA_CUBLAS_LIBRARIES}
		)
else()
	target_link_libraries(FIRESTARTER
		${FIRESTARTER_LINK_LIBRARIES}
		)
endif()

if(NOT DEFINED NIX_BUILD)
	SET(FIRESTARTER_NOT_NIX_BUILD_DEPENDENCIES
		HwlocInstall
		)

	if(DEFINED BUILD_CUDA)
		add_dependencies(FIRESTARTER_CUDA
			${FIRESTARTER_NOT_NIX_BUILD_DEPENDENCIES}
			)
	else()
		add_dependencies(FIRESTARTER
			${FIRESTARTER_NOT_NIX_BUILD_DEPENDENCIES}
			)
	endif()
endif()
