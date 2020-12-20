SET(FIRESTARTER_FILES
	firestarter/Main.cpp
	firestarter/Firestarter.cpp
	firestarter/LoadWorker.cpp
	firestarter/WatchdogWorker.cpp
	firestarter/DumpRegisterWorker.cpp

	firestarter/Environment/Environment.cpp
	firestarter/Environment/CPUTopology.cpp
	firestarter/Environment/Payload/Payload.cpp

	# here starts the x86 specific code
	firestarter/Environment/X86/X86Environment.cpp
	firestarter/Environment/X86/X86CPUTopology.cpp

	firestarter/Environment/X86/Payload/X86Payload.cpp
	firestarter/Environment/X86/Payload/AVX512Payload.cpp
	firestarter/Environment/X86/Payload/FMA4Payload.cpp
	firestarter/Environment/X86/Payload/FMAPayload.cpp
	firestarter/Environment/X86/Payload/ZENFMAPayload.cpp
	firestarter/Environment/X86/Payload/AVXPayload.cpp
	firestarter/Environment/X86/Payload/SSE2Payload.cpp
	)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	SET(FIRESTARTER_FILES 
		${FIRESTARTER_FILES}

		# measurement stuff
		firestarter/Measurement/MeasurementWorker.cpp
		firestarter/Measurement/Summary.cpp
		firestarter/Measurement/Metric/RAPL.cpp
		firestarter/Measurement/Metric/Perf.cpp

		# optimization stuff
		firestarter/Optimizer/Population.cpp
		)
endif()

SET(FIRESTARTER_LINK_LIBRARIES
	Threads::Threads
	hwloc
	AsmJit::AsmJit
	Nitro::log
	nlohmann_json::nlohmann_json
	)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT FIRESTARTER_LINK_STATIC)
	SET(FIRESTARTER_LINK_LIBRARIES
		${FIRESTARTER_LINK_LIBRARIES}
		-ldl
		)
endif()

if (FIRESTARTER_THREAD_AFFINITY)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_THREAD_AFFINITY")
endif()

if (FIRESTARTER_CUDA)
	find_package(CUDA REQUIRED)

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_BUILD_CUDA")

	add_executable(FIRESTARTER_CUDA
		${FIRESTARTER_FILES}
		firestarter/Cuda/Cuda.cpp
		)

	target_link_libraries(FIRESTARTER_CUDA
		${FIRESTARTER_LINK_LIBRARIES}
		cuda
		cudart
		cublas
		)
else()
	# static linking is not supported on Darwin, see Apple Technical QA1118
	if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	else()
		if (FIRESTARTER_LINK_STATIC)
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -DFIRESTARTER_LINK_STATIC")
		endif()
	endif()

	add_executable(FIRESTARTER
		${FIRESTARTER_FILES}
		)

	target_link_libraries(FIRESTARTER
		${FIRESTARTER_LINK_LIBRARIES}
		)
endif()

if (FIRESTARTER_BUILD_HWLOC)
	if (FIRESTARTER_CUDA)
		add_dependencies(FIRESTARTER_CUDA
			HwlocInstall
			)
	else()
		add_dependencies(FIRESTARTER
			HwlocInstall
			)
	endif()
endif()
