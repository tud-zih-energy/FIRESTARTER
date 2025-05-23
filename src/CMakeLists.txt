# Create the core firestarter library that is used in all builds and tests
add_library(firestartercore STATIC
	firestarter/Firestarter.cpp
	firestarter/SafeExit.cpp
	firestarter/WatchdogWorker.cpp
	firestarter/DumpRegisterWorker.cpp
	firestarter/CPUTopology.cpp
	firestarter/ThreadAffinity.cpp
	firestarter/FunctionSelection.cpp
	firestarter/ProcessorInformation.cpp

	firestarter/Config/Config.cpp
	firestarter/Config/CpuBind.cpp
	firestarter/Config/InstructionGroups.cpp
	firestarter/Config/MetricName.cpp
	
	firestarter/Measurement/Metric.cpp
	firestarter/Measurement/Summary.cpp

	firestarter/Payload/CompiledPayload.cpp
	firestarter/Payload/PayloadSettings.cpp

	# here starts the x86 specific code
	firestarter/X86/X86ProcessorInformation.cpp

	firestarter/X86/Platform/X86PlatformConfig.cpp

	firestarter/X86/Payload/X86Payload.cpp
	firestarter/X86/Payload/AVX512Payload.cpp
	firestarter/X86/Payload/FMA4Payload.cpp
	firestarter/X86/Payload/FMAPayload.cpp
	firestarter/X86/Payload/ZENFMAPayload.cpp
	firestarter/X86/Payload/AVXPayload.cpp
	firestarter/X86/Payload/SSE2Payload.cpp
	)

target_include_directories(firestartercore PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(firestartercore
	hwloc
	AsmJit::AsmJit
	Nitro::log
	nlohmann_json::nlohmann_json
	)

# Create the linux firestarter library that is used for specific linux only features
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	add_library(firestarterlinux STATIC
		# measurement stuff
		firestarter/Measurement/MeasurementWorker.cpp
		firestarter/Measurement/Metric/IPCEstimate.cpp
		firestarter/Measurement/Metric/RAPL.cpp
		firestarter/Measurement/Metric/Perf.cpp

		# optimization stuff
		firestarter/Optimizer/Population.cpp
		firestarter/Optimizer/OptimizerWorker.cpp
		firestarter/Optimizer/Util/MultiObjective.cpp
		firestarter/Optimizer/Algorithm/NSGA2.cpp
		)
	
	target_include_directories(firestarterlinux PUBLIC ${PROJECT_SOURCE_DIR}/include)

	target_link_libraries(firestarterlinux
		Nitro::log
		nlohmann_json::nlohmann_json
		)
endif()

add_library(firestartercombined STATIC
	# IpcEstimateMetricData::insertValue is accesses which is part of the firestarterlinux library.
	# This reference should be removed there and the file moved back to the firestartercore library.
	firestarter/LoadWorker.cpp
)

target_link_libraries(firestartercombined
	firestartercore
	)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	target_link_libraries(firestartercombined
		firestarterlinux
		)
endif()

SET(FIRESTARTER_FILES
	firestarter/Main.cpp
	)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	SET(FIRESTARTER_LIBRARIES
		firestartercombined
	)
else()
	SET(FIRESTARTER_LIBRARIES
		"-Wl,--whole-archive"
		firestartercombined
		"-Wl,--no-whole-archive"
	)
endif()

if ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_CUDA")
	add_executable(FIRESTARTER_CUDA
		${FIRESTARTER_FILES}
		firestarter/Cuda/Cuda.cpp
		)

	target_link_libraries(FIRESTARTER_CUDA
		${FIRESTARTER_LIBRARIES}
	)

	target_link_libraries(FIRESTARTER_CUDA
		CUDA::cuda_driver
		CUDA::cudart
		CUDA::curand
		CUDA::cublas
		)

	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
		target_link_libraries(FIRESTARTER_CUDA
                    ${CUDAToolkit_LIBRARY_ROOT}/lib/x64/cuda.lib 
			)
	else()
		target_link_libraries(FIRESTARTER_CUDA
			cuda
			-ldl
			)
	endif()

elseif ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_ONEAPI")
	add_executable(FIRESTARTER_ONEAPI
		${FIRESTARTER_FILES}
		firestarter/OneAPI/OneAPI.cpp
		)

	target_link_libraries(FIRESTARTER_ONEAPI
		${FIRESTARTER_LIBRARIES}
	)

	target_link_libraries(FIRESTARTER_ONEAPI
		mkl_sycl
		mkl_intel_ilp64
		mkl_sequential
		mkl_core
		sycl
		stdc++
		Threads::Threads
	)

elseif("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_HIP")
	add_executable(FIRESTARTER_HIP
		${FIRESTARTER_FILES}
		firestarter/Cuda/Cuda.cpp
		)

	target_link_libraries(FIRESTARTER_HIP
		${FIRESTARTER_LIBRARIES}
	)

	target_link_libraries(FIRESTARTER_HIP
		hip::host
		hip::hiprand
		roc::hipblas
		)

	if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
		target_link_libraries(FIRESTARTER_HIP
		hip
		-ldl)
	endif()
	

elseif(${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER")
	add_executable(FIRESTARTER
		${FIRESTARTER_FILES}
		)

	target_link_libraries(FIRESTARTER
		${FIRESTARTER_LIBRARIES}
	)

	target_link_libraries_darwin(NAME FIRESTARTER)

	# static linking is not supported on Darwin, see Apple Technical QA1118
	if((NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin") AND FIRESTARTER_LINK_STATIC)
		target_link_libraries(FIRESTARTER
			-static
			)

		# we are using pthreads
		# static linking with pthreads and std::condition_variabale is evil and will cause segfaults
		# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58909#c1
		if(CMAKE_THREAD_LIBS_INIT MATCHES "^.*pthread.*$")
			target_link_libraries(FIRESTARTER
				"-Wl,--whole-archive -lpthread -Wl,--no-whole-archive"
				)
		endif()
	endif()

	if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT FIRESTARTER_LINK_STATIC)
		target_link_libraries(FIRESTARTER
			-ldl
			)
	endif()

	if (NOT FIRESTARTER_LINK_STATIC)
		target_link_libraries(FIRESTARTER
			Threads::Threads
			)
	endif()
endif()