SET(FIRESTARTER_LIBRARY_FILES
	firestarter/Firestarter.cpp
	firestarter/SafeExit.cpp
	firestarter/WatchdogWorker.cpp
	firestarter/DumpRegisterWorker.cpp
	firestarter/CPUTopology.cpp
	firestarter/ThreadAffinity.cpp
	firestarter/FunctionSelection.cpp
	firestarter/ProcessorInformation.cpp
	firestarter/LoadWorker.cpp

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

# the selected tracing variant
if(FIRESTARTER_TRACING STREQUAL "None")
    message(STATUS "FIRESTARTER_TRACING is set to 'None', no tracing can be used with the resulting binary.")
else()
    list(APPEND FIRESTARTER_LIBRARY_FILES
        firestarter/Tracing/${FIRESTARTER_TRACING}.cpp
    )
endif()

# Specific linux only features
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	list(APPEND FIRESTARTER_LIBRARY_FILES
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
endif()

# Create library that is used in all builds and tests
add_library(firestartercore STATIC
	${FIRESTARTER_LIBRARY_FILES}
)

target_include_directories(firestartercore PUBLIC
	${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(firestartercore
	hwloc
	AsmJit::AsmJit
	Nitro::log
	nlohmann_json::nlohmann_json
)

# Main executable files
SET(FIRESTARTER_FILES
	firestarter/Main.cpp
)

# Files used in GPU versions
if ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_CUDA")
	list(APPEND FIRESTARTER_FILES
		firestarter/Cuda/Cuda.cpp
	)

elseif ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_ONEAPI")
	list(APPEND FIRESTARTER_FILES
		firestarter/OneAPI/OneAPI.cpp
	)
elseif("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_HIP")
	list(APPEND FIRESTARTER_FILES
		firestarter/Cuda/Cuda.cpp
	)
endif()

# Main executable
add_executable(${FIRESTARTER_BUILD_TYPE}
	${FIRESTARTER_FILES}
)

target_link_libraries(${FIRESTARTER_BUILD_TYPE}
	firestartercore
)

if ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_CUDA")
	target_link_libraries(${FIRESTARTER_BUILD_TYPE}
		CUDA::cuda_driver
		CUDA::cudart
		CUDA::curand
		CUDA::cublas
	)

	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
		target_link_libraries(${FIRESTARTER_BUILD_TYPE}
            ${CUDAToolkit_LIBRARY_ROOT}/lib/x64/cuda.lib 
		)
	else()
		target_link_libraries(${FIRESTARTER_BUILD_TYPE}
			-ldl
		)
	endif()
elseif ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_ONEAPI")
	target_link_libraries(${FIRESTARTER_BUILD_TYPE}
		mkl_sycl
		mkl_intel_ilp64
		mkl_sequential
		mkl_core
		sycl
		stdc++
		Threads::Threads
	)
elseif("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_HIP")
	target_link_libraries(${FIRESTARTER_BUILD_TYPE}
		hip::host
		hip::hiprand
		roc::hipblas
	)

	if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
		target_link_libraries(${FIRESTARTER_BUILD_TYPE}
			hip
			-ldl
		)
	endif()
elseif(${FIRESTARTER_BUILD_TYPE} STREQUAL "FIRESTARTER")
	target_link_libraries_darwin(NAME ${FIRESTARTER_BUILD_TYPE})

	# static linking is not supported on Darwin, see Apple Technical QA1118
	if((NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin") AND FIRESTARTER_LINK_STATIC)
		target_link_libraries(${FIRESTARTER_BUILD_TYPE}
			-static
			)

		# we are using pthreads
		# static linking with pthreads and std::condition_variabale is evil and will cause segfaults
		# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58909#c1
		if(CMAKE_THREAD_LIBS_INIT MATCHES "^.*pthread.*$")
			target_link_libraries(${FIRESTARTER_BUILD_TYPE}
				"-Wl,--whole-archive -lpthread -Wl,--no-whole-archive"
				)
		endif()
	endif()

	if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT FIRESTARTER_LINK_STATIC)
		target_link_libraries(${FIRESTARTER_BUILD_TYPE}
			-ldl
			)
	endif()

	if (NOT FIRESTARTER_LINK_STATIC)
		target_link_libraries(${FIRESTARTER_BUILD_TYPE}
			Threads::Threads
			)
	endif()
endif()