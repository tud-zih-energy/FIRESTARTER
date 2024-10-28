SET(FIRESTARTER_FILES
	firestarter/Config.cpp
	firestarter/Main.cpp
	firestarter/Firestarter.cpp
	firestarter/LoadWorker.cpp
	firestarter/SafeExit.cpp
	firestarter/WatchdogWorker.cpp
	firestarter/DumpRegisterWorker.cpp
	
	firestarter/Environment/X86/Platform/X86PlatformConfig.cpp

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

SET(FIRESTARTER_LINK_LIBRARIES
	)

if (FIRESTARTER_THREAD_AFFINITY)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_THREAD_AFFINITY")
endif()

if ("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_CUDA")
	find_package(CUDAToolkit REQUIRED)
	include_directories(${CUDAToolkit_INCLUDE_DIRS})

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_BUILD_CUDA")

	add_executable(FIRESTARTER_CUDA
		${FIRESTARTER_FILES}
		firestarter/Cuda/Cuda.cpp
		)
	target_compile_features(FIRESTARTER_CUDA PRIVATE cxx_std_17)

	if(FIRESTARTER_BUILD_HWLOC)
		add_dependencies(FIRESTARTER_CUDA
			HwlocInstall
			)
	endif()

	target_link_libraries(FIRESTARTER_CUDA
		hwloc
		AsmJit::AsmJit
		Nitro::log
		nlohmann_json::nlohmann_json
		Threads::Threads
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

	find_program(ICX_PRESENT icx)

	if(ICX_PRESENT)
		message(STATUS "Executable found: ${ICX_PRESENT}")
	else()
		message(FATAL_ERROR "OneAPI Intel Compiler icx not found")
	endif()
	SET(CMAKE_CXX_COMPILER "icx")
	SET(CMAKE_C_COMPILER "icx")

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -DFIRESTARTER_BUILD_ONEAPI")

	add_executable(FIRESTARTER_ONEAPI
		${FIRESTARTER_FILES}
		firestarter/OneAPI/OneAPI.cpp
		)
	target_compile_features(FIRESTARTER_ONEAPI PRIVATE cxx_std_17)

	if(FIRESTARTER_BUILD_HWLOC)
		add_dependencies(FIRESTARTER_ONEAPI
			HwlocInstall
			)
	endif()

	target_link_libraries(FIRESTARTER_ONEAPI
		hwloc
		AsmJit::AsmJit
		Nitro::log
		nlohmann_json::nlohmann_json
		Threads::Threads
		)
	target_link_libraries(FIRESTARTER_ONEAPI
		mkl_sycl
		mkl_intel_ilp64
		mkl_sequential
		mkl_core
		sycl
		stdc++
	)

elseif("${FIRESTARTER_BUILD_TYPE}" STREQUAL "FIRESTARTER_HIP")
	if (NOT DEFINED ROCM_PATH )
		set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
	endif ()
	# Search for rocm in common locations
	list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH}/lib ${ROCM_PATH})
	find_package(HIP REQUIRED)
	find_package(rocblas REQUIRED)
	find_package(rocrand REQUIRED)
	find_package(hiprand REQUIRED)
	find_package(hipblas REQUIRED)

	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIRESTARTER_BUILD_HIP")

	set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
	set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
	set( CMAKE_CXX_STANDARD 17 )

	add_executable(FIRESTARTER_HIP
		${FIRESTARTER_FILES}
		firestarter/Cuda/Cuda.cpp
		)
	target_compile_features(FIRESTARTER_HIP PRIVATE cxx_std_17)

	if(FIRESTARTER_BUILD_HWLOC)
		add_dependencies(FIRESTARTER_HIP
			HwlocInstall
			)
	endif()

	target_link_libraries(FIRESTARTER_HIP
		hwloc
		AsmJit::AsmJit
		Nitro::log
		nlohmann_json::nlohmann_json
		Threads::Threads
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
	target_compile_features(FIRESTARTER PRIVATE cxx_std_17)

	# static linking is not supported on Darwin, see Apple Technical QA1118
	if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
		find_library( COREFOUNDATION_LIBRARY CoreFoundation )
		find_library( IOKIT_LIBRARY IOKit )
		target_link_libraries(FIRESTARTER
			Threads::Threads
			${COREFOUNDATION_LIBRARY}
			${IOKIT_LIBRARY}
			)
	else()
		if (FIRESTARTER_LINK_STATIC)
			SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -DFIRESTARTER_LINK_STATIC")
			# we are using pthreads
			# static linking with pthreads and std::condition_variabale is evil and will cause segfaults
			# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58909#c1
			if(CMAKE_THREAD_LIBS_INIT MATCHES "^.*pthread.*$")
				target_link_libraries(FIRESTARTER
					"-Wl,--whole-archive -lpthread -Wl,--no-whole-archive"
					)
			endif()
		else()
			if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
				target_link_libraries(FIRESTARTER
					-ldl
					)
			endif()

			target_link_libraries(FIRESTARTER
				Threads::Threads
				)
		endif()
	endif()

	if(FIRESTARTER_BUILD_HWLOC)
		add_dependencies(FIRESTARTER
			HwlocInstall
			)
	endif()

	target_link_libraries(FIRESTARTER
		hwloc
		AsmJit::AsmJit
		Nitro::log
		nlohmann_json::nlohmann_json
		)
endif()
