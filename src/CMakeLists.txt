SET(FIRESTARTER_FILES
	firestarter/Main.cpp
	firestarter/Firestarter.cpp
	firestarter/LoadWorker.cpp
	firestarter/WatchdogWorker.cpp
	firestarter/DumpRegisterWorker.cpp
    )
    SET(PAYLOAD_FILES
firestarter/Environment/Environment.cpp
	    firestarter/Environment/CPUTopology.cpp
        firestarter/Environment/Payload/Payload.cpp)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86|i[3-6]86)$" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    SET(PAYLOAD_FILES
        ${PAYLOAD_FILES}
	firestarter/Environment/X86/X86Environment.cpp
	firestarter/Environment/X86/X86CPUTopology.cpp
	firestarter/Environment/X86/Payload/X86Payload.cpp
	firestarter/Environment/X86/Payload/AVX512Payload.cpp
	firestarter/Environment/X86/Payload/FMA4Payload.cpp
	firestarter/Environment/X86/Payload/FMAPayload.cpp
	firestarter/Environment/X86/Payload/ZENFMAPayload.cpp
	firestarter/Environment/X86/Payload/AVXPayload.cpp
	firestarter/Environment/X86/Payload/SSE2Payload.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    # here starts the AArch64 specific code
    SET(PAYLOAD_FILES
        ${PAYLOAD_FILES}
	firestarter/Environment/AArch64/AArch64Environment.cpp
	firestarter/Environment/AArch64/AArch64CPUTopology.cpp
	firestarter/Environment/AArch64/Payload/AArch64Payload.cpp
	firestarter/Environment/AArch64/Payload/AArch64DefaultPayload.cpp
	firestarter/Environment/AArch64/Payload/AArch64NEONFMAPayload.cpp
)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(hwloc REQUIRED IMPORTED_TARGET hwloc)

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

if(PROJECT_IS_TOP_LEVEL)
    add_executable(FIRESTARTER
        ${FIRESTARTER_FILES} ${PAYLOAD_FILES}
        )
endif()

add_library(payload ${PAYLOAD_FILES})

target_include_directories(payload PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(payload PUBLIC ../include)
target_link_libraries(payload PUBLIC
    PkgConfig::hwloc
    AsmJit::AsmJit
    Nitro::log
    nlohmann_json::nlohmann_json
    )

target_compile_features(payload PRIVATE cxx_std_17)
add_library(Firestarter::payload ALIAS payload)

if(PROJECT_IS_TOP_LEVEL)

target_link_libraries(FIRESTARTER
    PkgConfig::hwloc
    AsmJit::AsmJit
    Nitro::log
    nlohmann_json::nlohmann_json
    )
endif()
