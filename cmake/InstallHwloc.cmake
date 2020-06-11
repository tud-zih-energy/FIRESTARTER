# This will install the hwloc dependency for FIRESTARTER

include(ExternalProject)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	ExternalProject_Add(HwlocInstall
		PREFIX ${PROJECT_SOURCE_DIR}/lib/hwloc
		DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/download
		SOURCE_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/sources
		INSTALL_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/install
		URL https://download.open-mpi.org/release/hwloc/v2.2/hwloc-2.2.0.tar.gz
		URL_HASH SHA1=17f4d91ccf0cfe694e4000ec2e4e595790393d4c
		CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-static --disable-libudev --disable-shared --disable-doxygen
		BUILD_IN_SOURCE 1
		BUILD_COMMAND ${MAKE} -j
		INSTALL_COMMAND ${MAKE} install
		LOG_CONFIGURE 1
		LOG_BUILD 1
	)

	include_directories(${PROJECT_SOURCE_DIR}/lib/Hwloc/install/include)
	add_library(hwloc STATIC IMPORTED)
	set_target_properties(hwloc PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/Hwloc/install/lib/libhwloc.a)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
endif()
