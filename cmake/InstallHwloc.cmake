# This will install the hwloc dependency for FIRESTARTER

include(ExternalProject)

if (NOT DEFINED NIX_BUILD)
	if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
		ExternalProject_Add(
			HwlocInstall PREFIX ${PROJECT_SOURCE_DIR}/lib/Hwloc
			DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/download
			SOURCE_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/sources
			INSTALL_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/install
			URL https://download.open-mpi.org/release/hwloc/v2.2/hwloc-2.2.0.tar.gz
			URL_HASH SHA1=17f4d91ccf0cfe694e4000ec2e4e595790393d4c
			CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-static --disable-libudev --disable-shared --disable-doxygen --disable-libxml2 --disable-cairo --disable-io --disable-pci --disable-opencl --disable-cuda --disable-nvml --disable-gl --disable-libudev --disable-plugin-dlopen --disable-plugin-ltdl
			BUILD_IN_SOURCE 1
			BUILD_COMMAND make -j
			INSTALL_COMMAND make install
			)

		SET(HWLOC_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/install")
		SET(HWLOC_LIB_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/install")
	elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	endif()
endif()

include_directories(${HWLOC_INCLUDE_DIR}/include)
add_library(hwloc STATIC IMPORTED)
set_target_properties(hwloc PROPERTIES
	IMPORTED_LOCATION ${HWLOC_LIB_DIR}/lib/libhwloc.a
	)
