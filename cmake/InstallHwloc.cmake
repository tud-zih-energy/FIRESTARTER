# This will install the hwloc dependency for FIRESTARTER

include(ExternalProject)

if (FIRESTARTER_BUILD_HWLOC)
	add_library(hwloc STATIC IMPORTED)

	if (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
		ExternalProject_Add(
			HwlocInstall PREFIX ${PROJECT_SOURCE_DIR}/lib/Hwloc
			DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/download
			SOURCE_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/sources
			INSTALL_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/install
			URL https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.0.tar.gz
			URL_HASH SHA1=a5c2dad233609b1a1a7f2e905426b68bde725c70
			CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-static --disable-libudev --disable-shared --disable-doxygen --disable-libxml2 --disable-cairo --disable-io --disable-pci --disable-opencl --disable-cuda --disable-nvml --disable-gl --disable-libudev --disable-plugin-dlopen --disable-plugin-ltdl
			BUILD_IN_SOURCE 1
			BUILD_COMMAND make -j
			INSTALL_COMMAND make install
			)

		SET(HWLOC_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/install")
		SET(HWLOC_LIB_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/install")
		set_target_properties(hwloc PROPERTIES
			IMPORTED_LOCATION ${HWLOC_LIB_DIR}/lib/libhwloc.a
		)
	elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
		if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
			ExternalProject_Add(
				HwlocInstall PREFIX ${PROJECT_SOURCE_DIR}/lib/Hwloc
				DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/download
				SOURCE_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/sources
				INSTALL_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/install
				URL https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.0.tar.gz
				URL_HASH SHA1=a5c2dad233609b1a1a7f2e905426b68bde725c70
				CONFIGURE_COMMAND ""
				BUILD_COMMAND cd <SOURCE_DIR>\\contrib\\windows && MSBuild /p:Configuration=Release /p:Platform=x64 hwloc.sln
				INSTALL_COMMAND ""
				)

			SET(HWLOC_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/sources")
			SET(HWLOC_LIB_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/sources/contrib/windows/x64/Release/")
			set_target_properties(hwloc PROPERTIES
				IMPORTED_LOCATION ${HWLOC_LIB_DIR}/libhwloc.lib
			)
		# mingw
		else()
			ExternalProject_Add(
				HwlocInstall PREFIX ${PROJECT_SOURCE_DIR}/lib/Hwloc
				DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/download
				SOURCE_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/sources
				INSTALL_DIR ${PROJECT_SOURCE_DIR}/lib/Hwloc/install
				URL https://download.open-mpi.org/release/hwloc/v2.7/hwloc-win64-build-2.7.0.zip
				URL_HASH SHA256=c3845b298e64fd9acf62aa2eaa2d1beced0a3a4f9a0678dc6fd0b880fd0c23d4
				CONFIGURE_COMMAND ""
				BUILD_COMMAND ""
				INSTALL_COMMAND ""
				)

			SET(HWLOC_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/sources")
			SET(HWLOC_LIB_DIR "${PROJECT_SOURCE_DIR}/lib/Hwloc/sources")
			set_target_properties(hwloc PROPERTIES
				IMPORTED_LOCATION ${HWLOC_LIB_DIR}/lib/libhwloc.a
			)
		endif()
	endif()

	include_directories(${HWLOC_INCLUDE_DIR}/include)
endif()
