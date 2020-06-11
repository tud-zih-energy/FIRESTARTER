# This will install the LLVM dependency for FIRESTARTER
# Usage:
#   include(cmake/InstallLLVM.cmake)
#   install_llvm()
include(ExternalProject)

	if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
		ExternalProject_Add(LLVMInstall
			PREFIX ${PROJECT_SOURCE_DIR}/lib/LLVM
			DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/lib/LLVM/download
			SOURCE_DIR ${PROJECT_SOURCE_DIR}/lib/LLVM/sources
			URL https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
			URL_HASH SHA256=b25f592a0c00686f03e3b7db68ca6dc87418f681f4ead4df4745a01d9be63843
			CONFIGURE_COMMAND ""
			BUILD_COMMAND ""
			INSTALL_COMMAND ""
		)

		include_directories(${PROJECT_SOURCE_DIR}/lib/LLVM/sources/include)
		add_library(LLVMDemangle STATIC IMPORTED)
		set_target_properties(LLVMDemangle PROPERTIES
			IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/LLVM/sources/lib/libLLVMDemangle.a
		)
		add_library(LLVMSupport STATIC IMPORTED)
		set_target_properties(LLVMSupport PROPERTIES
			INTERFACE_LINK_LIBRARIES "rt;dl;tinfo;-lpthread;m;LLVMDemangle"
	       		IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/LLVM/sources/lib/libLLVMSupport.a
		)

	elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	endif()
