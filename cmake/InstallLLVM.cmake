# This will install the LLVM dependency for FIRESTARTER

include(ExternalProject)

if (NOT DEFINED LLVM_SOURCE_DIR)
	if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
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

		SET(LLVM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/LLVM/sources")
	elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
	endif()
endif()

include_directories(${LLVM_SOURCE_DIR}/include)
add_library(LLVMDemangle STATIC IMPORTED)
set_target_properties(LLVMDemangle PROPERTIES
	IMPORTED_LOCATION ${LLVM_SOURCE_DIR}/lib/libLLVMDemangle.a)
add_library(LLVMSupport STATIC IMPORTED)
set_target_properties(LLVMSupport PROPERTIES
	INTERFACE_LINK_LIBRARIES "rt;dl;tinfo;-lpthread;m;LLVMDemangle"
	IMPORTED_LOCATION ${LLVM_SOURCE_DIR}/lib/libLLVMSupport.a
	)
