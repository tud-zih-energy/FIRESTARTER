# This will install the Google Test dependency for FIRESTARTER

# configure build of googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)

# Do not execute the google test executable during build.
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

if (FIRESTARTER_FETCH_GOOGLETEST)
	include(FetchContent)

	# GoogleTest should use the latest commit available
	FetchContent_Declare(
		googletest
		URL https://github.com/google/googletest/archive/d122c0d435a6d305cdd50526127c84a98b77d87b.zip
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE
	)

	FetchContent_MakeAvailable(googletest)
else()
	find_package(GTest REQUIRED)
	message(STATUS "GTEST_INCLUDE_DIR: ${GTEST_INCLUDE_DIR}")
	message(STATUS "GTEST_LIBRARIES: ${GTEST_LIBRARIES}")
endif()