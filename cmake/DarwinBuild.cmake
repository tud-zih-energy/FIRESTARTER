if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library( COREFOUNDATION_LIBRARY CoreFoundation )
	find_library( IOKIT_LIBRARY IOKit )
endif()

# Function to link against the correct libraries on darwin
function(target_link_libraries_darwin)
    set(oneValueArgs NAME)
    cmake_parse_arguments(TARGET "" "${oneValueArgs}"
                          "" ${ARGN} )

	if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
		target_link_libraries(${TARGET_NAME}
			${COREFOUNDATION_LIBRARY}
			${IOKIT_LIBRARY}
			)
	endif()
endfunction()
