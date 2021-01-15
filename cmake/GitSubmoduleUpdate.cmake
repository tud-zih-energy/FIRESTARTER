# Copyright (c) 2018, Technische Universität Dresden, Germany
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
#    or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This code helps you initialize and update your git submodules
# As such it should be in your repository, not in a submodule - copy paste it there
# Use with:
#    include(cmake/GitSubmoduleUpdate.cmake)
#    git_submodule_update()

macro(_git_submodule_update_path git_path)
    message(STATUS "Checking git submodules in ${git_path}")
    execute_process(
            COMMAND ${GIT_EXECUTABLE} submodule status --recursive
            WORKING_DIRECTORY "${git_path}"
            OUTPUT_VARIABLE GIT_SUBMODULE_STATE
    )
    string(REPLACE "\n" ";" GIT_SUBMODULE_STATE ${GIT_SUBMODULE_STATE})
    foreach(submodule IN LISTS GIT_SUBMODULE_STATE)
        if(submodule MATCHES "[ ]*[+-][^ ]* ([^ ]*).*")
            message("  Found outdated submodule '${CMAKE_MATCH_1}'")
            set(GIT_SUBMODULE_RUN_UPDATE ON)
        endif()
    endforeach()
    if(GIT_SUBMODULE_RUN_UPDATE)
        message(STATUS "Updating git submodules")
        execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY "${git_path}"
        )
    endif()
endmacro()

macro(_is_git git_path result_variable)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --is-inside-work-tree
            WORKING_DIRECTORY ${git_path}
            OUTPUT_VARIABLE OUTPUT_DEV_NULL
            ERROR_VARIABLE OUTPUT_DEV_NULL
            RESULT_VARIABLE NOT_IN_GIT
    )
    if (NOT_IN_GIT)
        set(${result_variable} FALSE)
    else()
        set(${result_variable} TRUE)
    endif()
endmacro()

macro(git_submodule_update)
    find_package(Git)
    if (NOT Git_FOUND)
        message(STATUS "No git executable found. Skipping submodule check.")
    else()
        # If a global check in CMAKE_SOURCE_DIR was already performed recursively, we skip all
        # further checks. If we only do a local check in CMAKE_CURRENT_SOURCE_DIR, we don't set
        # this variable and repeat the checks.
        get_property(GIT_SUBMODULE_CHECKED GLOBAL PROPERTY GIT_SUBMODULE_CHECKED)
        if (NOT GIT_SUBMODULE_CHECKED)
            _is_git(${CMAKE_SOURCE_DIR} IN_GIT)
            if (IN_GIT)
                _git_submodule_update_path("${CMAKE_SOURCE_DIR}")
                set_property(GLOBAL PROPERTY GIT_SUBMODULE_CHECKED TRUE)
            else()
                _is_git(${CMAKE_CURRENT_SOURCE_DIR} IN_CURRENT_GIT)
                if (IN_CURRENT_GIT)
                    _git_submodule_update_path("${CMAKE_CURRENT_SOURCE_DIR}")
                else()
                    message(STATUS "No source directory is a git repository. Skipping submodule check.")
                endif()
            endif()
        endif()
    endif()
endmacro()
