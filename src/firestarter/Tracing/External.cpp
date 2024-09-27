/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/\>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/


extern "C" {
    #include <firestarter/Tracing/FIRESTARTER_External_Tracing.h>
}


#include <firestarter/Tracing/Tracing.hpp>

void firestarter::tracing::initialize(int argc, const char **argv){
    firestarter_tracing_initialize(argc, argv);
}

void firestarter::tracing::regionBegin(char const* region_name) {
    firestarter_tracing_region_begin(region_name);
}

void firestarter::tracing::regionEnd(char const* region_name) {
    firestarter_tracing_region_begin(region_name);
}