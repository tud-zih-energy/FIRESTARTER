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

#pragma once

#ifdef FIRESTARTER_TRACING
void firestarter_tracing_initialize(int argc, const char **argv);
void firestarter_tracing_region_begin(char const* region_name);
void firestarter_tracing_region_end(char const* region_name);
#else
#define firestarter_tracing_initialize(argc, argv) {}
#define firestarter_tracing_region_begin(region_name) {}
#define firestarter_tracing_region_end(region_name) {}
#endif