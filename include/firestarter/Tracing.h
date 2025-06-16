/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2025 TU Dresden, Center for Information Services and High
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

/// This file provides a interface to adopt different tracing targets to FIRESTARTER
#ifdef __cplusplus
extern "C" {
#endif

// Define the tracing funtions as doing nothing (;) if tracing is not enabled via compiler flags.
#ifdef FIRESTARTER_TRACING_DISABLED
#define firestarterTracingInitialize(Argc, Argv) ((void)0)
#define firestarterTracingRegionBegin(RegionName) ((void)0)
#define firestarterTracingRegionEnd(RegionName) ((void)0)
#else
void firestarterTracingInitialize(int Argc, const char** Argv);
void firestarterTracingRegionBegin(const char* RegionName);
void firestarterTracingRegionEnd(const char* RegionName);
#endif

#ifdef __cplusplus
};
#endif