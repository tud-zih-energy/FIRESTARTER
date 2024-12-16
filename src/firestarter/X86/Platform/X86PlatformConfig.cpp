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

// This file exists to get an entry in the compile commands database. Clangd will interpolate the include directories
// for header files based on the source file with the best matching score. This file should be the best score for the
// included header. Therefore we should not see any errors in this file for missing includes. For more infomation
// look in the LLVM code base: clang/lib/Tooling/InterpolatingCompilationDatabase.cpp

#include "firestarter/X86/Platform/X86PlatformConfig.hpp"