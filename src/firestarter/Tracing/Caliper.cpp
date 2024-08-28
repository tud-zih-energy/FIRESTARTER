/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2024 TU Dresden, Center for Information Services and High
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

#include <firestarter/Tracing/Tracing.hpp>

#include <adiak.hpp>
#include <caliper/cali.h>

using namespace firestarter::tracing;

Tracing::Tracing(){}
Tracing::~Tracing(){}

inline void Tracing::regionBegin(char const* region_name) {
    CALI_MARK_BEGIN(region_name);
}

inline void Tracing::regionEnd(char const* region_name) {
    CALI_MARK_END(region_name);
}

