/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

#include "firestarter/Environment/Payload/CompiledPayload.hpp"
#include "firestarter/Environment/Payload/Payload.hpp"

namespace firestarter::environment::payload {

void CompiledPayload::init(double* MemoryAddr, uint64_t BufferSize) { PayloadPtr->init(MemoryAddr, BufferSize); }

void CompiledPayload::lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) {
  PayloadPtr->lowLoadFunction(LoadVar, Period);
};

}; // namespace firestarter::environment::payload