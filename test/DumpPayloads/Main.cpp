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

#include "firestarter/Payload/Payload.hpp"
#include "firestarter/X86/Payload/AVX512Payload.hpp"
#include "firestarter/X86/Payload/AVXPayload.hpp"
#include "firestarter/X86/Payload/FMA4Payload.hpp"
#include "firestarter/X86/Payload/FMAPayload.hpp"
#include "firestarter/X86/Payload/SSE2Payload.hpp"
#include "firestarter/X86/Payload/ZENFMAPayload.hpp"

namespace {

/// Take a list of instructions and return a list with a pair containing the each instruction in the first element of
/// the pair and a one in the second.
auto oneEach(const std::list<std::string>& Instructions) -> firestarter::InstructionGroups {
  std::vector<std::pair<std::string, unsigned>> OneEach;
  for (const auto& Instruction : Instructions) {
    OneEach.emplace_back(Instruction, 1);
  }
  return firestarter::InstructionGroups{OneEach};
}

/// Dump the generated assembler code of the payload with some given settings. Each item is printed once.
void dumpPayload(firestarter::payload::Payload& PayloadPtr) {
  const auto& Instuctions = PayloadPtr.getAvailableInstructions();

  firestarter::payload::PayloadSettings Settings(/*Threads=*/{1},
                                                 /*DataCacheBufferSize=*/{32768, 1048576, 1441792},
                                                 /*RamBufferSize=*/1048576000,
                                                 /*Lines=*/3 * Instuctions.size(),
                                                 /*Groups=*/oneEach(Instuctions));

  (void)PayloadPtr.compilePayload(Settings, /*DumpRegisters=*/false, /*ErrorDetection=*/false,
                                  /*PrintAssembler=*/true,
                                  firestarter::payload::HighLoadControlFlowDescription::kStopOnLoadThreadWorkType);
}

} // namespace

auto main(int /*argc*/, const char** /*argv*/) -> int {
  const std::vector<std::shared_ptr<firestarter::payload::Payload>> PayloadPtrs = {
      std::make_unique<firestarter::x86::payload::AVX512Payload>(),
      std::make_unique<firestarter::x86::payload::FMAPayload>(),
      std::make_unique<firestarter::x86::payload::ZENFMAPayload>(),
      std::make_unique<firestarter::x86::payload::FMA4Payload>(),
      std::make_unique<firestarter::x86::payload::AVXPayload>(),
      std::make_unique<firestarter::x86::payload::SSE2Payload>()};

  for (const auto& PayloadPtr : PayloadPtrs) {
    firestarter::log::info() << "Payload " << PayloadPtr->name();
    dumpPayload(*PayloadPtr);
  }

  return EXIT_SUCCESS;
}