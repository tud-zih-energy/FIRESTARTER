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

#include <algorithm>
#include <cmath>

#include <firestarter/Environment/Payload/Payload.hpp>

namespace firestarter::environment::payload {

void CompiledPayload::init(double* MemoryAddr, uint64_t BufferSize) { PayloadPtr->init(MemoryAddr, BufferSize); }

void CompiledPayload::lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) {
  PayloadPtr->lowLoadFunction(LoadVar, Period);
};

auto Payload::getSequenceStartCount(const std::vector<std::string>& Sequence, const std::string& Start) -> unsigned {
  unsigned I = 0;

  for (const auto& Item : Sequence) {
    if (0 == Item.rfind(Start, 0)) {
      I++;
    }
  }

  return I;
}

auto Payload::generateSequence(std::vector<std::pair<std::string, unsigned>> const& Proportions)
    -> std::vector<std::string> {
  std::vector<std::pair<std::string, unsigned>> Prop = Proportions;

  Prop.erase(std::remove_if(Prop.begin(), Prop.end(), [](auto const& Pair) { return Pair.second == 0; }), Prop.end());

  std::vector<std::string> Sequence = {};

  if (Prop.empty()) {
    return Sequence;
  }

  auto It = Prop.begin();
  auto InsertIt = Sequence.begin();

  Sequence.insert(InsertIt, It->second, It->first);

  for (++It; It != Prop.end(); ++It) {
    for (unsigned I = 0; I < It->second; I++) {
      InsertIt = Sequence.begin();
      std::advance(InsertIt, 1 + std::floor(I * (Sequence.size() + It->second - I) / static_cast<float>(It->second)));
      Sequence.insert(InsertIt, It->first);
    }
  }

  return Sequence;
}

auto Payload::getL2LoopCount(const std::vector<std::string>& Sequence, const unsigned NumberOfLines,
                             const unsigned Size, const unsigned Threads) -> unsigned {
  if (getL2SequenceCount(Sequence) == 0) {
    return 0;
  }
  return static_cast<unsigned>(
      (0.8 * Size / 64 / Threads /
       (getL2SequenceCount(Sequence) * getNumberOfSequenceRepetitions(Sequence, NumberOfLines / Threads))));
}

auto Payload::getL3LoopCount(const std::vector<std::string>& Sequence, const unsigned NumberOfLines,
                             const unsigned Size, const unsigned Threads) -> unsigned {
  if (getL3SequenceCount(Sequence) == 0) {
    return 0;
  }
  return static_cast<unsigned>(
      (0.8 * Size / 64 / Threads /
       (getL3SequenceCount(Sequence) * getNumberOfSequenceRepetitions(Sequence, NumberOfLines / Threads))));
}

auto Payload::getRAMLoopCount(const std::vector<std::string>& Sequence, const unsigned NumberOfLines,
                              const unsigned Size, const unsigned Threads) -> unsigned {
  if (getRAMSequenceCount(Sequence) == 0) {
    return 0;
  }
  return static_cast<unsigned>(
      (1.0 * Size / 64 / Threads /
       (getRAMSequenceCount(Sequence) * getNumberOfSequenceRepetitions(Sequence, NumberOfLines / Threads))));
}

}; // namespace firestarter::environment::payload