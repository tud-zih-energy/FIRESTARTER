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

using namespace firestarter::environment::payload;

unsigned
Payload::getSequenceStartCount(const std::vector<std::string> &sequence,
                               const std::string start) {
  unsigned i = 0;

  for (const auto &item : sequence) {
    if (0 == item.rfind(start, 0)) {
      i++;
    }
  }

  return i;
}

std::vector<std::string> Payload::generateSequence(
    std::vector<std::pair<std::string, unsigned>> const &proportions) {
  std::vector<std::pair<std::string, unsigned>> prop = proportions;

  prop.erase(std::remove_if(prop.begin(), prop.end(),
                            [](auto const &pair) { return pair.second == 0; }),
             prop.end());

  std::vector<std::string> sequence = {};

  if (prop.size() == 0) {
    return sequence;
  }

  auto it = prop.begin();
  auto insertIt = sequence.begin();

  sequence.insert(insertIt, it->second, it->first);

  for (++it; it != prop.end(); ++it) {
    for (unsigned i = 0; i < it->second; i++) {
      insertIt = sequence.begin();
      std::advance(insertIt, 1 + floor(i * (sequence.size() + it->second - i) /
                                       (float)it->second));
      sequence.insert(insertIt, it->first);
    }
  }

  return sequence;
}

unsigned Payload::getL2LoopCount(const std::vector<std::string> &sequence,
                                 const unsigned numberOfLines,
                                 const unsigned size, const unsigned threads) {
  if (this->getL2SequenceCount(sequence) == 0) {
    return 0;
  }
  return (0.8 * size / 64 / threads /
          (this->getL2SequenceCount(sequence) *
           this->getNumberOfSequenceRepetitions(sequence,
                                                numberOfLines / threads)));
}

unsigned Payload::getL3LoopCount(const std::vector<std::string> &sequence,
                                 const unsigned numberOfLines,
                                 const unsigned size, const unsigned threads) {
  if (this->getL3SequenceCount(sequence) == 0) {
    return 0;
  }
  return (0.8 * size / 64 / threads /
          (this->getL3SequenceCount(sequence) *
           this->getNumberOfSequenceRepetitions(sequence,
                                                numberOfLines / threads)));
}

unsigned Payload::getRAMLoopCount(const std::vector<std::string> &sequence,
                                  const unsigned numberOfLines,
                                  const unsigned size, const unsigned threads) {
  if (this->getRAMSequenceCount(sequence) == 0) {
    return 0;
  }
  return (1.0 * size / 64 / threads /
          (this->getRAMSequenceCount(sequence) *
           this->getNumberOfSequenceRepetitions(sequence,
                                                numberOfLines / threads)));
}
