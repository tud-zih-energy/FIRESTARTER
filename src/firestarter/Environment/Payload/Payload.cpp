#include <cmath>

#include <firestarter/Environment/Payload/Payload.hpp>

using namespace firestarter::environment::payload;

unsigned Payload::getSequenceStartCount(const std::vector<std::string> sequence,
                                        const std::string start) {
  unsigned i = 0;

  for (const auto &item : sequence) {
    if (0 == item.rfind(start, 0)) {
      i++;
    }
  }

  return i;
}

std::vector<std::string>
Payload::generateSequence(const std::map<std::string, unsigned> proportions) {
  std::vector<std::string> sequence;
  auto it = std::begin(proportions);

  for (int i = 0; i < it->second; i++) {
    sequence.push_back(it->first);
  }

  for (++it; it != std::end(proportions); ++it) {
    if (it->second == 0) {
      continue;
    }
    for (int i = 0; i < it->second; i++) {
      auto insertIt = std::begin(sequence);
      std::advance(insertIt, 1 + i * floor((sequence.size() + it->second - i) /
                                           it->second));
      sequence.insert(insertIt, it->first);
    }
  }

  return sequence;
}

unsigned Payload::getL2LoopCount(const std::vector<std::string> sequence,
                                 const unsigned numberOfLines,
                                 const unsigned size) {
  if (this->getL2SequenceCount(sequence) == 0) {
    return 0;
  }
  return 0.8 * (size / 64 / this->getL2SequenceCount(sequence) /
                this->getNumberOfSequenceRepetitions(sequence, numberOfLines));
}

unsigned Payload::getL3LoopCount(const std::vector<std::string> sequence,
                                 const unsigned numberOfLines,
                                 const unsigned size) {
  if (this->getL3SequenceCount(sequence) == 0) {
    return 0;
  }
  return 0.8 * (size / 64 / this->getL3SequenceCount(sequence) /
                this->getNumberOfSequenceRepetitions(sequence, numberOfLines));
}

unsigned Payload::getRAMLoopCount(const std::vector<std::string> sequence,
                                  const unsigned numberOfLines,
                                  const unsigned size) {
  if (this->getRAMSequenceCount(sequence) == 0) {
    return 0;
  }
  return 1.0 * (size / 64 / this->getRAMSequenceCount(sequence) /
                this->getNumberOfSequenceRepetitions(sequence, numberOfLines));
}
