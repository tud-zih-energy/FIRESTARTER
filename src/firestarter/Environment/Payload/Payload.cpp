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

std::vector<std::string> Payload::generateSequence(
    const std::vector<std::pair<std::string, unsigned>> proportions) {
  std::vector<std::string> sequence;
  auto proportionsIt = std::begin(proportions);
  auto insertIt = std::begin(sequence);

  sequence.insert(insertIt, proportionsIt->second, proportionsIt->first);

  for (++proportionsIt; proportionsIt != std::end(proportions);
       proportionsIt++) {
    if (proportionsIt->second == 0) {
      continue;
    }
    for (unsigned i = 0; i < proportionsIt->second; i++) {
      insertIt = std::begin(sequence);
      std::advance(insertIt,
                   1 + floor(i * (sequence.size() + proportionsIt->second - i) /
                             (float)proportionsIt->second));
      sequence.insert(insertIt, proportionsIt->first);
    }
  }

  return sequence;
}

unsigned Payload::getL2LoopCount(const std::vector<std::string> sequence,
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

unsigned Payload::getL3LoopCount(const std::vector<std::string> sequence,
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

unsigned Payload::getRAMLoopCount(const std::vector<std::string> sequence,
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
