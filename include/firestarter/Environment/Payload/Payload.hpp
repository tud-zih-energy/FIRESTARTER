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

#pragma once

#include <initializer_list>
#include <list>
#include <string>
#include <vector>

namespace firestarter::environment::payload {

class Payload {
private:
  std::string _name;
  unsigned getSequenceStartCount(const std::vector<std::string> &sequence,
                                 const std::string start);

protected:
  unsigned _flops;
  unsigned _bytes;
  // number of instructions in load loop
  unsigned _instructions;
  // size of used simd registers in bytes
  unsigned _registerSize;
  // number of used simd registers
  unsigned _registerCount;

  std::vector<std::string> generateSequence(
      const std::vector<std::pair<std::string, unsigned>> &proportion);
  unsigned getL2SequenceCount(const std::vector<std::string> &sequence) {
    return getSequenceStartCount(sequence, "L2");
  };
  unsigned getL3SequenceCount(const std::vector<std::string> &sequence) {
    return getSequenceStartCount(sequence, "L3");
  };
  unsigned getRAMSequenceCount(const std::vector<std::string> &sequence) {
    return getSequenceStartCount(sequence, "RAM");
  };

  unsigned
  getNumberOfSequenceRepetitions(const std::vector<std::string> &sequence,
                                 const unsigned numberOfLines) {
    if (sequence.size() == 0) {
      return 0;
    }
    return numberOfLines / sequence.size();
  };

  unsigned getL2LoopCount(const std::vector<std::string> &sequence,
                          const unsigned numberOfLines, const unsigned size,
                          const unsigned threads);
  unsigned getL3LoopCount(const std::vector<std::string> &sequence,
                          const unsigned numberOfLines, const unsigned size,
                          const unsigned threads);
  unsigned getRAMLoopCount(const std::vector<std::string> &sequence,
                           const unsigned numberOfLines, const unsigned size,
                           const unsigned threads);

public:
  Payload(std::string name, unsigned registerSize, unsigned registerCount)
      : _name(name), _registerSize(registerSize),
        _registerCount(registerCount) {}
  virtual ~Payload() {}

  const std::string &name() const { return _name; }
  unsigned flops() const { return _flops; }
  unsigned bytes() const { return _bytes; }
  unsigned instructions() const { return _instructions; }
  unsigned registerSize() const { return _registerSize; }
  unsigned registerCount() const { return _registerCount; }

  virtual bool isAvailable() const = 0;

  virtual void lowLoadFunction(volatile unsigned long long *addrHigh,
                               unsigned long long period) = 0;

  virtual int compilePayload(
      std::vector<std::pair<std::string, unsigned>> const &proportion,
      unsigned instructionCacheSize,
      std::list<unsigned> const &dataCacheBufferSize, unsigned ramBufferSize,
      unsigned thread, unsigned numberOfLines, bool dumpRegisters,
      bool errorDetection) = 0;
  virtual std::list<std::string> getAvailableInstructions() const = 0;
  virtual void init(unsigned long long *memoryAddr,
                    unsigned long long bufferSize) = 0;
  virtual unsigned long long
  highLoadFunction(unsigned long long *addrMem,
                   volatile unsigned long long *addrHigh,
                   unsigned long long iterations) = 0;

  virtual Payload *clone() const = 0;
};

} // namespace firestarter::environment::payload
