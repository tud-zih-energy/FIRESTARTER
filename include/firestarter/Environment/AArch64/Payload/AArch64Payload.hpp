/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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

#include <firestarter/Environment/Payload/Payload.hpp>
#include <firestarter/Logging/Log.hpp>

#include <firestarter/DumpRegisterWorkerData.hpp>
#include <firestarter/LoadWorkerData.hpp>

#include <asmjit/a64.h>

#define INIT_BLOCKSIZE 1024

namespace firestarter::environment::aarch64::payload {

class AArch64Payload : public environment::payload::Payload {
private:
  // we can use this to check, if our platform support this payload
  asmjit::CpuFeatures const &_supportedFeatures;
  std::list<asmjit::CpuFeatures::ARM::Id> featureRequests;

protected:
  //  asmjit::CodeHolder code;
  asmjit::JitRuntime rt;
  // typedef int (*LoadFunction)(firestarter::ThreadData *);
  typedef unsigned long long (*LoadFunction)(unsigned long long *,
                                             volatile unsigned long long *,
                                             unsigned long long);
  LoadFunction loadFunction = nullptr;

  asmjit::CpuFeatures const &supportedFeatures() const {
    return this->_supportedFeatures;
  }

  template <class IterReg, class VectorReg>
  void emitErrorDetectionCode(asmjit::a64::Builder &cb, IterReg iter_reg,
                              asmjit::a64::GpX addrHigh_reg,
                              asmjit::a64::GpX pointer_reg,
                              asmjit::a64::GpX temp_reg,
                              asmjit::a64::GpX temp_reg2) {}

public:
  AArch64Payload(asmjit::CpuFeatures const &supportedFeatures,
             std::initializer_list<asmjit::CpuFeatures::ARM::Id> featureRequests,
             std::string name, unsigned registerSize, unsigned registerCount)
      : Payload(name, registerSize, registerCount),
        _supportedFeatures(supportedFeatures),
        featureRequests(featureRequests) {}

  bool isAvailable() const override {
    bool available = true;

    for (auto const &feature : featureRequests) {
      available &= this->_supportedFeatures.has(feature);
    }

    return available;
  };

    // A generic implemenation for all aarch64 payloads
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  void init(unsigned long long *memoryAddr, unsigned long long bufferSize,
            double firstValue, double lastValue);
#pragma GCC diagnostic pop
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  // use cpuid and usleep as low load
  void lowLoadFunction(volatile unsigned long long *addrHigh,
                       unsigned long long period) override;

  unsigned long long highLoadFunction(unsigned long long *addrMem,
                                      volatile unsigned long long *addrHigh,
                                      unsigned long long iterations) override;
};

} // namespace firestarter::environment::aarch64::payload
