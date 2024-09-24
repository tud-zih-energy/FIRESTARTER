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

#include <asmjit/x86.h>

#include <cstdint>
#include <firestarter/DumpRegisterWorkerData.hpp>
#include <firestarter/Environment/Payload/Payload.hpp>
#include <firestarter/LoadWorkerData.hpp>
#include <firestarter/Logging/Log.hpp>
#include <utility>

#define INIT_BLOCKSIZE 1024

namespace firestarter::environment::x86::payload {

class X86Payload : public environment::payload::Payload {
private:
  // we can use this to check, if our platform support this payload
  asmjit::CpuFeatures const& SupportedFeatures;
  std::list<asmjit::CpuFeatures::X86::Id> FeatureRequests;

protected:
  //  asmjit::CodeHolder code;
  asmjit::JitRuntime Rt;
  // typedef int (*LoadFunction)(firestarter::ThreadData *);
  using LoadFunctionType = uint64_t (*)(uint64_t*, volatile uint64_t*, uint64_t);
  LoadFunctionType LoadFunction = nullptr;

  [[nodiscard]] auto supportedFeatures() const -> asmjit::CpuFeatures const& { return this->SupportedFeatures; }

  template <class IterRegT, class VectorRegT>
  void emitErrorDetectionCode(asmjit::x86::Builder& Cb, IterRegT IterReg, asmjit::x86::Gpq AddrHighReg,
                              asmjit::x86::Gpq PointerReg, asmjit::x86::Gpq TempReg, asmjit::x86::Gpq TempReg2);

public:
  X86Payload(asmjit::CpuFeatures const& SupportedFeatures,
             std::initializer_list<asmjit::CpuFeatures::X86::Id> FeatureRequests, std::string Name,
             unsigned RegisterSize, unsigned RegisterCount)
      : Payload(std::move(Name), RegisterSize, RegisterCount)
      , SupportedFeatures(SupportedFeatures)
      , FeatureRequests(FeatureRequests) {}

  [[nodiscard]] auto isAvailable() const -> bool override {
    bool Available = true;

    for (auto const& Feature : FeatureRequests) {
      Available &= this->SupportedFeatures.has(Feature);
    }

    return Available;
  };

    // A generic implemenation for all x86 payloads
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
  void init(uint64_t* MemoryAddr, uint64_t BufferSize, double FirstValue, double LastValue);
#pragma GCC diagnostic pop
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  // use cpuid and usleep as low load
  void lowLoadFunction(volatile uint64_t* AddrHigh, uint64_t Period) override;

  auto highLoadFunction(uint64_t* AddrMem, volatile uint64_t* AddrHigh, uint64_t Iterations) -> uint64_t override;
};

} // namespace firestarter::environment::x86::payload
