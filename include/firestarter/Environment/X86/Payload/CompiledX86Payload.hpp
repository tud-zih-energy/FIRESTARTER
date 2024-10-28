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

#pragma once

#include "asmjit/core/jitruntime.h"
#include "firestarter/Environment/Payload/Payload.hpp"
#include "firestarter/Logging/Log.hpp"
#include <memory>

namespace firestarter::environment::x86::payload {

class CompiledX86Payload final : public environment::payload::CompiledPayload {
private:
  inline static asmjit::JitRuntime Runtime = asmjit::JitRuntime();

  static void deleter(CompiledX86Payload* Payload) {
    if (Payload && Payload->highLoadFunctionPtr()) {
      Runtime.release(Payload->highLoadFunctionPtr());
    }
  }

  static void deleter(CompiledPayload* Payload) { deleter(dynamic_cast<CompiledX86Payload*>(Payload)); }

  CompiledX86Payload(const environment::payload::PayloadStats& Stats,
                     std::unique_ptr<environment::payload::Payload>&& PayloadPtr, HighLoadFunctionPtr HighLoadFunction)
      : CompiledPayload(Stats, std::move(PayloadPtr), HighLoadFunction) {}

public:
  CompiledX86Payload() = delete;
  ~CompiledX86Payload() override = default;

  template <class DerivedPayload>
  [[nodiscard]] static auto create(environment::payload::PayloadStats Stats, asmjit::CodeHolder& Code) -> UniquePtr {
    HighLoadFunctionPtr HighLoadFunction{};
    const auto Err = Runtime.add(&HighLoadFunction, &Code);
    if (Err) {
      workerLog::error() << "Asmjit adding Assembler to JitRuntime failed";
    }

    return {new CompiledX86Payload(Stats, std::move(std::make_unique<DerivedPayload>()), HighLoadFunction), deleter};
  }
};

} // namespace firestarter::environment::x86::payload
