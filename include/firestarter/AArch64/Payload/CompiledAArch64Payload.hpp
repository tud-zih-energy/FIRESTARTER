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

#include "firestarter/Logging/Log.hpp"
#include "firestarter/Payload/CompiledPayload.hpp"

#include <asmjit/asmjit.h>
#include <memory>

namespace firestarter::aarch64::payload {

/// This class provides the functionality to compile a payload created with asmjit and create a unique pointer to the
/// CompiledPayload class which can be used to execute the functions of this payload.
class CompiledAArch64Payload final : public firestarter::payload::CompiledPayload {
private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  inline static asmjit::JitRuntime Runtime = asmjit::JitRuntime();

  /// Custom deleter to release the memory of the high load function from the asmjit runtime.
  static void deleter(CompiledAArch64Payload* Payload) {
    if (Payload && Payload->highLoadFunctionPtr()) {
      Runtime.release(Payload->highLoadFunctionPtr());
    }
  }
  static void deleter(CompiledPayload* Payload) { deleter(dynamic_cast<CompiledAArch64Payload*>(Payload)); }

  /// Wrap the CompiledPayload class and forward all arguments.
  CompiledAArch64Payload(const firestarter::payload::PayloadStats& Stats,
                         std::unique_ptr<firestarter::payload::Payload>&& PayloadPtr,
                         HighLoadFunctionPtr HighLoadFunction)
      : CompiledPayload(Stats, std::move(PayloadPtr), HighLoadFunction) {}

public:
  CompiledAArch64Payload() = delete;
  ~CompiledAArch64Payload() override = default;

  /// Create a unique pointer to a compiled payload from payload stats and assembly in a code holder.
  template <class DerivedPayload>
  [[nodiscard]] static auto create(firestarter::payload::PayloadStats Stats, asmjit::CodeHolder& Code) -> UniquePtr {
    HighLoadFunctionPtr HighLoadFunction{};
    const auto Err = Runtime.add(&HighLoadFunction, &Code);
    if (Err) {
      workerLog::error() << "Asmjit adding Assembler to JitRuntime failed";
    }

    return {new CompiledAArch64Payload(Stats, std::move(std::make_unique<DerivedPayload>()), HighLoadFunction),
            deleter};
  }
};

} // namespace firestarter::aarch64::payload
