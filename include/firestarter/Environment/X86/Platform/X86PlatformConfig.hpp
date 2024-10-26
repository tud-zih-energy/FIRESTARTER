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

#include "../../Platform/PlatformConfig.hpp"
#include "../Payload/X86Payload.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <vector> // IWYU pragma: keep

namespace firestarter::environment::x86::platform {

class X86PlatformConfig : public environment::platform::PlatformConfig {
private:
  unsigned Family;
  std::list<unsigned> Models;
  unsigned CurrentFamily;
  unsigned CurrentModel;

public:
  X86PlatformConfig(std::string Name, unsigned Family, std::initializer_list<unsigned> Models,
                    std::initializer_list<unsigned> Threads, unsigned InstructionCacheSize,
                    std::initializer_list<unsigned> DataCacheBufferSize, unsigned RamBuffersize, unsigned Lines,
                    unsigned CurrentFamily, unsigned CurrentModel, std::unique_ptr<payload::X86Payload>&& Payload)
      : PlatformConfig(std::move(Name), Threads, InstructionCacheSize, DataCacheBufferSize, RamBuffersize, Lines,
                       std::move(Payload))
      , Family(Family)
      , Models(Models)
      , CurrentFamily(CurrentFamily)
      , CurrentModel(CurrentModel) {}

  [[nodiscard]] auto isDefault() const -> bool override {
    return Family == CurrentFamily && (std::find(Models.begin(), Models.end(), CurrentModel) != Models.end()) &&
           isAvailable();
  }
};

} // namespace firestarter::environment::x86::platform
