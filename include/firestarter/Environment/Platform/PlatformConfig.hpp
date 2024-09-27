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

#include "../Payload/Payload.hpp"
#include <initializer_list>
#include <map>
#include <sstream>
#include <string>
#include <utility>

namespace firestarter::environment::platform {

class PlatformConfig {
private:
  std::string Name;
  std::list<unsigned> Threads;
  std::unique_ptr<payload::Payload> Payload;

protected:
  unsigned InstructionCacheSize;
  std::list<unsigned> DataCacheBufferSize;
  unsigned RamBufferSize;
  unsigned Lines;

public:
  PlatformConfig() = delete;

  PlatformConfig(std::string Name, std::list<unsigned> Threads, unsigned InstructionCacheSize,
                 std::initializer_list<unsigned> DataCacheBufferSize, unsigned RamBufferSize, unsigned Lines,
                 std::unique_ptr<payload::Payload>&& Payload)
      : Name(std::move(Name))
      , Threads(std::move(Threads))
      , Payload(std::move(Payload))
      , InstructionCacheSize(InstructionCacheSize)
      , DataCacheBufferSize(DataCacheBufferSize)
      , RamBufferSize(RamBufferSize)
      , Lines(Lines) {}
  virtual ~PlatformConfig() = default;

  [[nodiscard]] auto name() const -> const std::string& { return Name; }
  [[nodiscard]] auto instructionCacheSize() const -> unsigned { return InstructionCacheSize; }
  [[nodiscard]] auto dataCacheBufferSize() const -> const std::list<unsigned>& { return DataCacheBufferSize; }
  [[nodiscard]] auto ramBufferSize() const -> unsigned { return RamBufferSize; }
  [[nodiscard]] auto lines() const -> unsigned { return Lines; }
  [[nodiscard]] auto payload() const -> payload::Payload const& { return *Payload; }

  [[nodiscard]] auto getThreadMap() const -> std::map<unsigned, std::string> {
    std::map<unsigned, std::string> ThreadMap;

    for (auto const& Thread : Threads) {
      std::stringstream FunctionName;
      FunctionName << "FUNC_" << name() << "_" << payload().name() << "_" << Thread << "T";
      ThreadMap[Thread] = FunctionName.str();
    }

    return ThreadMap;
  }

  [[nodiscard]] auto isAvailable() const -> bool { return payload().isAvailable(); }

  [[nodiscard]] virtual auto isDefault() const -> bool = 0;

  [[nodiscard]] virtual auto getDefaultPayloadSettings() const -> std::vector<std::pair<std::string, unsigned>> = 0;

  [[nodiscard]] auto getDefaultPayloadSettingsString() const -> std::string {
    std::stringstream Ss;

    for (auto const& [name, value] : this->getDefaultPayloadSettings()) {
      Ss << name << ":" << value << ",";
    }

    auto Str = Ss.str();
    if (Str.size() > 0) {
      Str.pop_back();
    }

    return Str;
  }
};

} // namespace firestarter::environment::platform
