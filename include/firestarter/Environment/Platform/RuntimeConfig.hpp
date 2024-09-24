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

#include <cassert>
#include <firestarter/Environment/Platform/PlatformConfig.hpp>

namespace firestarter::environment::platform {

class RuntimeConfig {
private:
  PlatformConfig const& PlatformConfigValue;
  std::unique_ptr<payload::Payload> Payload;
  unsigned Thread;
  std::vector<std::pair<std::string, unsigned>> PayloadSettings;
  unsigned InstructionCacheSize;
  std::list<unsigned> DataCacheBufferSize;
  unsigned RamBufferSize;
  unsigned Lines;

public:
  RuntimeConfig(PlatformConfig const& PlatformConfigValue, unsigned Thread, unsigned DetectedInstructionCacheSize)
      : PlatformConfigValue(PlatformConfigValue)
      , Payload(nullptr)
      , Thread(Thread)
      , PayloadSettings(PlatformConfigValue.getDefaultPayloadSettings())
      , InstructionCacheSize(PlatformConfigValue.instructionCacheSize())
      , DataCacheBufferSize(PlatformConfigValue.dataCacheBufferSize())
      , RamBufferSize(PlatformConfigValue.ramBufferSize())
      , Lines(PlatformConfigValue.lines()) {
    if (DetectedInstructionCacheSize != 0) {
      this->InstructionCacheSize = DetectedInstructionCacheSize;
    }
  };

  RuntimeConfig(const RuntimeConfig& Other)
      : PlatformConfigValue(Other.platformConfig())
      , Payload(Other.platformConfig().payload().clone())
      , Thread(Other.thread())
      , PayloadSettings(Other.payloadSettings())
      , InstructionCacheSize(Other.instructionCacheSize())
      , DataCacheBufferSize(Other.dataCacheBufferSize())
      , RamBufferSize(Other.ramBufferSize())
      , Lines(Other.lines()) {}

  ~RuntimeConfig() { Payload.reset(); }

  [[nodiscard]] auto platformConfig() const -> PlatformConfig const& { return PlatformConfigValue; }
  [[nodiscard]] auto payload() const -> payload::Payload& {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    assert(("Payload pointer is null. Each thread has to use it's own "
            "RuntimeConfig",
            Payload != nullptr));
#pragma GCC diagnostic pop
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    return *Payload;
  }
  [[nodiscard]] auto thread() const -> unsigned { return Thread; }
  [[nodiscard]] auto payloadSettings() const -> const std::vector<std::pair<std::string, unsigned>>& {
    return PayloadSettings;
  }
  [[nodiscard]] auto payloadItems() const -> std::vector<std::string> {
    std::vector<std::string> Items;
    Items.reserve(PayloadSettings.size());
    for (auto const& Pair : PayloadSettings) {
      Items.push_back(Pair.first);
    }
    return Items;
  }

  [[nodiscard]] auto instructionCacheSize() const -> unsigned { return InstructionCacheSize; }
  [[nodiscard]] auto dataCacheBufferSize() const -> const std::list<unsigned>& { return DataCacheBufferSize; }
  [[nodiscard]] auto ramBufferSize() const -> unsigned { return RamBufferSize; }
  [[nodiscard]] auto lines() const -> unsigned { return Lines; }

  void setPayloadSettings(std::vector<std::pair<std::string, unsigned>> const& PayloadSettings) {
    this->PayloadSettings = PayloadSettings;
  }

  void setLineCount(unsigned LineCount) { this->Lines = LineCount; }

  void printCodePathSummary() const {
    log::info() << "\n"
                << "  Taking " << platformConfig().payload().name() << " path optimized for " << platformConfig().name()
                << " - " << thread() << " thread(s) per core\n"
                << "  Used buffersizes per thread:";

    if (instructionCacheSize() != 0) {
      log::info() << "    - L1i-Cache: " << instructionCacheSize() / thread() << " Bytes";
    }

    unsigned I = 1;
    for (auto const& Bytes : dataCacheBufferSize()) {
      log::info() << "    - L" << I << "d-Cache: " << Bytes / thread() << " Bytes";
      I++;
    }

    log::info() << "    - Memory: " << ramBufferSize() / thread() << " Bytes";
  }
};

} // namespace firestarter::environment::platform
