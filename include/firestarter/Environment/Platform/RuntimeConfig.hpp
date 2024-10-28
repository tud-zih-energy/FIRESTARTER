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

#include "../../Logging/Log.hpp"
#include "../Platform/PlatformConfig.hpp"
#include <cassert>

namespace firestarter::environment::platform {

// This is effectivly a wrapper around a PlatformConfig that allow overriding some vairables.
// TODO: move these functions into the PlatformConfig and make them non const. The default PlatformConfig(s) shall be
// const.
class RuntimeConfig {
private:
  std::shared_ptr<PlatformConfig> SelectedPlatformConfig;
  unsigned Thread;
  std::vector<std::pair<std::string, unsigned>> PayloadSettings;
  unsigned InstructionCacheSize;
  std::list<unsigned> DataCacheBufferSize;
  unsigned RamBufferSize;
  unsigned Lines;

public:
  RuntimeConfig(const std::shared_ptr<PlatformConfig>& SelectedPlatformConfig, unsigned Thread,
                unsigned DetectedInstructionCacheSize)
      : SelectedPlatformConfig(SelectedPlatformConfig)
      , Thread(Thread)
      , PayloadSettings(SelectedPlatformConfig->getDefaultPayloadSettings())
      , InstructionCacheSize(SelectedPlatformConfig->instructionCacheSize())
      , DataCacheBufferSize(SelectedPlatformConfig->dataCacheBufferSize())
      , RamBufferSize(SelectedPlatformConfig->ramBufferSize())
      , Lines(SelectedPlatformConfig->lines()) {
    if (DetectedInstructionCacheSize != 0) {
      this->InstructionCacheSize = DetectedInstructionCacheSize;
    }
  };

  ~RuntimeConfig() = default;

  [[nodiscard]] auto platformConfig() const -> PlatformConfig const& { return *SelectedPlatformConfig; }
  [[nodiscard]] auto payload() const -> const payload::Payload& { return SelectedPlatformConfig->payload(); }
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
