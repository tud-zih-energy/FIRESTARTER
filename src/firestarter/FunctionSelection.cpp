/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/FunctionSelection.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/Platform/PlatformConfigAndThreads.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>

namespace firestarter {

auto FunctionSelection::selectAvailableFunction(unsigned FunctionId, const ProcessorInformation& ProcessorInfos,
                                                const CPUTopology& Topology, bool AllowUnavailablePayload) const
    -> std::unique_ptr<platform::PlatformConfig> {
  unsigned Id = 1;
  std::optional<std::string> DefaultPayloadName;
  const auto ProcessorICacheSize = Topology.instructionCacheSize();

  for (const auto& Platform : platform::PlatformConfigAndThreads::fromPlatformConfigs(platformConfigs())) {
    // the selected function
    if (Id == FunctionId) {
      if (!Platform.Config->payload()->isAvailable(ProcessorInfos.cpuFeatures())) {
        const auto ErrorString = "Function " + std::to_string(FunctionId) + " (\"" +
                                 Platform.Config->functionName(Platform.ThreadCount) + "\") requires " +
                                 Platform.Config->payload()->name() + ", which is not supported by the processor.";
        if (AllowUnavailablePayload) {
          log::warn() << ErrorString;
        } else {
          throw std::invalid_argument(ErrorString);
        }
      }
      // found function
      return Platform.Config->cloneConcreate(ProcessorICacheSize, Platform.ThreadCount);
    }
    Id++;
  }

  throw std::invalid_argument("unknown function id: " + std::to_string(FunctionId) + ", see --avail for available ids");
}

auto FunctionSelection::selectDefaultOrFallbackFunction(const ProcessorInformation& ProcessorInfos,
                                                        const CPUTopology& Topology) const
    -> std::unique_ptr<platform::PlatformConfig> {
  std::optional<std::string> DefaultPayloadName;
  const auto ProcessorICacheSize = Topology.instructionCacheSize();
  const auto ProcessorThreadsPerCore = Topology.homogenousResourceCount().NumThreadsPerCore;

  for (const auto& Platform : platform::PlatformConfigAndThreads::fromPlatformConfigs(platformConfigs())) {
    // default function
    if (Platform.Config->isDefault(ProcessorInfos)) {
      if (Platform.ThreadCount == ProcessorThreadsPerCore) {
        return Platform.Config->cloneConcreate(ProcessorICacheSize, Platform.ThreadCount);
      }
      DefaultPayloadName = Platform.Config->payload()->name();
    }
  }

  // no default found
  // use fallback
  if (DefaultPayloadName) {
    // default payload available, but number of threads per core is not
    // supported
    log::warn() << "No " << *DefaultPayloadName << " code path for " << ProcessorThreadsPerCore << " threads per core!";
  }
  log::warn() << ProcessorInfos.vendor() << " " << ProcessorInfos.model()
              << " is not supported by this version of FIRESTARTER!\n"
              << "Check project website for updates.";

  // loop over available implementation and check if they are marked as
  // fallback
  for (const auto& FallbackPlatformConfigPtr : fallbackPlatformConfigs()) {
    if (FallbackPlatformConfigPtr->payload()->isAvailable(ProcessorInfos.cpuFeatures())) {
      unsigned SelectedThreadsPerCore{};

      // find the fallback implementation with the correct thread per core count or select the first available thread
      // per core count
      {
        const auto& Threads = FallbackPlatformConfigPtr->constRef().settings().threads();
        const auto& ThreadIt = std::find(Threads.cbegin(), Threads.cend(), ProcessorThreadsPerCore);
        if (ThreadIt == Threads.cend()) {
          SelectedThreadsPerCore = Threads.front();
        } else {
          SelectedThreadsPerCore = *ThreadIt;
        }
      }

      log::warn() << "Using function " << FallbackPlatformConfigPtr->functionName(SelectedThreadsPerCore)
                  << " as fallback.\n"
                  << "You can use the parameter --function to try other "
                     "functions.";
      return FallbackPlatformConfigPtr->cloneConcreate(ProcessorICacheSize, SelectedThreadsPerCore);
    }
  }

  // no fallback found
  throw std::invalid_argument("No fallback implementation found for available ISA "
                              "extensions.");
}

auto FunctionSelection::selectFunction(std::optional<unsigned> FunctionId, const ProcessorInformation& ProcessorInfos,
                                       const CPUTopology& Topology, bool AllowUnavailablePayload) const
    -> std::unique_ptr<platform::PlatformConfig> {
  if (FunctionId) {
    return selectAvailableFunction(*FunctionId, ProcessorInfos, Topology, AllowUnavailablePayload);
  }
  return selectDefaultOrFallbackFunction(ProcessorInfos, Topology);
}

void FunctionSelection::printFunctionSummary(const ProcessorInformation& ProcessorInfos, bool ForceYes) const {
  log::info() << " available load-functions:\n"
              << "  ID   | NAME                           | available on this "
                 "system | payload default setting\n"
              << "  "
                 "-------------------------------------------------------------"
                 "-------------------------------------------------------------"
                 "-----------------------------";

  auto Id = 1U;

  for (const auto& Platform : platform::PlatformConfigAndThreads::fromPlatformConfigs(platformConfigs())) {
    const char* Available =
        (Platform.Config->payload()->isAvailable(ProcessorInfos.cpuFeatures()) || ForceYes) ? "yes" : "no";
    const auto& FunctionName = Platform.Config->functionName(Platform.ThreadCount);
    const auto& InstructionGroupsString = Platform.Config->constRef().settings().groups();

    log::info() << "  " << std::right << std::setw(4) << Id << " | " << std::left << std::setw(30) << FunctionName
                << " | " << std::left << std::setw(24) << Available << " | " << InstructionGroupsString;
    Id++;
  }
}

} // namespace firestarter