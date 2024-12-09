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

#include "firestarter/Environment/X86/X86Environment.hpp"
#include "firestarter/Logging/Log.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <regex>

namespace firestarter::environment::x86 {

void X86Environment::selectFunction(std::optional<unsigned> FunctionId, const CPUTopology& Topology,
                                    bool AllowUnavailablePayload) {
  unsigned Id = 1;
  std::optional<std::string> DefaultPayloadName;
  const auto ProcessorICacheSize = Topology.instructionCacheSize();
  const auto ProcessorThreadsPerCore = Topology.homogenousResourceCount().NumThreadsPerCore;

  for (const auto& PlatformConfigPtr : PlatformConfigs) {
    for (auto const& ThreadsPerCore : PlatformConfigPtr->settings().threads()) {
      if (FunctionId) {
        // the selected function
        if (Id == *FunctionId) {
          if (!PlatformConfigPtr->isAvailable(processorInfos())) {
            const auto ErrorString =
                "Function " + std::to_string(*FunctionId) + " (\"" + PlatformConfigPtr->functionName(ThreadsPerCore) +
                "\") requires " + PlatformConfigPtr->payload()->name() + ", which is not supported by the processor.";
            if (AllowUnavailablePayload) {
              log::warn() << ErrorString;
            } else {
              throw std::invalid_argument(ErrorString);
            }
          }
          // found function
          setConfig(PlatformConfigPtr->cloneConcreate(ProcessorICacheSize, ThreadsPerCore));
          return;
        }
      } else {
        // default function
        if (PlatformConfigPtr->isDefault(processorInfos())) {
          if (ThreadsPerCore == ProcessorThreadsPerCore) {
            setConfig(PlatformConfigPtr->cloneConcreate(ProcessorICacheSize, ThreadsPerCore));
            return;
          }
          DefaultPayloadName = PlatformConfigPtr->payload()->name();
        }
        Id++;
      }
    }
  }

  if (FunctionId) {
    throw std::invalid_argument("unknown function id: " + std::to_string(*FunctionId) +
                                ", see --avail for available ids");
  }

  // no default found
  // use fallback
  if (DefaultPayloadName) {
    // default payload available, but number of threads per core is not
    // supported
    log::warn() << "No " << *DefaultPayloadName << " code path for " << ProcessorThreadsPerCore << " threads per core!";
  }
  log::warn() << processorInfos().vendor() << " " << processorInfos().model()
              << " is not supported by this version of FIRESTARTER!\n"
              << "Check project website for updates.";

  // loop over available implementation and check if they are marked as
  // fallback
  for (const auto& FallbackPlatformConfigPtr : FallbackPlatformConfigs) {
    if (FallbackPlatformConfigPtr->isAvailable(processorInfos())) {
      std::optional<unsigned> SelectedThreadsPerCore;
      // find the fallback implementation with the correct thread per core count
      for (auto const& ThreadsPerCore : FallbackPlatformConfigPtr->settings().threads()) {
        if (ThreadsPerCore == ProcessorThreadsPerCore) {
          SelectedThreadsPerCore = ThreadsPerCore;
        }
      }
      // Otherwise select the first available thread per core count
      if (!SelectedThreadsPerCore) {
        SelectedThreadsPerCore = FallbackPlatformConfigPtr->settings().threads().front();
      }
      setConfig(FallbackPlatformConfigPtr->cloneConcreate(ProcessorICacheSize, *SelectedThreadsPerCore));
      log::warn() << "Using function " << FallbackPlatformConfigPtr->functionName(*SelectedThreadsPerCore)
                  << " as fallback.\n"
                  << "You can use the parameter --function to try other "
                     "functions.";
      return;
    }
  }

  // no fallback found
  throw std::invalid_argument("No fallback implementation found for available ISA "
                              "extensions.");
}

void X86Environment::selectInstructionGroups(std::string Groups) {
  const auto Delimiter = ',';
  const std::regex Re("^(\\w+):(\\d+)$");
  const auto AvailableInstructionGroups = config().payload()->getAvailableInstructions();

  std::stringstream Ss(Groups);
  std::vector<std::pair<std::string, unsigned>> PayloadSettings = {};

  while (Ss.good()) {
    std::string Token;
    std::smatch M;
    std::getline(Ss, Token, Delimiter);

    if (std::regex_match(Token, M, Re)) {
      if (std::find(AvailableInstructionGroups.begin(), AvailableInstructionGroups.end(), M[1].str()) ==
          AvailableInstructionGroups.end()) {
        throw std::invalid_argument("Invalid instruction-group: " + M[1].str() +
                                    "\n       --run-instruction-groups format: multiple INST:VAL "
                                    "pairs comma-seperated");
      }
      auto Num = std::stoul(M[2].str());
      if (Num == 0) {
        throw std::invalid_argument("instruction-group VAL may not contain number 0"
                                    "\n       --run-instruction-groups format: multiple INST:VAL "
                                    "pairs comma-seperated");
      }
      PayloadSettings.emplace_back(M[1].str(), Num);
    } else {
      throw std::invalid_argument("Invalid symbols in instruction-group: " + Token +
                                  "\n       --run-instruction-groups format: multiple INST:VAL "
                                  "pairs comma-seperated");
    }
  }

  config().settings().selectInstructionGroups(PayloadSettings);

  log::info() << "  Running custom instruction group: " << Groups;
}

void X86Environment::printAvailableInstructionGroups() {
  std::stringstream Ss;

  for (auto const& Item : config().payload()->getAvailableInstructions()) {
    Ss << Item << ",";
  }

  auto S = Ss.str();
  if (!S.empty()) {
    S.pop_back();
  }

  log::info() << " available instruction-groups for payload " << config().payload()->name() << ":\n"
              << "  " << S;
}

void X86Environment::setLineCount(unsigned LineCount) { config().settings().setLineCount(LineCount); }

void X86Environment::printSelectedCodePathSummary() { config().printCodePathSummary(); }

void X86Environment::printFunctionSummary(bool ForceYes) {
  log::info() << " available load-functions:\n"
              << "  ID   | NAME                           | available on this "
                 "system | payload default setting\n"
              << "  "
                 "-------------------------------------------------------------"
                 "-------------------------------------------------------------"
                 "-----------------------------";

  auto Id = 1U;

  for (auto const& Config : PlatformConfigs) {
    for (auto const& ThreadsPerCore : Config->settings().threads()) {
      const char* Available = (Config->isAvailable(processorInfos()) || ForceYes) ? "yes" : "no";
      const auto& FunctionName = Config->functionName(ThreadsPerCore);
      const auto& InstructionGroupsString = Config->settings().getInstructionGroupsString();

      log::info() << "  " << std::right << std::setw(4) << Id << " | " << std::left << std::setw(30) << FunctionName
                  << " | " << std::left << std::setw(24) << Available << " | " << InstructionGroupsString;
      Id++;
    }
  }
}

} // namespace firestarter::environment::x86