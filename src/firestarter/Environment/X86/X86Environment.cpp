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

#include <firestarter/Environment/X86/X86Environment.hpp>
#include <firestarter/Logging/Log.hpp>

#include <algorithm>
#include <cstdio>
#include <regex>

namespace firestarter::environment::x86 {

void X86Environment::evaluateFunctions() {
  for (const auto& Ctor : PlatformConfigsCtor) {
    // add asmjit for model and family detection
    PlatformConfigs.emplace_back(
        Ctor(topology().featuresAsmjit(), topology().familyId(), topology().modelId(), topology().numThreadsPerCore()));
  }

  for (const auto& Ctor : FallbackPlatformConfigsCtor) {
    FallbackPlatformConfigs.emplace_back(
        Ctor(topology().featuresAsmjit(), topology().familyId(), topology().modelId(), topology().numThreadsPerCore()));
  }
}

auto X86Environment::selectFunction(unsigned FunctionId, bool AllowUnavailablePayload) -> int {
  unsigned id = 1;
  std::string defaultPayloadName("");

  // if functionId is 0 get the default or fallback
  for (const auto& Config : PlatformConfigs) {
    for (auto const& [thread, functionName] : Config->getThreadMap()) {
      // the selected function
      if (id == FunctionId) {
        if (!Config->isAvailable()) {
          log::error() << "Function " << FunctionId << " (\"" << functionName << "\") requires "
                       << Config->payload().name() << ", which is not supported by the processor.";
          if (!AllowUnavailablePayload) {
            return EXIT_FAILURE;
          }
        }
        // found function
        SelectedConfig =
            new ::firestarter::environment::platform::RuntimeConfig(*Config, thread, topology().instructionCacheSize());
        return EXIT_SUCCESS;
      }
      // default function
      if (0 == FunctionId && Config->isDefault()) {
        if (thread == topology().numThreadsPerCore()) {
          SelectedConfig = new ::firestarter::environment::platform::RuntimeConfig(*Config, thread,
                                                                                   topology().instructionCacheSize());
          return EXIT_SUCCESS;
        } else {
          defaultPayloadName = Config->payload().name();
        }
      }
      id++;
    }
  }

  // no default found
  // use fallback
  if (0 == FunctionId) {
    if (!defaultPayloadName.empty()) {
      // default payload available, but number of threads per core is not
      // supported
      log::warn() << "No " << defaultPayloadName << " code path for " << topology().numThreadsPerCore()
                  << " threads per core!";
    }
    log::warn() << topology().vendor() << " " << topology().model()
                << " is not supported by this version of FIRESTARTER!\n"
                << "Check project website for updates.";

    // loop over available implementation and check if they are marked as
    // fallback
    for (const auto& Config : FallbackPlatformConfigs) {
      if (Config->isAvailable()) {
        auto selectedThread = 0;
        auto selectedFunctionName = std::string("");
        for (auto const& [thread, functionName] : Config->getThreadMap()) {
          if (thread == topology().numThreadsPerCore()) {
            selectedThread = thread;
            selectedFunctionName = functionName;
          }
        }
        if (selectedThread == 0) {
          selectedThread = Config->getThreadMap().begin()->first;
          selectedFunctionName = Config->getThreadMap().begin()->second;
        }
        SelectedConfig = new ::firestarter::environment::platform::RuntimeConfig(*Config, selectedThread,
                                                                                 topology().instructionCacheSize());
        log::warn() << "Using function " << selectedFunctionName << " as fallback.\n"
                    << "You can use the parameter --function to try other "
                       "functions.";
        return EXIT_SUCCESS;
      }
    }

    // no fallback found
    log::error() << "No fallback implementation found for available ISA "
                    "extensions.";
    return EXIT_FAILURE;
  }

  log::error() << "unknown function id: " << FunctionId << ", see --avail for available ids";
  return EXIT_FAILURE;
}

int X86Environment::selectInstructionGroups(std::string groups) {
  const std::string delimiter = ",";
  const std::regex re("^(\\w+):(\\d+)$");
  const auto availableInstructionGroups = selectedConfig().platformConfig().payload().getAvailableInstructions();

  std::stringstream ss(groups);
  std::vector<std::pair<std::string, unsigned>> payloadSettings = {};

  while (ss.good()) {
    std::string token;
    std::smatch m;
    std::getline(ss, token, ',');

    if (std::regex_match(token, m, re)) {
      if (std::find(availableInstructionGroups.begin(), availableInstructionGroups.end(), m[1].str()) ==
          availableInstructionGroups.end()) {
        log::error() << "Invalid instruction-group: " << m[1].str()
                     << "\n       --run-instruction-groups format: multiple INST:VAL "
                        "pairs comma-seperated";
        return EXIT_FAILURE;
      }
      int num = std::stoul(m[2].str());
      if (num == 0) {
        log::error() << "instruction-group VAL may not contain number 0"
                     << "\n       --run-instruction-groups format: multiple INST:VAL "
                        "pairs comma-seperated";
        return EXIT_FAILURE;
      }
      payloadSettings.push_back(std::make_pair(m[1].str(), num));
    } else {
      log::error() << "Invalid symbols in instruction-group: " << token
                   << "\n       --run-instruction-groups format: multiple INST:VAL "
                      "pairs comma-seperated";
      return EXIT_FAILURE;
    }
  }

  selectedConfig().setPayloadSettings(payloadSettings);

  log::info() << "  Running custom instruction group: " << groups;

  return EXIT_SUCCESS;
}

void X86Environment::printAvailableInstructionGroups() {
  std::stringstream ss;

  for (auto const& item : selectedConfig().platformConfig().payload().getAvailableInstructions()) {
    ss << item << ",";
  }

  auto s = ss.str();
  if (s.size() > 0) {
    s.pop_back();
  }

  log::info() << " available instruction-groups for payload " << selectedConfig().platformConfig().payload().name()
              << ":\n"
              << "  " << s;
}

void X86Environment::setLineCount(unsigned lineCount) { selectedConfig().setLineCount(lineCount); }

void X86Environment::printSelectedCodePathSummary() { selectedConfig().printCodePathSummary(); }

void X86Environment::printFunctionSummary() {
  log::info() << " available load-functions:\n"
              << "  ID   | NAME                           | available on this "
                 "system | payload default setting\n"
              << "  "
                 "-------------------------------------------------------------"
                 "-------------------------------------------------------------"
                 "-----------------------------";

  unsigned id = 1;

  for (auto const& config : PlatformConfigs) {
    for (auto const& [thread, functionName] : config->getThreadMap()) {
      const char* available = config->isAvailable() ? "yes" : "no";
      const char* fmt = "  %4u | %-30s | %-24s | %s";
      int sz = std::snprintf(nullptr, 0, fmt, id, functionName.c_str(), available,
                             config->getDefaultPayloadSettingsString().c_str());
      std::vector<char> buf(sz + 1);
      std::snprintf(&buf[0], buf.size(), fmt, id, functionName.c_str(), available,
                    config->getDefaultPayloadSettingsString().c_str());
      log::info() << std::string(&buf[0]);
      id++;
    }
  }
}

} // namespace firestarter::environment::x86