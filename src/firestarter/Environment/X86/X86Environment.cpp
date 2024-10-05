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

void X86Environment::selectFunction(unsigned FunctionId, bool AllowUnavailablePayload) {
  unsigned Id = 1;
  std::string DefaultPayloadName;

  // if functionId is 0 get the default or fallback
  for (const auto& Config : PlatformConfigs) {
    for (auto const& [thread, functionName] : Config->getThreadMap()) {
      // the selected function
      if (Id == FunctionId) {
        if (!Config->isAvailable()) {
          const auto ErrorString = "Function " + std::to_string(FunctionId) + " (\"" + functionName + "\") requires " +
                                   Config->payload().name() + ", which is not supported by the processor.";
          if (AllowUnavailablePayload) {
            log::error() << ErrorString;
          } else {
            throw std::invalid_argument(ErrorString);
          }
        }
        // found function
        SelectedConfig =
            new ::firestarter::environment::platform::RuntimeConfig(*Config, thread, topology().instructionCacheSize());
        return;
      }
      // default function
      if (0 == FunctionId && Config->isDefault()) {
        if (thread == topology().numThreadsPerCore()) {
          SelectedConfig = new ::firestarter::environment::platform::RuntimeConfig(*Config, thread,
                                                                                   topology().instructionCacheSize());
          return;
        }
        DefaultPayloadName = Config->payload().name();
      }
      Id++;
    }
  }

  // no default found
  // use fallback
  if (0 == FunctionId) {
    if (!DefaultPayloadName.empty()) {
      // default payload available, but number of threads per core is not
      // supported
      log::warn() << "No " << DefaultPayloadName << " code path for " << topology().numThreadsPerCore()
                  << " threads per core!";
    }
    log::warn() << topology().vendor() << " " << topology().model()
                << " is not supported by this version of FIRESTARTER!\n"
                << "Check project website for updates.";

    // loop over available implementation and check if they are marked as
    // fallback
    for (const auto& Config : FallbackPlatformConfigs) {
      if (Config->isAvailable()) {
        auto SelectedThread = 0;
        auto SelectedFunctionName = std::string("");
        for (auto const& [Thread, FunctionName] : Config->getThreadMap()) {
          if (Thread == topology().numThreadsPerCore()) {
            SelectedThread = Thread;
            SelectedFunctionName = FunctionName;
          }
        }
        if (SelectedThread == 0) {
          SelectedThread = Config->getThreadMap().begin()->first;
          SelectedFunctionName = Config->getThreadMap().begin()->second;
        }
        SelectedConfig = new ::firestarter::environment::platform::RuntimeConfig(*Config, SelectedThread,
                                                                                 topology().instructionCacheSize());
        log::warn() << "Using function " << SelectedFunctionName << " as fallback.\n"
                    << "You can use the parameter --function to try other "
                       "functions.";
        return;
      }
    }

    // no fallback found
    throw std::invalid_argument("No fallback implementation found for available ISA "
                                "extensions.");
  }

  throw std::invalid_argument("unknown function id: " + std::to_string(FunctionId) + ", see --avail for available ids");
}

void X86Environment::selectInstructionGroups(std::string Groups) {
  const std::string Delimiter = ",";
  const std::regex Re("^(\\w+):(\\d+)$");
  const auto AvailableInstructionGroups = selectedConfig().platformConfig().payload().getAvailableInstructions();

  std::stringstream Ss(Groups);
  std::vector<std::pair<std::string, unsigned>> PayloadSettings = {};

  while (Ss.good()) {
    std::string Token;
    std::smatch M;
    std::getline(Ss, Token, ',');

    if (std::regex_match(Token, M, Re)) {
      if (std::find(AvailableInstructionGroups.begin(), AvailableInstructionGroups.end(), M[1].str()) ==
          AvailableInstructionGroups.end()) {
        throw std::invalid_argument("Invalid instruction-group: " + M[1].str() +
                                    "\n       --run-instruction-groups format: multiple INST:VAL "
                                    "pairs comma-seperated");
      }
      int Num = std::stoul(M[2].str());
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

  selectedConfig().setPayloadSettings(PayloadSettings);

  log::info() << "  Running custom instruction group: " << Groups;
}

void X86Environment::printAvailableInstructionGroups() {
  std::stringstream Ss;

  for (auto const& Item : selectedConfig().platformConfig().payload().getAvailableInstructions()) {
    Ss << Item << ",";
  }

  auto S = Ss.str();
  if (!S.empty()) {
    S.pop_back();
  }

  log::info() << " available instruction-groups for payload " << selectedConfig().platformConfig().payload().name()
              << ":\n"
              << "  " << S;
}

void X86Environment::setLineCount(unsigned LineCount) { selectedConfig().setLineCount(LineCount); }

void X86Environment::printSelectedCodePathSummary() { selectedConfig().printCodePathSummary(); }

void X86Environment::printFunctionSummary() {
  log::info() << " available load-functions:\n"
              << "  ID   | NAME                           | available on this "
                 "system | payload default setting\n"
              << "  "
                 "-------------------------------------------------------------"
                 "-------------------------------------------------------------"
                 "-----------------------------";

  unsigned Id = 1;

  for (auto const& Config : PlatformConfigs) {
    for (auto const& [thread, functionName] : Config->getThreadMap()) {
      const char* Available = Config->isAvailable() ? "yes" : "no";
      const char* Fmt = "  %4u | %-30s | %-24s | %s";
      int Sz = std::snprintf(nullptr, 0, Fmt, Id, functionName.c_str(), Available,
                             Config->getDefaultPayloadSettingsString().c_str());
      std::vector<char> Buf(Sz + 1);
      std::snprintf(Buf.data(), Buf.size(), Fmt, Id, functionName.c_str(), Available,
                    Config->getDefaultPayloadSettingsString().c_str());
      log::info() << std::string(Buf.data());
      Id++;
    }
  }
}

} // namespace firestarter::environment::x86