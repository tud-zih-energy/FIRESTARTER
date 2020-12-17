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

#include <firestarter/Environment/X86/X86Environment.hpp>
#include <firestarter/Logging/Log.hpp>

#include <algorithm>
#include <cstdio>
#include <regex>

using namespace firestarter::environment::x86;

void X86Environment::evaluateFunctions() {
  for (auto ctor : this->platformConfigsCtor) {
    // add asmjit for model and family detection
    this->platformConfigs.push_back(
        ctor(this->topology().featuresAsmjit(), this->topology().familyId(),
             this->topology().modelId(), this->topology().numThreadsPerCore()));
  }

  for (auto ctor : this->fallbackPlatformConfigsCtor) {
    this->fallbackPlatformConfigs.push_back(
        ctor(this->topology().featuresAsmjit(), this->topology().familyId(),
             this->topology().modelId(), this->topology().numThreadsPerCore()));
  }
}

int X86Environment::selectFunction(unsigned functionId,
                                   bool allowUnavailablePayload) {
  unsigned id = 1;
  std::string defaultPayloadName("");

  // if functionId is 0 get the default or fallback
  for (auto config : this->platformConfigs) {
    for (auto const &[thread, functionName] : config->getThreadMap()) {
      // the selected function
      if (id == functionId) {
        if (!config->isAvailable()) {
          log::error() << "Function " << functionId << " (\"" << functionName
                       << "\") requires " << config->payload().name()
                       << ", which is not supported by the processor.";
          if (!allowUnavailablePayload) {
            return EXIT_FAILURE;
          }
        }
        // found function
        this->_selectedConfig =
            new ::firestarter::environment::platform::RuntimeConfig(
                *config, thread, this->topology().instructionCacheSize());
        return EXIT_SUCCESS;
      }
      // default function
      if (0 == functionId && config->isDefault()) {
        if (thread == this->topology().numThreadsPerCore()) {
          this->_selectedConfig =
              new ::firestarter::environment::platform::RuntimeConfig(
                  *config, thread, this->topology().instructionCacheSize());
          return EXIT_SUCCESS;
        } else {
          defaultPayloadName = config->payload().name();
        }
      }
      id++;
    }
  }

  // no default found
  // use fallback
  if (0 == functionId) {
    if (!defaultPayloadName.empty()) {
      // default payload available, but number of threads per core is not
      // supported
      log::warn() << "No " << defaultPayloadName << " code path for "
                  << this->topology().numThreadsPerCore()
                  << " threads per core!";
    }
    log::warn() << this->topology().vendor() << " " << this->topology().model()
                << " is not supported by this version of FIRESTARTER!\n"
                << "Check project website for updates.";

    // loop over available implementation and check if they are marked as
    // fallback
    for (auto config : this->fallbackPlatformConfigs) {
      if (config->isAvailable()) {
        auto selectedThread = 0;
        auto selectedFunctionName = std::string("");
        for (auto const &[thread, functionName] : config->getThreadMap()) {
          if (thread == this->topology().numThreadsPerCore()) {
            selectedThread = thread;
            selectedFunctionName = functionName;
          }
        }
        if (selectedThread == 0) {
          selectedThread = config->getThreadMap().begin()->first;
          selectedFunctionName = config->getThreadMap().begin()->second;
        }
        this->_selectedConfig =
            new ::firestarter::environment::platform::RuntimeConfig(
                *config, selectedThread,
                this->topology().instructionCacheSize());
        log::warn() << "Using function " << selectedFunctionName
                    << " as fallback.\n"
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

  log::error() << "unknown function id: " << functionId
               << ", see --avail for available ids";
  return EXIT_FAILURE;
}

int X86Environment::selectInstructionGroups(std::string groups) {
  const std::string delimiter = ",";
  const std::regex re("^(\\w+):(\\d+)$");
  const auto availableInstructionGroups = this->selectedConfig()
                                              .platformConfig()
                                              .payload()
                                              .getAvailableInstructions();

  std::stringstream ss(groups);
  std::vector<std::pair<std::string, unsigned>> payloadSettings = {};

  while (ss.good()) {
    std::string token;
    std::smatch m;
    std::getline(ss, token, ',');

    if (std::regex_match(token, m, re)) {
      if (std::find(availableInstructionGroups.begin(),
                    availableInstructionGroups.end(),
                    m[1].str()) == availableInstructionGroups.end()) {
        log::error() << "Invalid instruction-group: " << m[1].str();
        return EXIT_FAILURE;
      }
      int num = std::stoul(m[2].str());
      if (num == 0) {
        log::error() << "instruction-group VAL may not contain number 0";
        return EXIT_FAILURE;
      }
      payloadSettings.push_back(std::make_pair(m[1].str(), num));
    } else {
      log::error() << "Invalid symbols in instruction-group: " << token;
      return EXIT_FAILURE;
    }
  }

  this->selectedConfig().setPayloadSettings(payloadSettings);

  log::info() << "  Running custom instruction group: " << groups;

  return EXIT_SUCCESS;
}

void X86Environment::printAvailableInstructionGroups() {
  std::stringstream ss;

  for (auto const &item : this->selectedConfig()
                              .platformConfig()
                              .payload()
                              .getAvailableInstructions()) {
    ss << item << ",";
  }

  auto s = ss.str();
  if (s.size() > 0) {
    s.pop_back();
  }

  log::info() << " available instruction-groups for payload "
              << this->selectedConfig().platformConfig().payload().name()
              << ":\n"
              << "  " << s;
}

void X86Environment::setLineCount(unsigned lineCount) {
  this->selectedConfig().setLineCount(lineCount);
}

void X86Environment::printSelectedCodePathSummary() {
  this->selectedConfig().printCodePathSummary();
}

void X86Environment::printFunctionSummary() {
  log::info() << " available load-functions:\n"
              << "  ID   | NAME                           | available on this "
                 "system | payload default setting\n"
              << "  "
                 "-------------------------------------------------------------"
                 "-------------------------------------------------------------"
                 "-----------------------------";

  unsigned id = 1;

  for (auto const &config : this->platformConfigs) {
    for (auto const &[thread, functionName] : config->getThreadMap()) {
      const char *available = config->isAvailable() ? "yes" : "no";
      const char *fmt = "  %4u | %-30s | %-24s | %s";
      int sz =
          std::snprintf(nullptr, 0, fmt, id, functionName.c_str(), available,
                        config->getDefaultPayloadSettingsString().c_str());
      std::vector<char> buf(sz + 1);
      std::snprintf(&buf[0], buf.size(), fmt, id, functionName.c_str(),
                    available,
                    config->getDefaultPayloadSettingsString().c_str());
      log::info() << std::string(&buf[0]);
      id++;
    }
  }
}
