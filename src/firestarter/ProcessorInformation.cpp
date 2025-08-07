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

#include "firestarter/ProcessorInformation.hpp"
#include "firestarter/CpuFeatures.hpp"
#include "firestarter/CpuModel.hpp"
#include "firestarter/Logging/Log.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <utility>

#if defined(_WIN32) or defined(__APPLE__)
#include <array>
#endif

namespace firestarter {

void ProcessorInformation::print() const {
  std::stringstream Ss;

  for (auto const& Entry : features()) {
    Ss << Entry << " ";
  }

  log::info() << "  processor characteristics:\n"
              << "    architecture:       " << architecture() << "\n"
              << "    vendor:             " << vendor() << "\n"
              << "    processor-name:     " << processorName() << "\n"
              << "    model:              " << model() << "\n"
              << "    frequency:          " << clockrate() / 1000000 << " MHz\n"
              << "    supported features: " << Ss.str();
}

ProcessorInformation::ProcessorInformation(std::string Architecture, std::unique_ptr<CpuFeatures>&& Features,
                                           std::unique_ptr<CpuModel>&& Model)
    : Features(std::move(Features))
    , Model(std::move(Model))
    , Architecture(std::move(Architecture)) {

  // get vendor, processor name and clockrate for linux
#if defined(linux) || defined(__linux__)
  {
    auto ProcCpuinfo = getFileAsStream("/proc/cpuinfo");
    std::string Line;
    std::string ClockrateStr = "0";

    while (std::getline(ProcCpuinfo, Line, '\n')) {
      const std::regex VendorIdRe("^vendor_id.*:\\s*(.*)\\s*$");
      const std::regex ModelNameRe("^model name.*:\\s*(.*)\\s*$");
      const std::regex CpuMHzRe("^cpu MHz.*:\\s*(.*)\\s*$");
      std::smatch VendorIdMatch;
      std::smatch ModelNameMatch;
      std::smatch CpuMHzMatch;

      if (std::regex_match(Line, VendorIdMatch, VendorIdRe)) {
        Vendor = VendorIdMatch[1].str();
      }

      if (std::regex_match(Line, ModelNameMatch, ModelNameRe)) {
        ProcessorName = ModelNameMatch[1].str();
      }

      if (std::regex_match(Line, CpuMHzMatch, CpuMHzRe)) {
        ClockrateStr = CpuMHzMatch[1].str();
      }
    }

    if (Vendor.empty()) {
      log::warn() << "Could determine vendor from /proc/cpuinfo";
    }

    if (ProcessorName.empty()) {
      log::warn() << "Could determine processor-name from /proc/cpuinfo";
    }

    if (ClockrateStr == "0") {
      firestarter::log::warn() << "Can't determine clockrate from /proc/cpuinfo";
    } else {
      firestarter::log::trace() << "Clockrate from /proc/cpuinfo is " << ClockrateStr;
      Clockrate = static_cast<uint64_t>(1000000U) * std::stoi(ClockrateStr);
    }

    auto Governor = scalingGovernor();
    if (!Governor.empty()) {

      auto ScalingCurFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").str();
      auto CpuinfoCurFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq").str();
      auto ScalingMaxFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq").str();
      auto CpuinfoMaxFreq = getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").str();

      if (Governor == "performance" || Governor == "powersave") {
        if (ScalingCurFreq.empty()) {
          if (!CpuinfoCurFreq.empty()) {
            ClockrateStr = CpuinfoCurFreq;
          }
        } else {
          ClockrateStr = ScalingCurFreq;
        }
      } else {
        if (ScalingMaxFreq.empty()) {
          if (!CpuinfoMaxFreq.empty()) {
            ClockrateStr = CpuinfoMaxFreq;
          }
        } else {
          ClockrateStr = ScalingMaxFreq;
        }
      }

      Clockrate = static_cast<uint64_t>(1000U) * std::stoi(ClockrateStr);
    }
  }
#endif

  // try to detect processor name for macos
#ifdef __APPLE__
  {
    // use sysctl to detect the name
    std::array<char, 128> Buffer{};
    const auto* Cmd = "sysctl -n machdep.cpu.brand_string";
    std::unique_ptr<FILE, decltype(&pclose)> Pipe(popen(Cmd, "r"), pclose);
    if (!Pipe) {
      log::warn() << "Could not determine processor-name";
    }
    if (fgets(Buffer.data(), Buffer.size(), Pipe.get()) != nullptr) {
      auto Str = std::string(Buffer.data());
      Str.erase(std::remove(Str.begin(), Str.end(), '\n'), Str.end());
      ProcessorName = Str;
    }
  }
#endif

// try to detect processor name for windows
#ifdef _WIN32
  {
    // use wmic
    std::array<char, 128> Buffer{};
    const auto* Cmd = "wmic cpu get name";
    std::unique_ptr<FILE, decltype(&_pclose)> Pipe(_popen(Cmd, "r"), _pclose);
    if (!Pipe) {
      log::warn() << "Could not determine processor-name";
    }
    auto Line = 0;
    while (fgets(Buffer.data(), Buffer.size(), Pipe.get()) != nullptr) {
      if (Line != 1) {
        Line++;
        continue;
      }

      auto Str = std::string(Buffer.data());
      Str.erase(std::remove(Str.begin(), Str.end(), '\n'), Str.end());
      ProcessorName = Str;
    }
  }
#endif
}

auto ProcessorInformation::getFileAsStream(std::string const& FilePath) -> std::stringstream {
  std::ifstream File(FilePath);
  std::stringstream Ss;

  if (!File.is_open()) {
    log::trace() << "Could not open " << FilePath;
  } else {
    Ss << File.rdbuf();
    File.close();
  }

  return Ss;
}

auto ProcessorInformation::scalingGovernor() -> std::string {
  return getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").str();
}

}; // namespace firestarter