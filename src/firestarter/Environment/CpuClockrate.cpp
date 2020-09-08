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

#include <firestarter/Environment/Environment.hpp>
#include <firestarter/Logging/Log.hpp>

#include <regex>

using namespace firestarter::environment;

std::stringstream Environment::getScalingGovernor(void) {
  return this->getFileAsStream(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
}

int Environment::getCpuClockrate(void) {
  std::string clockrate = "0";

  auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
  if (procCpuinfo.str().empty()) {
    return EXIT_FAILURE;
  }

  std::string line;

  while (std::getline(procCpuinfo, line, '\n')) {
    const std::regex cpuMhzRe("^cpu MHz.*:\\s*(.*)\\s*$");
    std::smatch m;

    if (std::regex_match(line, m, cpuMhzRe)) {
      clockrate = m[1].str();
    }
  }

  if (clockrate == "0") {
    firestarter::log::warn() << "Can't determine clockrate from /proc/cpuinfo";
  }

  firestarter::log::trace() << "Clockrate from /proc/cpuinfo is " << clockrate;

  auto governor = this->getScalingGovernor().str();
  if (governor.empty()) {
    return EXIT_FAILURE;
  }

  auto scalingCurFreq =
      this->getFileAsStream(
              "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
          .str();
  auto cpuinfoCurFreq =
      this->getFileAsStream(
              "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq")
          .str();
  auto scalingMaxFreq =
      this->getFileAsStream(
              "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
          .str();
  auto cpuinfoMaxFreq =
      this->getFileAsStream(
              "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
          .str();

  if (governor.compare("performance") || governor.compare("powersave")) {
    if (scalingCurFreq.empty()) {
      if (!cpuinfoCurFreq.empty()) {
        clockrate = cpuinfoCurFreq;
      }
    } else {
      clockrate = scalingCurFreq;
    }
  } else {
    if (scalingMaxFreq.empty()) {
      if (!cpuinfoMaxFreq.empty()) {
        clockrate = cpuinfoMaxFreq;
      }
    } else {
      clockrate = scalingMaxFreq;
    }
  }

  this->_clockrate = 1000 * std::stoi(clockrate);

  return EXIT_SUCCESS;
}
