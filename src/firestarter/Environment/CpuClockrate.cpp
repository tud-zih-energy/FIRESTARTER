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

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

using namespace firestarter::environment;

std::unique_ptr<llvm::MemoryBuffer> Environment::getScalingGovernor(void) {
  return this->getFileAsStream(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
}

int Environment::getCpuClockrate(void) {
  auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
  if (nullptr == procCpuinfo) {
    return EXIT_FAILURE;
  }

  llvm::SmallVector<llvm::StringRef, 512> lines;
  llvm::SmallVector<llvm::StringRef, 2> clockrateVector;
  procCpuinfo->getBuffer().split(lines, "\n");

  for (size_t i = 0; i < lines.size(); i++) {
    if (lines[i].startswith("cpu MHz")) {
      lines[i].split(clockrateVector, ':');
      break;
    }
  }

  std::string clockrate = 0;

  if (clockrateVector.size() == 2) {
    clockrate = clockrateVector[1].str();
    clockrate.erase(0, 1);
  } else {
    firestarter::log::warn() << "Can't determine clockrate from /proc/cpuinfo";
  }

  std::unique_ptr<llvm::MemoryBuffer> scalingGovernor;
  if (nullptr == (scalingGovernor = this->getScalingGovernor())) {
    return EXIT_FAILURE;
  }

  std::string governor = scalingGovernor->getBuffer().str();

  auto scalingCurFreq = this->getFileAsStream(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
  auto cpuinfoCurFreq = this->getFileAsStream(
      "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq");
  auto scalingMaxFreq = this->getFileAsStream(
      "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
  auto cpuinfoMaxFreq = this->getFileAsStream(
      "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");

  if (governor.compare("performance") || governor.compare("powersave")) {
    if (nullptr == scalingCurFreq) {
      if (nullptr != cpuinfoCurFreq) {
        clockrate = cpuinfoCurFreq->getBuffer().str();
      }
    } else {
      clockrate = scalingCurFreq->getBuffer().str();
    }
  } else {
    if (nullptr == scalingMaxFreq) {
      if (nullptr != cpuinfoMaxFreq) {
        clockrate = cpuinfoMaxFreq->getBuffer().str();
      }
    } else {
      clockrate = scalingMaxFreq->getBuffer().str();
    }
  }

  this->_clockrate = 1000 * std::stoi(clockrate);

  return EXIT_SUCCESS;
}
