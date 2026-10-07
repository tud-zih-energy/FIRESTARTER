/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/AArch64/AArch64ProcessorInformation.hpp"
#include "firestarter/AArch64/AArch64CpuFeatures.hpp"
#include "firestarter/AArch64/AArch64CpuModel.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/ProcessorInformation.hpp"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <sstream>

namespace firestarter::aarch64 {

AArch64ProcessorInformation::AArch64ProcessorInformation()
    : ProcessorInformation("aarch64", std::make_unique<AArch64CpuFeatures>(asmjit::CpuInfo::host().features()),
                           std::make_unique<AArch64CpuModel>(midrImplementer(), midrPartNum(), midrRevision()))
    , CpuInfo(asmjit::CpuInfo::host())
    , Vendor(implementerToString(midrImplementer()))
    , ProcessorName(partNumToString(midrPartNum())) {

  {
    std::stringstream Ss;
    Ss << "AArch64 Processor (Implementer: 0x" << std::hex << std::uppercase << midrImplementer() << ", Part: 0x"
       << midrPartNum() << ", Rev: r" << std::dec << midrRevision() << ")";
    Model = Ss.str();
  }

  // Build the feature list from asmjit
  for (auto FeatureId = 0; FeatureId <= (int)asmjit::CpuFeatures::ARM::Id::kMaxValue; FeatureId++) {
    if (!CpuInfo.hasFeature(FeatureId)) {
      continue;
    }

    asmjit::String Sb;
    auto Error = asmjit::Formatter::formatFeature(Sb, CpuInfo.arch(), FeatureId);
    if (Error != asmjit::ErrorCode::kErrorOk) {
      log::warn() << "Formatting cpu features got asmjit error: " << Error;
    }

    FeatureList.emplace_back(Sb.data());
  }
}

// AArch64 does not have a direct equivalent of x86's TSC-based clockrate measurement.
// Return 0 to indicate unknown clockrate.
auto AArch64ProcessorInformation::clockrate() const -> uint64_t { return 0; }

// AArch64 timestamp using the generic timer counter (CNTVCT_EL0).
auto AArch64ProcessorInformation::timestamp() const -> uint64_t {
  uint64_t Tsc = 0;
  __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(Tsc));
  return Tsc;
}

unsigned AArch64ProcessorInformation::midrImplementer() {
  uint64_t Midr = 0;
  __asm__ __volatile__("mrs %0, midr_el1" : "=r"(Midr));
  return static_cast<unsigned>((Midr >> 24) & 0xFF);
}

unsigned AArch64ProcessorInformation::midrPartNum() {
  uint64_t Midr = 0;
  __asm__ __volatile__("mrs %0, midr_el1" : "=r"(Midr));
  return static_cast<unsigned>((Midr >> 4) & 0xFFF);
}

unsigned AArch64ProcessorInformation::midrRevision() {
  uint64_t Midr = 0;
  __asm__ __volatile__("mrs %0, midr_el1" : "=r"(Midr));
  return static_cast<unsigned>(Midr & 0xF);
}

std::string AArch64ProcessorInformation::implementerToString(unsigned Implementer) {
  switch (Implementer) {
  case 0x41:
    return "ARM";
  case 0x42:
    return "Broadcom";
  case 0x43:
    return "Cavium";
  case 0x44:
    return "DEC";
  case 0x48:
    return "Fujitsu";
  case 0x4E:
    return "Nvidia";
  case 0x50:
    return "APM";
  case 0x51:
    return "Qualcomm";
  case 0x53:
    return "Samsung";
  case 0x56:
    return "HiSilicon";
  case 0x61:
    return "Apple";
  case 0x69:
    return "Intel";
  case 0x70:
    return "NXP";
  case 0xC0:
    return "Ampere";
  default:
    return "Unknown (0x" + std::to_string(Implementer) + ")";
  }
}

std::string AArch64ProcessorInformation::partNumToString(unsigned PartNum) {
  switch (PartNum) {
  case 0x000:
    return "Cortex-A53";
  case 0x001:
    return "Cortex-A55 / Neoverse N1";
  case 0x002:
    return "Neoverse V1";
  case 0x003:
    return "Cortex-A35";
  case 0x004:
    return "Cortex-A34";
  case 0x005:
    return "Cortex-A65";
  case 0x006:
    return "Cortex-A55";
  case 0x008:
    return "Cortex-A75";
  case 0x009:
    return "Cortex-A76";
  case 0x00A:
    return "Cortex-R52";
  case 0x00B:
    return "Cortex-M55";
  case 0x00C:
    return "Cortex-A75";
  case 0x00D:
    return "Cortex-A72";
  case 0x00E:
    return "Cortex-A73";
  case 0x00F:
    return "Cortex-A57";
  case 0x010:
    return "Cortex-R52+";
  case 0x011:
    return "Cortex-M85";
  case 0x013:
    return "Cortex-A76";
  case 0x014:
    return "Cortex-A78";
  case 0x015:
    return "Cortex-A77";
  case 0x016:
    return "Cortex-A76 AE";
  case 0x017:
    return "Cortex-A710";
  case 0x018:
    return "Cortex-A715";
  case 0x019:
    return "Cortex-A510";
  case 0x01A:
    return "Cortex-A510";
  case 0x01B:
    return "Cortex-R82";
  case 0x01C:
    return "Cortex-M85";
  case 0x01D:
    return "Cortex-M55";
  case 0x01E:
    return "Cortex-X1";
  case 0x01F:
    return "Cortex-X2";
  case 0x020:
    return "Neoverse V2";
  case 0x021:
    return "Cortex-X2";
  case 0x022:
    return "Neoverse N2";
  case 0x023:
    return "Cortex-X3";
  case 0x024:
    return "Cortex-X4";
  case 0x025:
    return "Cortex-A520";
  case 0x026:
    return "Cortex-A720";
  case 0x027:
    return "Cortex-X4";
  case 0xD0C:
    return "Neoverse N1";
  case 0xD0D:
    return "Neoverse V1";
  case 0xD49:
    return "Neoverse N2";
  case 0xD4F:
    return "Neoverse V2";
  default:
    return "Unknown (0x" + std::to_string(PartNum) + ")";
  }
}

} // namespace firestarter::aarch64
