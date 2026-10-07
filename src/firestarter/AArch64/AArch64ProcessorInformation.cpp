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
    : ProcessorInformation(
          "aarch64", std::make_unique<AArch64CpuFeatures>(asmjit::CpuInfo::host().features()),
          std::make_unique<AArch64CpuModel>(midrImplementer(), midrPartNum(), midrRevision()))
    , CpuInfo(asmjit::CpuInfo::host())
    , Vendor(CpuInfo.vendor()) {

  {
    std::stringstream Ss;
    Ss << "AArch64 Processor (Implementer: 0x" << std::hex << std::uppercase
       << midrImplementer() << ", Part: 0x" << midrPartNum() << ", Rev: r" << std::dec
       << midrRevision() << ")";
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
auto AArch64ProcessorInformation::clockrate() const -> uint64_t {
  return 0;
}

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

} // namespace firestarter::aarch64
