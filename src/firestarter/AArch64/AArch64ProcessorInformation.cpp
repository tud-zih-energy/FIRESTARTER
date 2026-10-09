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
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace firestarter::aarch64 {

#if defined(__APPLE__)
// On Apple Silicon, MIDR_EL1 is not accessible from user space (it traps with
// SIGILL). We therefore identify the chip via sysctl instead. The MIDR fields
// are only used to construct the CpuModel (for ordering), so we return the
// Apple implementer code with a zeroed part/revision.
static auto appleCpuBrandString() -> std::string {
  char Buffer[256] = {};
  size_t Size = sizeof(Buffer);
  if (sysctlbyname("machdep.cpu.brand_string", Buffer, &Size, nullptr, 0) != 0) {
    return "Apple Silicon";
  }
  return Buffer;
}
#endif

AArch64ProcessorInformation::AArch64ProcessorInformation()
#if defined(__APPLE__)
    : ProcessorInformation("aarch64", std::make_unique<AArch64CpuFeatures>(asmjit::CpuInfo::host().features()),
                           std::make_unique<AArch64CpuModel>(0x61, 0, 0))
    , CpuInfo(asmjit::CpuInfo::host())
    , Vendor("Apple")
    , ProcessorName(appleCpuBrandString()) {
  Model = ProcessorName;
#else
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
#endif

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

// The clock rate used to convert timestamp() (CNTVCT_EL0) deltas into seconds.
// This is the generic timer counter frequency read from CNTFRQ_EL0, i.e. the
// frequency of the counter that timestamp() reads. Note that this is a
// reference frequency, not the core clock.
auto AArch64ProcessorInformation::clockrate() const -> uint64_t {
  uint64_t Freq = 0;
  __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(Freq));
  if (Freq != 0) {
    return Freq;
  }
  // Firmware left CNTFRQ_EL0 at 0; fall back to measuring the counter
  // frequency against the actually elapsed wall-clock time.
  return measureClockrate();
}

// Fallback: measure the generic timer counter frequency (Hz) by counting
// CNTVCT_EL0 ticks over a sleep interval and dividing by the actually elapsed
// steady_clock time (the nominal sleep duration is not reliable).
auto AArch64ProcessorInformation::measureClockrate() const -> uint64_t {
  constexpr auto SleepDuration = std::chrono::milliseconds(100);

  const auto WallStart = std::chrono::steady_clock::now();
  const uint64_t StartTsc = timestamp();
  std::this_thread::sleep_for(SleepDuration);
  const uint64_t EndTsc = timestamp();
  const auto WallEnd = std::chrono::steady_clock::now();

  const uint64_t TicksElapsed = EndTsc - StartTsc;
  const double SecondsElapsed = std::chrono::duration<double>(WallEnd - WallStart).count();
  if (SecondsElapsed <= 0.0) {
    return 0;
  }

  return static_cast<uint64_t>(TicksElapsed / SecondsElapsed);
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
  case 0x46:
    return "Fujitsu";
  case 0x48:
    return "HiSilicon";
  case 0x49:
    return "Infineon";
  case 0x4D:
    return "Motorola";
  case 0x4E:
    return "Nvidia";
  case 0x50:
    return "APM";
  case 0x51:
    return "Qualcomm";
  case 0x56:
    return "Marvell";
  case 0x61:
    return "Apple";
  case 0x69:
    return "Intel";
  case 0x6D:
    return "Microsoft";
  case 0xC0:
    return "Ampere";
  default:
    return "Unknown (0x" + std::to_string(Implementer) + ")";
  }
}

std::string AArch64ProcessorInformation::partNumToString(unsigned PartNum) {
  switch (PartNum) {
  case 0xD00:
    return "Foundation";
  case 0xD03:
    return "Cortex-A53";
  case 0xD04:
    return "Cortex-A35";
  case 0xD05:
    return "Cortex-A55";
  case 0xD07:
    return "Cortex-A57";
  case 0xD08:
    return "Cortex-A72";
  case 0xD09:
    return "Cortex-A73";
  case 0xD0A:
    return "Cortex-A75";
  case 0xD0B:
    return "Cortex-A76";
  case 0xD0C:
    return "Neoverse N1";
  case 0xD0D:
    return "Cortex-A77";
  case 0xD0E:
    return "Cortex-A76AE";
  case 0xD0F:
    return "AEM v8A";
  case 0xD40:
    return "Neoverse V1";
  case 0xD41:
    return "Cortex-A78";
  case 0xD42:
    return "Cortex-A78AE";
  case 0xD44:
    return "Cortex-X1";
  case 0xD46:
    return "Cortex-A510";
  case 0xD47:
    return "Cortex-A710";
  case 0xD48:
    return "Cortex-X2";
  case 0xD49:
    return "Neoverse N2";
  case 0xD4B:
    return "Cortex-A78C";
  case 0xD4C:
    return "Cortex-X1C";
  case 0xD4D:
    return "Cortex-A715";
  case 0xD4E:
    return "Cortex-X3";
  case 0xD4F:
    return "Neoverse V2";
  case 0xD80:
    return "Cortex-A520";
  case 0xD81:
    return "Cortex-A720";
  case 0xD82:
    return "Cortex-X4";
  case 0xD83:
    return "Neoverse V3AE";
  case 0xD84:
    return "Neoverse V3";
  case 0xD85:
    return "Cortex-X925";
  case 0xD87:
    return "Cortex-A725";
  case 0xD88:
    return "Cortex-A520AE";
  case 0xD89:
    return "Cortex-A720AE";
  case 0xD8A:
    return "C1-Nano";
  case 0xD8B:
    return "C1-Pro";
  case 0xD8C:
    return "C1-Ultra";
  case 0xD8E:
    return "Neoverse N3";
  case 0xD90:
    return "C1-Premium";
  default:
    return "Unknown (0x" + std::to_string(PartNum) + ")";
  }
}

} // namespace firestarter::aarch64
