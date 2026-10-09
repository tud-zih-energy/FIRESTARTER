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

#pragma once

#include "firestarter/ProcessorInformation.hpp"

#include <asmjit/asmjit.h>

namespace firestarter::aarch64 {

/// This class models the properties of an AArch64 processor.
class AArch64ProcessorInformation final : public ProcessorInformation {
public:
  AArch64ProcessorInformation();

  /// Getter for the list of CPU features
  [[nodiscard]] auto features() const -> std::list<std::string> const& override { return this->FeatureList; }

  /// Getter for the clockrate in Hz
  [[nodiscard]] auto clockrate() const -> uint64_t override;

  /// Get the current hardware timestamp
  [[nodiscard]] auto timestamp() const -> uint64_t override;

  /// The CPU vendor derived from MIDR_EL1 implementer field.
  [[nodiscard]] auto vendor() const -> std::string const& final { return Vendor; }
  /// Get the string containing the processor model.
  [[nodiscard]] auto model() const -> std::string const& final { return Model; }
  /// The processor name derived from MIDR_EL1 part number.
  [[nodiscard]] auto processorName() const -> std::string const& override { return ProcessorName; }

private:
  /// Read the implementer field from MIDR_EL1 (bits [31:24])
  static unsigned midrImplementer();
  /// Read the part number field from MIDR_EL1 (bits [15:4])
  static unsigned midrPartNum();
  /// Read the revision field from MIDR_EL1 (bits [3:0])
  static unsigned midrRevision();
  /// Convert an implementer code to a vendor string.
  static std::string implementerToString(unsigned Implementer);
  /// Convert a part number to a processor name string.
  static std::string partNumToString(unsigned PartNum);
  /// Measure the current clockrate in Hz using the generic timer counter.
  auto measureClockrate() const -> uint64_t;

  /// The asmjit CpuInfo for the current processor
  asmjit::CpuInfo CpuInfo;
  /// The list of cpu features that are supported by the current processor
  std::list<std::string> FeatureList;

  /// The CPU vendor derived from MIDR_EL1 implementer field.
  std::string Vendor;
  /// The processor name derived from MIDR_EL1 part number.
  std::string ProcessorName;
  /// Model string for the AArch64 processor
  std::string Model;
};

} // namespace firestarter::aarch64
