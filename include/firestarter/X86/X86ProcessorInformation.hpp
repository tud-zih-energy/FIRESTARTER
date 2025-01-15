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

#pragma once

#include "firestarter/ProcessorInformation.hpp"

#include <asmjit/asmjit.h>

namespace firestarter::x86 {

/// This class models the properties of a x86_64 processor.
class X86ProcessorInformation final : public ProcessorInformation {
public:
  X86ProcessorInformation();

  /// Getter for the list of CPU features
  [[nodiscard]] auto features() const -> std::list<std::string> const& override { return this->FeatureList; }

  /// Getter for the clockrate in Hz
  [[nodiscard]] auto clockrate() const -> uint64_t override;

  /// Get the current hardware timestamp
  [[nodiscard]] auto timestamp() const -> uint64_t override;

  /// The CPU vendor i.e., Intel or AMD.
  [[nodiscard]] auto vendor() const -> std::string const& final { return Vendor; }
  /// Get the string containing family, model and stepping ids.
  [[nodiscard]] auto model() const -> std::string const& final { return Model; }

private:
  /// Does this processor support timestamp counters
  [[nodiscard]] auto hasRdtsc() const -> bool { return this->HasRdtsc; }
  /// Does this processor have invariant timestamp counters
  [[nodiscard]] auto hasInvariantRdtsc() const -> bool { return this->HasInvariantRdtsc; }

  /// A wrapper to the cpuid call to keep a consitent interface between Windows and other platforms.
  static void cpuid(uint64_t* Rax, uint64_t* Rbx, uint64_t* Rcx, uint64_t* Rdx);

  /// The asmjit CpuInfo for the current processor
  asmjit::CpuInfo CpuInfo;
  /// The list of cpufeatures that are supported by the current processpr
  std::list<std::string> FeatureList;

  /// Does this processor support timestamp counters
  bool HasRdtsc;
  /// Does this processor have invariant timestamp counters
  bool HasInvariantRdtsc;

  /// The CPU vendor i.e., Intel or AMD.
  std::string Vendor;
  /// Model string containing family, model and stepping ids.
  std::string Model;
};

} // namespace firestarter::x86