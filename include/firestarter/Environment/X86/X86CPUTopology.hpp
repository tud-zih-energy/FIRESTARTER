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

#include <asmjit/asmjit.h>

#include "../CPUTopology.hpp"

namespace firestarter::environment::x86 {

class X86CPUTopology final : public CPUTopology {
public:
  X86CPUTopology();

  friend auto operator<<(std::ostream& Stream, X86CPUTopology const& CpuTopology) -> std::ostream&;

  [[nodiscard]] auto features() const -> std::list<std::string> const& override { return this->FeatureList; }
  [[nodiscard]] auto featuresAsmjit() const -> const asmjit::CpuFeatures& { return this->CpuInfo.features(); }

  [[nodiscard]] auto clockrate() const -> uint64_t override;

  [[nodiscard]] auto timestamp() const -> uint64_t override;

  [[nodiscard]] auto familyId() const -> unsigned { return this->CpuInfo.familyId(); }
  [[nodiscard]] auto modelId() const -> unsigned { return this->CpuInfo.modelId(); }
  [[nodiscard]] auto stepping() const -> unsigned { return this->CpuInfo.stepping(); }

private:
  [[nodiscard]] auto hasRdtsc() const -> bool { return this->HasRdtsc; }
  [[nodiscard]] auto hasInvariantRdtsc() const -> bool { return this->HasInvariantRdtsc; }
  static void cpuid(uint64_t* Rax, uint64_t* Rbx, uint64_t* Rcx, uint64_t* Rdx);

  asmjit::CpuInfo CpuInfo;
  std::list<std::string> FeatureList;

  bool HasRdtsc;
  bool HasInvariantRdtsc;
};

inline auto operator<<(std::ostream& Stream, X86CPUTopology const& CpuTopology) -> std::ostream& {
  return CpuTopology.print(Stream);
}

} // namespace firestarter::environment::x86