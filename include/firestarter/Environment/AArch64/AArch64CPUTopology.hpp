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

#include <firestarter/Environment/CPUTopology.hpp>

#include <asmjit/asmjit.h>

namespace firestarter::environment::aarch64 {

class AArch64CPUTopology final : public CPUTopology {
public:
  AArch64CPUTopology();

  friend std::ostream &operator<<(std::ostream &stream,
                                  AArch64CPUTopology const &cpuTopology);

  std::list<std::string> const &features() const override {
    return this->featureList;
  }
  const asmjit::CpuFeatures& featuresAsmjit() const{
    return this->cpuInfo.features();
  }

  std::string const &vendor() const override { return this->_vendor; }
  std::string const &model() const override { return this->_model; }

  unsigned long long clockrate() const override;

  unsigned long long timestamp() const override;

  unsigned familyId() const { return this->cpuInfo.familyId(); }
  unsigned modelId() const { return this->cpuInfo.modelId(); }
  unsigned stepping() const { return this->cpuInfo.stepping(); }

private:

  asmjit::CpuInfo cpuInfo;
  std::list<std::string> featureList;

  std::string _vendor;
  std::string _model;
};

inline std::ostream &operator<<(std::ostream &stream,
                                AArch64CPUTopology const &cpuTopology) {
  return cpuTopology.print(stream);
}

} // namespace firestarter::environment::aarch64
