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

#include "firestarter/CpuFeatures.hpp"

#include <asmjit/x86.h>
#include <cassert>
#include <stdexcept>

namespace firestarter::x86 {

/// This class models the cpu features on the x86_64 platform.
class X86CpuFeatures : public CpuFeatures {
private:
  /// This list contains the features (cpu extenstions) that are requied to execute the payload.
  asmjit::CpuFeatures AsmjitFeatures;

public:
  X86CpuFeatures() = default;
  explicit X86CpuFeatures(const asmjit::CpuFeatures& FeatureRequests)
      : AsmjitFeatures(FeatureRequests) {}

  explicit operator const asmjit::CpuFeatures&() const noexcept { return AsmjitFeatures; }

  [[nodiscard]] auto add(asmjit::CpuFeatures::X86::Id Id) -> X86CpuFeatures& {
    AsmjitFeatures.add(Id);
    return *this;
  }

  /// Check if this class has all features which are given in the argument.
  /// \arg Features The features which should be check if they are available.
  /// \returns true if this class has all features given in the argument.
  [[nodiscard]] auto hasAll(const CpuFeatures& Features) const -> bool override {
    const auto* DerivedFeatures = dynamic_cast<const X86CpuFeatures*>(&Features);
    if (!DerivedFeatures) {
      throw std::runtime_error("Features is not of the correct type X86CpuFeatures");
    }

    return AsmjitFeatures.hasAll(DerivedFeatures->AsmjitFeatures);
  }
};

} // namespace firestarter::x86
