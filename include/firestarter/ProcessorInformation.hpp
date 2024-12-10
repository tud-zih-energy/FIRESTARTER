/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2024 TU Dresden, Center for Information Services and High
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

#include <cstdint>
#include <list>
#include <ostream>
#include <sstream>
#include <string>

extern "C" {
#include <hwloc.h>
}

namespace firestarter {

/// This class models the properties of a processor.
class ProcessorInformation {
public:
  explicit ProcessorInformation(std::string Architecture);
  virtual ~ProcessorInformation() = default;

  /// Getter for the clockrate in Hz
  [[nodiscard]] virtual auto clockrate() const -> uint64_t { return Clockrate; }

  /// Get the current hardware timestamp
  [[nodiscard]] virtual auto timestamp() const -> uint64_t = 0;

  /// The CPU vendor i.e., Intel or AMD.
  [[nodiscard]] virtual auto vendor() const -> std::string const& { return Vendor; }

  /// The model of the processor. With X86 this is the the string of Family, Model and Stepping.
  [[nodiscard]] virtual auto model() const -> std::string const& = 0;

  /// Print the information about this process to a stream.
  void print() const;

protected:
  /// The CPU architecture e.g., x86_64
  [[nodiscard]] auto architecture() const -> std::string const& { return Architecture; }
  /// The processor name, this includes the vendor specific name
  [[nodiscard]] virtual auto processorName() const -> std::string const& { return ProcessorName; }
  /// Getter for the list of CPU features
  [[nodiscard]] virtual auto features() const -> std::list<std::string> const& = 0;

  /// Read the scaling_govenor file of cpu0 on linux and return the contents as a string.
  [[nodiscard]] static auto scalingGovernor() -> std::string;

private:
  /// The CPU vendor i.e., Intel or AMD.
  std::string Vendor;

  /// Helper function to open a filepath and return a stringstream with its contents.
  /// \arg FilePath The file to open
  /// \returns A stringstream with the contents of the file.
  [[nodiscard]] static auto getFileAsStream(std::string const& FilePath) -> std::stringstream;

  /// The CPU architecture e.g., x86_64
  std::string Architecture;
  /// The processor name, this includes the vendor specific name
  std::string ProcessorName;
  /// Clockrate of the CPU in Hz
  uint64_t Clockrate = 0;
};

} // namespace firestarter
