/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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

#include "firestarter/Measurement/MetricInterface.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/// The wrapper for the C interface to the RaplMetric metric.
class RaplMetricData {
private:
  /// Datastructure to hold the path of the sysfs rapl entry, the last reading (improtant to detect overflows), the
  /// counter of the number of overflows and the maximum value that the reading will have.
  struct ReaderDef {
  private:
    /// The path to the sysfs root of this metric
    std::string Path;
    /// The name of this metric
    std::string Name;
    /// The last reading of this counter
    int64_t LastReading;
    /// The number of times the counter overflowed
    int64_t Overflow = 0;
    /// The max reading of the counter as specified in the 'max_energy_range_uj' sysfs entry.
    int64_t Max;

    /// Recusively get the name of the RAPL metric, e.g. package-0-dram. The dram metric is in a directory where the
    /// device is set to the package domain.
    /// \arg Path The path to the current sysfs root
    /// \arg Name The currently parsed name of the metric
    /// \returns The recursively parsed name of the metric.
    // NOLINTNEXTLINE(misc-no-recursion)
    static auto getNameRecursive(const std::string& Path, const std::string& Name = "") -> std::string {
      // If the current path contains a name, prepend it to the string
      std::string ParsedName;

      {
        std::ifstream NameStream(Path + "/name");
        if (!NameStream.good()) {
          // else return the name
          return Name;
        }

        std::getline(NameStream, ParsedName);
      }

      std::string DevicePath;

      {
        std::ifstream DeviceStream(Path + "/device");
        if (!DeviceStream.good()) {
          // else return the name
          return Name;
        }

        std::getline(DeviceStream, DevicePath);
      }

      // Got to the device that is set in the path and call the function on it.
      return getNameRecursive(DevicePath, ParsedName + "-" + Name);
    };

  public:
    ReaderDef() = delete;

    /// \arg Path The sysfs root path of this metric
    explicit ReaderDef(std::string Path)
        : Path(std::move(Path)) {

      Name = getNameRecursive(Path);
      if (Name.empty()) {
        // an error opening the file occured
        throw std::invalid_argument("Not a valid metric");
      }

      std::stringstream EnergyUjPath;
      EnergyUjPath << Path << "/energy_uj";
      std::ifstream EnergyReadingStream(EnergyUjPath.str());
      if (!EnergyReadingStream.good()) {
        throw std::runtime_error("Could not read energy_uj");
      }

      std::stringstream MaxEnergyUjRangePath;
      MaxEnergyUjRangePath << Path << "/max_energy_range_uj";
      std::ifstream MaxEnergyReadingStream(MaxEnergyUjRangePath.str());
      if (!MaxEnergyReadingStream.good()) {
        throw std::runtime_error("Could not read max_energy_range_uj");
      }

      std::string Buffer;

      std::getline(EnergyReadingStream, Buffer);
      LastReading = std::stoll(Buffer);

      std::getline(MaxEnergyReadingStream, Buffer);
      Max = std::stoll(Buffer);
    };

    /// Get the name of this metric
    auto name() -> auto& { return Name; }

    /// Read the RAPL counter and update the internal state
    void read() {
      std::string Buffer;

      std::stringstream EnergyUjPath;
      EnergyUjPath << Path << "/energy_uj";
      std::ifstream EnergyReadingStream(EnergyUjPath.str());
      std::getline(EnergyReadingStream, Buffer);
      const auto Reading = std::stoll(Buffer);

      if (Reading < LastReading) {
        Overflow += 1;
      }

      LastReading = Reading;
    };

    /// Get the converted value of the last reading.
    /// \returns The value of the RAPL counter in joule
    [[nodiscard]] auto lastReading() const -> double {
      return 1.0E-6 * static_cast<double>((Overflow * Max) + LastReading);
    };
  };

  /// The path to the sysfs rapl entries
  static constexpr const char* RaplPath = "/sys/class/powercap";

  /// The error string of this metric
  std::string ErrorString;

  /// The vector of readers that hold the path and read values from the sysfs rapl
  std::vector<std::shared_ptr<ReaderDef>> Readers;

  /// The vector of reader that is used to accumulate the overall metric value
  std::vector<std::shared_ptr<ReaderDef>> AccumulateReaders;

  /// The null terminated vector of submetric names.
  std::vector<char*> SubmetricNames = {nullptr};

  RaplMetricData() = default;

public:
  RaplMetricData(RaplMetricData const&) = delete;
  void operator=(RaplMetricData const&) = delete;

  /// Get the instance of this metric
  static auto instance() -> RaplMetricData& {
    static RaplMetricData Instance;
    return Instance;
  }

  /// Deinit the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto fini() -> int32_t;

  /// Init the metric.
  /// \returns EXIT_SUCCESS on success.
  static auto init() -> int32_t;

  /// Get a vector of submetric names. This is required to know the name of a submetric that is just described via an
  /// index throughout this metric interface.
  /// \returns The NULL terminated array of submetric names (char *)
  static auto getSubmetricNames() -> char** { return instance().SubmetricNames.data(); }

  /// Get a reading of the sysfs-powercap-rapl metric.
  /// \arg Value The pointer to which the value will be saved.
  /// \returns EXIT_SUCCESS if we got a new value.
  static auto getReading(double* Value) -> int32_t;

  /// Get error in case return code not being EXIT_SUCCESS.
  /// \returns The error string.
  static auto getError() -> const char*;

  /// This function should be called every 30s. It will make shure that we do not miss an overflow of a counter and
  /// therefore get a wrong reading.
  static void callback();
};

/// This metric provides power measurements through the RAPL interface. Either psys measurement is choosen or if this is
/// not available the sum of packages and drams.
inline static MetricInterface RaplMetric{
    /*Name=*/"sysfs-powercap-rapl",
    /*Type=*/
    {/*Absolute=*/0, /*Accumalative=*/1, /*DivideByThreadCount=*/0, /*InsertCallback=*/0, /*IgnoreStartStopDelta=*/0,
     /*Reserved=*/0},
    /*Unit=*/"J",
    /*CallbackTime=*/30000000,
    /*Callback=*/RaplMetricData::callback,
    /*Init=*/RaplMetricData::init,
    /*Fini=*/RaplMetricData::fini,
    /*GetSubmetricNames=*/
    RaplMetricData::getSubmetricNames,
    /*GetReading=*/RaplMetricData::getReading,
    /*GetError=*/RaplMetricData::getError,
    /*RegisterInsertCallback=*/nullptr,
};