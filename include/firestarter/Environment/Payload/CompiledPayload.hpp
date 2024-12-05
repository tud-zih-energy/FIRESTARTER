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

#include "firestarter/Constants.hpp"
#include "firestarter/Environment/Payload/PayloadStats.hpp"

#include <chrono>
#include <memory>
#include <utility>

namespace firestarter::environment::payload {

class Payload;

/// This class represents a payload that can be executed. It is created by calling compilePayload of the payload class
/// with specific settings. It contains a reference to the init and low load functions (which do not change with payload
/// settings) and the high load function which changes based on the settings. The stats of the high load function (nb.
/// of flops, bytes of memory accessed and instructions) can also be retrieved.
class CompiledPayload {
public:
  CompiledPayload() = delete;
  virtual ~CompiledPayload() = default;

  /// A unique ptr for the CompiledPayload with a custom deleter.
  using UniquePtr = std::unique_ptr<CompiledPayload, void (*)(CompiledPayload*)>;

  using HighLoadFunctionPtr = uint64_t (*)(double*, volatile LoadThreadWorkType*, uint64_t);

  /// Getter for the stats of the high load function of the compiled payload
  [[nodiscard]] auto stats() const -> const PayloadStats& { return Stats; };

  /// Function to initialize the memory used by the high load function.
  /// \arg MemoryAddr The pointer to the memory.
  /// \arg BufferSize The number of doubles that is allocated in MemoryAddr.
  void init(double* MemoryAddr, uint64_t BufferSize);

  /// Function to produce a low load on the cpu.
  /// \arg LoadVar The variable that controls the load. If this variable changes from LoadThreadWorkType::LowLoad to
  /// something else this function will return.
  /// \arg Period The period of the low/high load switching. This function may sleep a fraction of this period.
  void lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period);

  /// Function to produce high load on the cpu.
  /// \arg MemoryAddr The pointer to the memory.
  /// \arg LoadVar The variable that controls the load. If this variable changes from LoadThreadWorkType::LoadHigh to
  /// something else this function will return.
  /// \arg Iterations The current iteration counter. This number will be incremented for every iteration of the high
  /// load loop.
  /// \returns The iteration counter passed into this function plus the number of iteration of the high load loop.
  [[nodiscard]] auto highLoadFunction(double* MemoryAddr, volatile LoadThreadWorkType& LoadVar, uint64_t Iterations)
      -> uint64_t {
    return HighLoadFunction(MemoryAddr, &LoadVar, Iterations);
  }

protected:
  /// Constructor for the CompiledPayload.
  /// \arg Stats The stats of the high load function from the payload.
  /// \arg PayloadPtr A unique pointer to the payload class to allow calling the init and low load functions which do
  /// not change based on different payload settings.
  /// \arg HighLoadFunction The pointer to the compiled high load function.
  CompiledPayload(const PayloadStats& Stats, std::unique_ptr<Payload>&& PayloadPtr,
                  HighLoadFunctionPtr HighLoadFunction)
      : Stats(Stats)
      , PayloadPtr(std::move(PayloadPtr))
      , HighLoadFunction(HighLoadFunction) {}

  /// Getter for the pointer to the high load function. We need to access this pointer directly to free the associated
  /// memory from asmjit.
  [[nodiscard]] auto highLoadFunctionPtr() -> HighLoadFunctionPtr { return HighLoadFunction; }

private:
  /// The stats of the compiled payload.
  PayloadStats Stats;
  /// The pointer to the payload class to allow calling the init and low load functions which do not change based on
  /// different payload settings.
  std::unique_ptr<Payload> PayloadPtr;
  /// The pointer to the compiled high load function.
  HighLoadFunctionPtr HighLoadFunction;
};

} // namespace firestarter::environment::payload