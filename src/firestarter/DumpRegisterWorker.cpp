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

#ifdef FIRESTARTER_DEBUG_FEATURES

#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#include <fstream>
#include <sstream>
#include <thread>

namespace {
auto hammingDistance(uint64_t X, uint64_t Y) -> unsigned {
  unsigned Dist = 0;

  for (uint64_t Val = X ^ Y; Val > 0; Val >>= 1) {
    Dist += Val & 1;
  }

  return Dist;
}

auto registerNameBySize(unsigned RegisterSize) -> std::string {
  switch (RegisterSize) {
  case 2:
    return "xmm";
  case 4:
    return "ymm";
  case 8:
    return "zmm";
  default:
    return "unknown";
  }
}
} // namespace

namespace firestarter {

auto Firestarter::initDumpRegisterWorker(std::chrono::seconds DumpTimeDelta, const std::string& DumpFilePath) -> int {
  // Create the data for the worker thread. The thread will dump the register contents periodically and calculate the
  // hamming distance between dumps.
  auto Data = std::make_unique<DumpRegisterWorkerData>(this->LoadThreads.begin()->second, DumpTimeDelta, DumpFilePath);

  // Spawn the thread.
  DumpRegisterWorkerThread = std::thread(Firestarter::dumpRegisterWorker, std::move(Data));

  return EXIT_SUCCESS;
}

void Firestarter::joinDumpRegisterWorker() { this->DumpRegisterWorkerThread.join(); }

void Firestarter::dumpRegisterWorker(std::unique_ptr<DumpRegisterWorkerData> Data) {

  pthread_setname_np(pthread_self(), "DumpRegWorker");

  const auto RegisterCount = Data->LoadWorkerDataPtr->config().payload().registerCount();
  const auto RegisterSize = Data->LoadWorkerDataPtr->config().payload().registerSize();
  const auto Offset = RegisterCount * RegisterSize;
  const std::string RegisterPrefix = registerNameBySize(RegisterSize);

  auto& DumpRegisterStructRef = Data->LoadWorkerDataPtr->Memory->ExtraVars.Drs;
  auto& DumpVar = DumpRegisterStructRef.DumpVar;
  // memory of simd variables is before the padding
  const auto* DumpMemAddr = static_cast<volatile uint64_t*>(DumpRegisterStructRef.Padding) - Offset;

  // allocate continous memory that fits the register contents
  auto Last = std::vector<uint64_t>(Offset);

  std::stringstream DumpFilePath;
  DumpFilePath << Data->DumpFilePath;
#if defined(__MINGW32__) || defined(__MINGW64__)
  dumpFilePath << "\\";
#else
  DumpFilePath << "/";
#endif
  DumpFilePath << "hamming_distance.csv";
  auto DumpFile = std::ofstream(DumpFilePath.str());

  // dump the header to the csv file
  DumpFile << "total_hamming_distance,";
  for (auto I = 0U; I < RegisterCount; I++) {
    for (auto J = 0U; J < RegisterSize; J++) {
      DumpFile << RegisterPrefix << I << "[" << J << "]";

      if (J != RegisterSize - 1) {
        DumpFile << ",";
      }
    }

    if (I != RegisterCount - 1) {
      DumpFile << ",";
    }
  }
  DumpFile << '\n' << std::flush;

  // do not output the hamming distance for the first run
  bool SkipFirst = true;

  // continue until stop and dump the registers every data->dumpTimeDelta
  // seconds
  for (; Data->LoadWorkerDataPtr->LoadVar != LoadThreadWorkType::LoadStop;) {
    // signal the thread to dump its largest SIMD registers
    DumpVar = DumpVariable::Start;
    __asm__ __volatile__("mfence;");
    while (DumpVar == DumpVariable::Start) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto Current = std::vector<uint64_t>(Offset);
    // copy the register content to minimize the interruption of the load worker
    std::memcpy(Current.data(), (void*)DumpMemAddr, Current.size() * sizeof(decltype(Current)::value_type));

    // skip the first output, as we first have to get some valid values for last
    if (!SkipFirst) {
      // calculate the total hamming distance
      auto TotalHammingDistance = 0U;
      for (auto I = 0U; I < RegisterCount * RegisterSize; I++) {
        TotalHammingDistance += hammingDistance(Current[I], Last[I]);
      }

      DumpFile << TotalHammingDistance << ",";

      // dump the hamming distance of each double (last, current) pair
      for (int I = RegisterCount - 1; I >= 0; I--) {
        // auto registerNum = registerCount - 1 - i;

        for (auto J = 0U; J < RegisterSize; J++) {
          auto Index = (RegisterSize * I) + J;
          auto Hd = static_cast<uint64_t>(hammingDistance(Current[Index], Last[Index]));

          DumpFile << Hd;
          if (J != RegisterSize - 1) {
            DumpFile << ",";
          }
        }

        if (I != 0) {
          DumpFile << ",";
        }
      }

      DumpFile << '\n' << std::flush;
    } else {
      SkipFirst = false;
    }

    Last = std::move(Current);

    std::this_thread::sleep_for(std::chrono::seconds(Data->DumpTimeDelta));
  }

  DumpFile.close();
}

} // namespace firestarter

#endif