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

#ifdef FIRESTARTER_DEBUG_FEATURES

#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#include <fstream>
#include <sstream>
#include <thread>

using namespace firestarter;

namespace {
static unsigned hammingDistance(unsigned long long x, unsigned long long y) {
  unsigned dist = 0;

  for (unsigned long long val = x ^ y; val > 0; val >>= 1) {
    dist += val & 1;
  }

  return dist;
}

static std::string registerNameBySize(unsigned registerSize) {
  switch (registerSize) {
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

int Firestarter::initDumpRegisterWorker(std::chrono::seconds dumpTimeDelta,
                                        std::string dumpFilePath) {

  auto data = std::make_unique<DumpRegisterWorkerData>(
      this->loadThreads.begin()->second, dumpTimeDelta, dumpFilePath);

  this->dumpRegisterWorkerThread =
      std::thread(Firestarter::dumpRegisterWorker, std::move(data));

  return EXIT_SUCCESS;
}

void Firestarter::joinDumpRegisterWorker() {
  this->dumpRegisterWorkerThread.join();
}

void Firestarter::dumpRegisterWorker(
    std::unique_ptr<DumpRegisterWorkerData> data) {

  pthread_setname_np(pthread_self(), "DumpRegWorker");

  int registerCount = data->loadWorkerData->config().payload().registerCount();
  int registerSize = data->loadWorkerData->config().payload().registerSize();
  std::string registerPrefix = registerNameBySize(registerSize);
  auto offset = sizeof(DumpRegisterStruct) / sizeof(unsigned long long);

  auto dumpRegisterStruct = reinterpret_cast<DumpRegisterStruct *>(
      data->loadWorkerData->addrMem - offset);

  auto dumpVar = reinterpret_cast<volatile unsigned long long *>(
      &dumpRegisterStruct->dumpVar);
  // memory of simd variables is before the padding
  volatile unsigned long long *dumpMemAddr =
      dumpRegisterStruct->padding - registerCount * registerSize;

  // TODO: handle error
  // TODO: maybe use aligned_malloc to make memcpy more efficient and don't
  // interrupt the workload as much?
  unsigned long long *last = reinterpret_cast<unsigned long long *>(
      malloc(sizeof(unsigned long long) * offset));
  unsigned long long *current = reinterpret_cast<unsigned long long *>(
      malloc(sizeof(unsigned long long) * offset));

  std::stringstream dumpFilePath;
  dumpFilePath << data->dumpFilePath;
#if defined(__MINGW32__) || defined(__MINGW64__)
  dumpFilePath << "\\";
#else
  dumpFilePath << "/";
#endif
  dumpFilePath << "hamming_distance.csv";
  auto dumpFile = std::ofstream(dumpFilePath.str());

  // dump the header to the csv file
  dumpFile << "total_hamming_distance,";
  for (int i = 0; i < registerCount; i++) {
    for (int j = 0; j < registerSize; j++) {
      dumpFile << registerPrefix << i << "[" << j << "]";

      if (j != registerSize - 1) {
        dumpFile << ",";
      }
    }

    if (i != registerCount - 1) {
      dumpFile << ",";
    }
  }
  dumpFile << std::endl << std::flush;

  // do not output the hamming distance for the first run
  bool skipFirst = true;

  // continue until stop and dump the registers every data->dumpTimeDelta
  // seconds
  for (; *data->loadWorkerData->addrHigh != LOAD_STOP;) {
    // signal the thread to dump its largest SIMD registers
    *dumpVar = DumpVariable::Start;
    __asm__ __volatile__("mfence;");
    while (*dumpVar == DumpVariable::Start) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // copy the register content to minimize the interruption of the load worker
    std::memcpy(current, (void *)dumpMemAddr,
                sizeof(unsigned long long) * offset);

    // skip the first output, as we first have to get some valid values for last
    if (!skipFirst) {
      // calculate the total hamming distance
      int totalHammingDistance = 0;
      for (int i = 0; i < registerCount * registerSize; i++) {
        totalHammingDistance += hammingDistance(current[i], last[i]);
      }

      dumpFile << totalHammingDistance << ",";

      // dump the hamming distance of each double (last, current) pair
      for (int i = registerCount - 1; i >= 0; i--) {
        // auto registerNum = registerCount - 1 - i;

        for (auto j = 0; j < registerSize; j++) {
          auto index = registerSize * i + j;
          auto hd = static_cast<unsigned long long>(
              hammingDistance(current[index], last[index]));

          dumpFile << hd;
          if (j != registerSize - 1) {
            dumpFile << ",";
          }
        }

        if (i != 0) {
          dumpFile << ",";
        }
      }

      dumpFile << std::endl << std::flush;
    } else {
      skipFirst = false;
    }

    std::memcpy(last, current, sizeof(unsigned long long) * offset);

    std::this_thread::sleep_for(std::chrono::seconds(data->dumpTimeDelta));
  }

  dumpFile.close();

  free(last);
  free(current);
}

#endif
