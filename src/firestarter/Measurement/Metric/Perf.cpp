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

#include "firestarter/Measurement/Metric/Perf.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h> // IWYU pragma: keep
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {
// NOLINTNEXTLINE(misc-include-cleaner)
auto perfEventOpen(struct perf_event_attr* HwEvent, pid_t Pid, int Cpu, int GroupFd, uint64_t Flags) -> int {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,misc-include-cleaner)
  return static_cast<int>(syscall(__NR_perf_event_open, HwEvent, Pid, Cpu, GroupFd, Flags));
}
} // namespace

auto PerfMetric::fini() -> int32_t {
  auto& Instance = instance();

  for (const auto& Group : Instance.Groups) {
    if (Group.InstructionsFd >= 0) {
      close(Group.InstructionsFd);
    }
    if (Group.CpuCyclesFd >= 0) {
      close(Group.CpuCyclesFd);
    }
  }
  Instance.Groups.clear();
  Instance.CpuFilter = false;
  PerfFreqMetric.Type.DivideByThreadCount = 1;
  Instance.InitDone = false;
  return EXIT_SUCCESS;
}

auto PerfMetric::init() -> int32_t {
  auto& Instance = instance();

  if (Instance.InitDone) {
    return Instance.InitValue;
  }

  std::vector<int> Cpus;
  // No other functions are setting the environment variables
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (const char* CpuEnv = std::getenv("FIRESTARTER_PERF_CPU")) {
    const std::string_view CpuValue(CpuEnv);
    auto Remaining = CpuValue;
    while (true) {
      const auto Comma = Remaining.find(',');
      const auto Token = Remaining.substr(0, Comma);
      int Cpu = -1;
      const auto Result = std::from_chars(Token.data(), Token.data() + Token.size(), Cpu);
      if (Token.empty() || Result.ec != std::errc{} || Result.ptr != Token.data() + Token.size() || Cpu < 0 ||
          std::find(Cpus.begin(), Cpus.end(), Cpu) != Cpus.end()) {
        Instance.ErrorString = std::string(CpuValue) + " is not a valid value for FIRESTARTER_PERF_CPU.";
        Instance.InitValue = EXIT_FAILURE;
        Instance.InitDone = true;
        return EXIT_FAILURE;
      }
      Cpus.push_back(Cpu);
      if (Comma == std::string_view::npos) {
        break;
      }
      Remaining.remove_prefix(Comma + 1);
    }
  } else {
    Cpus.push_back(-1);
  }

  if (access(PerfEventParanoidFile, F_OK) == -1) {
    // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    // The official way of knowing if perf_event_open() support is enabled
    // is checking for the existence of the file
    // /proc/sys/kernel/perf_event_paranoid.
    Instance.ErrorString =
        "syscall perf_event_open not supported or file " + std::string(PerfEventParanoidFile) + " does not exist";
    Instance.InitValue = EXIT_FAILURE;
    Instance.InitDone = true;
    return EXIT_FAILURE;
  }

  struct perf_event_attr CpuCyclesAttr {};
  std::memset(&CpuCyclesAttr, 0, sizeof(struct perf_event_attr));
  CpuCyclesAttr.type = PERF_TYPE_HARDWARE;
  CpuCyclesAttr.size = sizeof(struct perf_event_attr);
  CpuCyclesAttr.config = PERF_COUNT_HW_CPU_CYCLES;
  CpuCyclesAttr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  //     inherit
  // The inherit bit specifies that this counter should count
  // events of child tasks as well as the task specified.  This
  // applies only to new children, not to any existing children
  // at the time the counter is created (nor to any new
  // children of existing children).
  //
  // Inherit does not work for some combinations of read_format
  // values, such as PERF_FORMAT_GROUP.
  //
  // ---
  //
  // As for kernel versions >= 4.13 this is not true.
  // This commit
  // https://github.com/torvalds/linux/commit/ba5213ae6b88fb170c4771fef6553f759c7d8cdd
  // changed the check
  // - if (attr->inherit && (attr->read_format & PERF_FORMAT_GROUP))
  // + if (attr->inherit && (attr->sample_type & PERF_SAMPLE_READ))
  CpuCyclesAttr.inherit = 1;
  CpuCyclesAttr.exclude_kernel = 1;
  CpuCyclesAttr.exclude_hv = 1;

  struct perf_event_attr InstructionsAttr {};
  InstructionsAttr.type = PERF_TYPE_HARDWARE;
  InstructionsAttr.size = sizeof(struct perf_event_attr);
  InstructionsAttr.config = PERF_COUNT_HW_INSTRUCTIONS;
  InstructionsAttr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  InstructionsAttr.inherit = 1;
  InstructionsAttr.exclude_kernel = 1;
  InstructionsAttr.exclude_hv = 1;

  const auto Fail = [&Instance](const std::string& Error) {
    fini();
    Instance.ErrorString = Error;
    Instance.InitValue = EXIT_FAILURE;
    Instance.InitDone = true;
    return EXIT_FAILURE;
  };

  for (const int Cpu : Cpus) {
    Instance.Groups.emplace_back();
    auto& Group = Instance.Groups.back();
    // pid 0 tracks the calling thread and newly created descendants, restricted
    // to Cpu when it is non-negative.
    Group.CpuCyclesFd = perfEventOpen(&CpuCyclesAttr, 0, Cpu, -1, 0);

    if (Group.CpuCyclesFd < 0) {
      return Fail("perf_event_open failed for PERF_COUNT_HW_CPU_CYCLES");
    }

    Group.InstructionsFd = perfEventOpen(&InstructionsAttr, 0, Cpu, Group.CpuCyclesFd, 0);
    if (Group.InstructionsFd < 0) {
      return Fail("perf_event_open failed for PERF_COUNT_HW_INSTRUCTIONS");
    }

    // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
    if (ioctl(Group.CpuCyclesFd, PERF_EVENT_IOC_ID, &Group.CpuCyclesId) < 0 ||
        ioctl(Group.InstructionsFd, PERF_EVENT_IOC_ID, &Group.InstructionsId) < 0 ||
        ioctl(Group.CpuCyclesFd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0 ||
        ioctl(Group.CpuCyclesFd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) {
      return Fail("perf event ioctl failed");
    }
    // NOLINTEND(cppcoreguidelines-pro-type-vararg)
    if (read(Group.CpuCyclesFd, &Group.Last, sizeof(Group.Last)) != sizeof(Group.Last) || Group.Last.Nr != 2) {
      return Fail("group read failed in init");
    }
  }

  Instance.CpuFilter = Cpus.front() != -1;
  PerfFreqMetric.Type.DivideByThreadCount = Instance.CpuFilter ? 0 : 1;
  Instance.InitValue = EXIT_SUCCESS;
  Instance.InitDone = true;
  return EXIT_SUCCESS;
}

auto PerfMetric::valueFromId(struct ReadFormat* Reader, uint64_t Id) -> uint64_t {
  for (decltype(Reader->Nr) I = 0; I < Reader->Nr; ++I) {
    assert(I < 2 && "Index is out of bounds");
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    if (Id == Reader->Values[I].Id) {
      return Reader->Values[I].Value;
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  }

  return 0;
}

auto PerfMetric::getReading(double* IpcValue, double* FreqValue) -> int32_t {
  auto& Instance = instance();

  if (Instance.Groups.empty()) {
    Instance.ErrorString = "perf metric is not initialized";
    return EXIT_FAILURE;
  }

  std::vector<ReadFormat> Readings(Instance.Groups.size());
  for (size_t I = 0; I < Instance.Groups.size(); ++I) {
    if (read(Instance.Groups[I].CpuCyclesFd, &Readings[I], sizeof(ReadFormat)) != sizeof(ReadFormat) ||
        Readings[I].Nr != 2) {
      fini();
      Instance.ErrorString = "group read failed";
      return EXIT_FAILURE;
    }
  }

  uint64_t InstructionsDiff = 0;
  uint64_t CyclesDiff = 0;
  double Cycles = 0;
  for (size_t I = 0; I < Instance.Groups.size(); ++I) {
    auto& Group = Instance.Groups[I];
    auto& Reading = Readings[I];
    const auto CurrentCycles = valueFromId(&Reading, Group.CpuCyclesId);
    Cycles += static_cast<double>(CurrentCycles);
    if (IpcValue != nullptr) {
      InstructionsDiff += valueFromId(&Reading, Group.InstructionsId) - valueFromId(&Group.Last, Group.InstructionsId);
      CyclesDiff += CurrentCycles - valueFromId(&Group.Last, Group.CpuCyclesId);
      Group.Last = Reading;
    }
  }
  if (IpcValue != nullptr) {
    *IpcValue = static_cast<double>(InstructionsDiff) / static_cast<double>(CyclesDiff);
  }
  if (FreqValue != nullptr) {
    *FreqValue = Cycles / 1e9;
    if (Instance.CpuFilter) {
      *FreqValue /= static_cast<double>(Instance.Groups.size());
    }
  }

  return EXIT_SUCCESS;
}

auto PerfMetric::getReadingIpc(double* Value, uint64_t NumElems) -> int32_t {
  assert(NumElems == 1 && "The number of elements should be exctly one, since no submetrics are available.");
  return getReading(Value, nullptr);
}

auto PerfMetric::getReadingFreq(double* Value, uint64_t NumElems) -> int32_t {
  assert(NumElems == 1 && "The number of elements should be exctly one, since no submetrics are available.");
  return getReading(nullptr, Value);
}

auto PerfMetric::getError() -> const char* {
  const char* ErrorCString = instance().ErrorString.c_str();
  return ErrorCString;
}