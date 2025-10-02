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

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h> // IWYU pragma: keep
#include <unistd.h>

namespace {
// NOLINTNEXTLINE(misc-include-cleaner)
auto perfEventOpen(struct perf_event_attr* HwEvent, pid_t Pid, int Cpu, int GroupFd, uint64_t Flags) -> int {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,misc-include-cleaner)
  return static_cast<int>(syscall(__NR_perf_event_open, HwEvent, Pid, Cpu, GroupFd, Flags));
}
} // namespace

auto PerfMetric::fini() -> int32_t {
  auto& Instance = instance();

  if (!(Instance.CpuCyclesFd < 0)) {
    close(Instance.CpuCyclesFd);
    Instance.CpuCyclesFd = -1;
  }
  if (!(Instance.InstructionsFd < 0)) {
    close(Instance.InstructionsFd);
    Instance.InstructionsFd = -1;
  }
  Instance.InitDone = false;
  return EXIT_SUCCESS;
}

auto PerfMetric::init() -> int32_t {
  auto& Instance = instance();

  if (Instance.InitDone) {
    return Instance.InitValue;
  }

  // No other functions are setting the environment variables
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (const char* Cpu = std::getenv("FIRESTARTER_PERF_CPU")) {
    // Collect the perf metrics only from a specific CPU.
    Instance.PerfCpu = std::stoi(Cpu);
    // As we only collect metrics for one CPU, we do not need to divide the collected metrics by the thread count.
    PerfMetric::PerfFreqMetric.Type.DivideByThreadCount = 0;
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

  Instance.CpuCyclesFd = perfEventOpen(&CpuCyclesAttr,
                                       // pid == 0 and cpu == -1
                                       // This measures the calling process/thread on any CPU.
                                       0, Instance.PerfCpu,
                                       // The group_fd argument allows event groups to be created.  An event
                                       // group has one event which is the group leader.  The leader is
                                       // created first, with group_fd = -1.  The rest of the group members
                                       // are created with subsequent perf_event_open() calls with group_fd
                                       // being set to the file descriptor of the group leader.
                                       -1, 0);

  if (Instance.CpuCyclesFd < 0) {
    fini();
    Instance.ErrorString = "perf_event_open failed for PERF_COUNT_HW_CPU_CYCLES";
    Instance.InitValue = EXIT_FAILURE;
    Instance.InitDone = true;
    return EXIT_FAILURE;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  ioctl(Instance.CpuCyclesFd, PERF_EVENT_IOC_ID, &Instance.CpuCyclesId);

  struct perf_event_attr InstructionsAttr {};
  std::memset(&InstructionsAttr, 0, sizeof(struct perf_event_attr));
  InstructionsAttr.type = PERF_TYPE_HARDWARE;
  InstructionsAttr.size = sizeof(struct perf_event_attr);
  InstructionsAttr.config = PERF_COUNT_HW_INSTRUCTIONS;
  InstructionsAttr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  InstructionsAttr.inherit = 1;
  InstructionsAttr.exclude_kernel = 1;
  InstructionsAttr.exclude_hv = 1;

  Instance.InstructionsFd = perfEventOpen(&InstructionsAttr,
                                          // pid == 0 and cpu == -1
                                          // This measures the calling process/thread on any CPU.
                                          0, Instance.PerfCpu,
                                          // The group_fd argument allows event groups to be created.  An event
                                          // group has one event which is the group leader.  The leader is
                                          // created first, with group_fd = -1.  The rest of the group members
                                          // are created with subsequent perf_event_open() calls with group_fd
                                          // being set to the file descriptor of the group leader.
                                          Instance.CpuCyclesFd, 0);

  if (Instance.InstructionsFd < 0) {
    fini();
    Instance.ErrorString = "perf_event_open failed for PERF_COUNT_HW_INSTRUCTIONS";
    Instance.InitValue = EXIT_FAILURE;
    Instance.InitDone = true;
    return EXIT_FAILURE;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  ioctl(Instance.InstructionsFd, PERF_EVENT_IOC_ID, &Instance.InstructionsId);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  ioctl(Instance.CpuCyclesFd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  ioctl(Instance.CpuCyclesFd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

  if (0 == read(Instance.CpuCyclesFd, &Instance.Last, sizeof(Last))) {
    fini();
    Instance.ErrorString = "group read failed in init";
    Instance.InitValue = EXIT_FAILURE;
    Instance.InitDone = true;
    return EXIT_FAILURE;
  }

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

  if (Instance.CpuCyclesFd < 0 || Instance.InstructionsFd < 0) {
    fini();
    return EXIT_FAILURE;
  }

  struct ReadFormat ReadValues {};

  if (0 == read(Instance.CpuCyclesFd, &ReadValues, sizeof(ReadValues))) {
    fini();
    Instance.ErrorString = "group read failed";
    return EXIT_FAILURE;
  }

  if (IpcValue != nullptr) {
    std::array<uint64_t, 2> Diff = {
        valueFromId(&ReadValues, Instance.InstructionsId) - valueFromId(&Instance.Last, Instance.InstructionsId),
        valueFromId(&ReadValues, Instance.CpuCyclesId) - valueFromId(&Instance.Last, Instance.CpuCyclesId)};

    std::memcpy(&Instance.Last, &ReadValues, sizeof(Last));

    *IpcValue = static_cast<double>(Diff[0]) / static_cast<double>(Diff[1]);
  }

  if (FreqValue != nullptr) {
    *FreqValue = static_cast<double>(valueFromId(&ReadValues, Instance.CpuCyclesId)) / 1e9;
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