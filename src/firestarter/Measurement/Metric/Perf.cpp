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

#include <cstring>
#include <string>

#include <firestarter/Measurement/Metric/Perf.hpp>

extern "C" {
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
}

static auto perfEventOpen(struct perf_event_attr* HwEvent, pid_t Pid, int Cpu, int GroupFd, unsigned long Flags)
    -> long {
  return syscall(__NR_perf_event_open, HwEvent, Pid, Cpu, GroupFd, Flags);
}

auto PerfMetricData::fini() -> int32_t {
  if (!(CpuCyclesFd < 0)) {
    close(CpuCyclesFd);
    CpuCyclesFd = -1;
  }
  if (!(InstructionsFd < 0)) {
    close(InstructionsFd);
    InstructionsFd = -1;
  }
  InitDone = false;
  return EXIT_SUCCESS;
}

auto PerfMetricData::init() -> int32_t {
  if (InitDone) {
    return InitValue;
  }

  if (access(PerfEventParanoidFile, F_OK) == -1) {
    // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    // The official way of knowing if perf_event_open() support is enabled
    // is checking for the existence of the file
    // /proc/sys/kernel/perf_event_paranoid.
    ErrorString =
        "syscall perf_event_open not supported or file " + std::string(PerfEventParanoidFile) + " does not exist";
    InitValue = EXIT_FAILURE;
    InitDone = true;
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

  if ((CpuCyclesFd = perfEventOpen(&CpuCyclesAttr,
                                   // pid == 0 and cpu == -1
                                   // This measures the calling process/thread on any CPU.
                                   0, -1,
                                   // The group_fd argument allows event groups to be created.  An event
                                   // group has one event which is the group leader.  The leader is
                                   // created first, with group_fd = -1.  The rest of the group members
                                   // are created with subsequent perf_event_open() calls with group_fd
                                   // being set to the file descriptor of the group leader.
                                   -1, 0)) < 0) {
    fini();
    ErrorString = "perf_event_open failed for PERF_COUNT_HW_CPU_CYCLES";
    InitValue = EXIT_FAILURE;
    InitDone = true;
    return EXIT_FAILURE;
  }

  ioctl(CpuCyclesFd, PERF_EVENT_IOC_ID, &CpuCyclesId);

  struct perf_event_attr InstructionsAttr {};
  std::memset(&InstructionsAttr, 0, sizeof(struct perf_event_attr));
  InstructionsAttr.type = PERF_TYPE_HARDWARE;
  InstructionsAttr.size = sizeof(struct perf_event_attr);
  InstructionsAttr.config = PERF_COUNT_HW_INSTRUCTIONS;
  InstructionsAttr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  InstructionsAttr.inherit = 1;
  InstructionsAttr.exclude_kernel = 1;
  InstructionsAttr.exclude_hv = 1;

  if ((InstructionsFd = perfEventOpen(&InstructionsAttr,
                                      // pid == 0 and cpu == -1
                                      // This measures the calling process/thread on any CPU.
                                      0, -1,
                                      // The group_fd argument allows event groups to be created.  An event
                                      // group has one event which is the group leader.  The leader is
                                      // created first, with group_fd = -1.  The rest of the group members
                                      // are created with subsequent perf_event_open() calls with group_fd
                                      // being set to the file descriptor of the group leader.
                                      CpuCyclesFd, 0)) < 0) {
    fini();
    ErrorString = "perf_event_open failed for PERF_COUNT_HW_INSTRUCTIONS";
    InitValue = EXIT_FAILURE;
    InitDone = true;
    return EXIT_FAILURE;
  }

  ioctl(InstructionsFd, PERF_EVENT_IOC_ID, &InstructionsId);

  ioctl(CpuCyclesFd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(CpuCyclesFd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

  if (0 == read(CpuCyclesFd, &Last, sizeof(Last))) {
    fini();
    ErrorString = "group read failed in init";
    InitValue = EXIT_FAILURE;
    InitDone = true;
    return EXIT_FAILURE;
  }

  InitValue = EXIT_SUCCESS;
  InitDone = true;
  return EXIT_SUCCESS;
}

auto PerfMetricData::valueFromId(struct ReadFormat* Values, uint64_t Id) -> uint64_t {
  for (decltype(Values->Nr) I = 0; I < Values->Nr; ++I) {
    if (Id == Values->Values[I].Id) {
      return Values->Values[I].Value;
    }
  }

  return 0;
}

auto PerfMetricData::getReading(double* IpcValue, double* FreqValue) -> int32_t {

  if (CpuCyclesFd < 0 || InstructionsFd < 0) {
    fini();
    return EXIT_FAILURE;
  }

  struct ReadFormat ReadValues {};

  if (0 == read(CpuCyclesFd, &ReadValues, sizeof(ReadValues))) {
    fini();
    ErrorString = "group read failed";
    return EXIT_FAILURE;
  }

  if (IpcValue != nullptr) {
    uint64_t Diff[2];
    Diff[0] = valueFromId(&ReadValues, InstructionsId) - valueFromId(&Last, InstructionsId);
    Diff[1] = valueFromId(&ReadValues, CpuCyclesId) - valueFromId(&Last, CpuCyclesId);

    std::memcpy(&Last, &ReadValues, sizeof(Last));

    *IpcValue = (double)Diff[0] / (double)Diff[1];
  }

  if (FreqValue != nullptr) {
    *FreqValue = (double)valueFromId(&ReadValues, CpuCyclesId) / 1e9;
  }

  return EXIT_SUCCESS;
}

auto PerfMetricData::getReadingIpc(double* Value) -> int32_t { return getReading(Value, nullptr); }

auto PerfMetricData::getReadingFreq(double* Value) -> int32_t { return getReading(nullptr, Value); }

auto PerfMetricData::getError() -> const char* {
  const char* ErrorCString = ErrorString.c_str();
  return ErrorCString;
}