/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2026 TU Dresden, Center for Information Services and High
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
 *****************************************************************************/

#include "firestarter/Measurement/Metric/Perf.hpp"

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <linux/perf_event.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <vector>

namespace {
struct Event {
  int Cpu;
  int GroupFd;
  uint64_t Config;
};
std::vector<Event> Events;
std::vector<int> Closed;
int FailOpen = -1;
int FailReadFd = -1;
uint64_t Scale = 0;

void check(bool Condition, const char* Message) {
  if (!Condition) {
    throw std::runtime_error(Message);
  }
}

void reset() {
  PerfMetric::fini();
  Events.clear();
  Closed.clear();
  FailOpen = -1;
  FailReadFd = -1;
  Scale = 0;
  unsetenv("FIRESTARTER_PERF_CPU");
}
} // namespace

extern "C" auto __wrap_access(const char*, int) -> int { return 0; }
extern "C" auto __wrap_syscall(long Number, ...) -> long {
  va_list Args;
  va_start(Args, Number);
  auto* Attr = va_arg(Args, perf_event_attr*);
  const auto Pid = va_arg(Args, int);
  const auto Cpu = va_arg(Args, int);
  const auto GroupFd = va_arg(Args, int);
  va_end(Args);
  check(Pid == 0, "events must track the calling thread");
  if (static_cast<int>(Events.size()) == FailOpen) {
    return -1;
  }
  Events.push_back({Cpu, GroupFd, Attr->config});
  return static_cast<long>(Events.size() + 9);
}
extern "C" auto __wrap_ioctl(int Fd, unsigned long Request, ...) -> int {
  if (Request == PERF_EVENT_IOC_ID) {
    va_list Args;
    va_start(Args, Request);
    auto* Id = va_arg(Args, uint64_t*);
    va_end(Args);
    *Id = static_cast<uint64_t>(Fd);
  }
  return 0;
}
extern "C" auto __wrap_read(int Fd, void* Buffer, size_t Size) -> ssize_t {
  if (Fd == FailReadFd) {
    return -1;
  }
  const uint64_t GroupIndex = static_cast<uint64_t>((Fd - 10) / 2);
  const std::array<uint64_t, 5> Data{2, Scale * (GroupIndex + 1) * 1000000000, static_cast<uint64_t>(Fd),
                                     Scale * (GroupIndex + 1) * (GroupIndex + 2) * 1000000000,
                                     static_cast<uint64_t>(Fd + 1)};
  check(Size == sizeof(Data), "unexpected group read size");
  std::memcpy(Buffer, Data.data(), Size);
  return static_cast<ssize_t>(Size);
}
extern "C" auto __wrap_close(int Fd) -> int {
  Closed.push_back(Fd);
  return 0;
}

auto main() -> int {
  try {
    for (const char* Invalid : {"", "abc", "-1", "1,", ",1", "1,,2", "1,1", "2x", "999999999999999999999"}) {
      reset();
      setenv("FIRESTARTER_PERF_CPU", Invalid, 1);
      check(PerfMetric::init() == EXIT_FAILURE, "invalid CPU list accepted");
      check(std::string(PerfMetric::getError()).find("FIRESTARTER_PERF_CPU") != std::string::npos,
            "invalid CPU error missing");
      check(PerfMetric::init() == EXIT_FAILURE && Events.empty(), "invalid initialization not cached");
    }

    reset();
    setenv("FIRESTARTER_PERF_CPU", "2,5", 1);
    check(PerfMetric::init() == EXIT_SUCCESS, "subset initialization failed");
    check(Events.size() == 4 && Events[0].Cpu == 2 && Events[1].Cpu == 2 && Events[2].Cpu == 5 &&
              Events[3].Cpu == 5 && Events[0].GroupFd == -1 && Events[1].GroupFd == 10 &&
              Events[2].GroupFd == -1 && Events[3].GroupFd == 12 &&
              Events[0].Config == PERF_COUNT_HW_CPU_CYCLES && Events[1].Config == PERF_COUNT_HW_INSTRUCTIONS,
          "incorrect perf event groups");
    check(!PerfMetric::PerfFreqMetric.Type.DivideByThreadCount, "subset must not use thread count");
    Scale = 1;
    double Value = 0;
    check(PerfMetric::getReadingFreq(&Value, 1) == EXIT_SUCCESS && Value == 1.5, "incorrect subset frequency");
    check(PerfMetric::getReadingIpc(&Value, 1) == EXIT_SUCCESS && Value == 8.0 / 3.0, "incorrect aggregate IPC");
    Scale = 2;
    check(PerfMetric::getReadingIpc(&Value, 1) == EXIT_SUCCESS && Value == 8.0 / 3.0, "incorrect IPC delta");
    check(PerfMetric::fini() == EXIT_SUCCESS && Closed.size() == 4, "subset descriptors not closed");

    reset();
    check(PerfMetric::init() == EXIT_SUCCESS && Events.size() == 2 && Events[0].Cpu == -1,
          "default must track all CPUs");
    check(PerfMetric::PerfFreqMetric.Type.DivideByThreadCount, "default must use thread count");
    Scale = 1;
    check(PerfMetric::getReadingFreq(&Value, 1) == EXIT_SUCCESS && Value == 1.0, "incorrect default frequency");

    reset();
    setenv("FIRESTARTER_PERF_CPU", "7", 1);
    check(PerfMetric::init() == EXIT_SUCCESS && !PerfMetric::PerfFreqMetric.Type.DivideByThreadCount,
          "single CPU initialization failed");
    reset();
    setenv("FIRESTARTER_PERF_CPU", "2,5", 1);
    FailOpen = 3;
    check(PerfMetric::init() == EXIT_FAILURE && Closed.size() == 3, "partial initialization leaked descriptors");
    check(std::string(PerfMetric::getError()).find("INSTRUCTIONS") != std::string::npos, "missing open error");

    reset();
    setenv("FIRESTARTER_PERF_CPU", "2,5", 1);
    check(PerfMetric::init() == EXIT_SUCCESS, "subset initialization failed");
    FailReadFd = 12;
    check(PerfMetric::getReadingFreq(&Value, 1) == EXIT_FAILURE && Closed.size() == 4,
          "failed read did not clean up groups");
    reset();
    return EXIT_SUCCESS;
  } catch (const std::exception& Error) {
    std::cerr << Error.what() << '\n';
    reset();
    return EXIT_FAILURE;
  }
}
