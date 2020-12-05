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

#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

extern "C" {
#include <firestarter/Measurement/Metric/PerfIPC.h>
#include <firestarter/Measurement/MetricInterface.h>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define PERF_EVENT_PARANOID "/proc/sys/kernel/perf_event_paranoid"

static const char *unit = std::string("").c_str();
static unsigned long long callback_time = 0;
static std::string errorString = "";

static int cpu_cycles_fd = -1;
static int instructions_fd = -1;

struct read_format {
  uint64_t value;
};

static struct read_format last[2];

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int fini(void) {
  if (!(cpu_cycles_fd < 0)) {
    close(cpu_cycles_fd);
  }
  if (!(instructions_fd < 0)) {
    close(instructions_fd);
  }
  return EXIT_SUCCESS;
}

static int init(void) {
  if (!fs::exists(PERF_EVENT_PARANOID)) {
    // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    // The official way of knowing if perf_event_open() support is enabled
    // is checking for the existence of the file
    // /proc/sys/kernel/perf_event_paranoid.
    errorString =
        "syscall perf_event_open not supported or file " PERF_EVENT_PARANOID
        " does not exist";
    return EXIT_FAILURE;
  }

  struct perf_event_attr cpu_cycles_attr = {
      .type = PERF_TYPE_HARDWARE,
      .size = sizeof(struct perf_event_attr),
      .config = PERF_COUNT_HW_CPU_CYCLES,
      .read_format = 0,
      .disabled = 0,
      .inherit = 1,
      .exclude_kernel = 1,
      .exclude_hv = 1,
      .inherit_stat = 1,
  };
  if ((cpu_cycles_fd = perf_event_open(
           &cpu_cycles_attr,
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
    errorString = "perf_event_open failed for PERF_COUNT_HW_CPU_CYCLES";
    return EXIT_FAILURE;
  }

  struct perf_event_attr instructions_attr = {
      .type = PERF_TYPE_HARDWARE,
      .size = sizeof(struct perf_event_attr),
      .config = PERF_COUNT_HW_INSTRUCTIONS,
      .read_format = 0,
      .disabled = 0,
      .inherit = 1,
      .exclude_kernel = 1,
      .exclude_hv = 1,
      .inherit_stat = 1,
  };

  if ((instructions_fd = perf_event_open(
           &instructions_attr,
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
    errorString = "perf_event_open failed for PERF_COUNT_HW_INSTRUCTIONS";
    return EXIT_FAILURE;
  }

  ioctl(cpu_cycles_fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(instructions_fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(cpu_cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
  ioctl(instructions_fd, PERF_EVENT_IOC_ENABLE, 0);

  if (0 == read(cpu_cycles_fd, &last[0], sizeof(*last))) {
    fini();
    errorString = "PERF_COUNT_HW_CPU_CYCLES read failed in init";
    return EXIT_FAILURE;
  }

  if (0 == read(instructions_fd, &last[1], sizeof(*last))) {
    fini();
    errorString = "PERF_COUNT_HW_INSTRUCTIONS read failed in init";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

static int get_reading(double *value) {

  if (cpu_cycles_fd < 0 || instructions_fd < 0) {
    fini();
    return EXIT_FAILURE;
  }

  struct read_format read_values[2];

  if (0 == read(cpu_cycles_fd, &read_values[0], sizeof(*read_values))) {
    fini();
    errorString = "PERF_COUNT_HW_CPU_CYCLES read failed in init";
    return EXIT_FAILURE;
  }

  if (0 == read(instructions_fd, &read_values[1], sizeof(*read_values))) {
    fini();
    errorString = "PERF_COUNT_HW_INSTRUCTIONS read failed in init";
    return EXIT_FAILURE;
  }

  uint64_t diff[2];
  diff[0] = read_values[0].value - last[0].value;
  diff[1] = read_values[1].value - last[1].value;

  double ipc = (double)diff[1] / (double)diff[0];

  std::memcpy(last, read_values, sizeof(last));

  if (value != nullptr) {
    *value = ipc;
  }

  return EXIT_SUCCESS;
}

static const char *get_error(void) {
  const char *errorCString = errorString.c_str();
  return errorCString;
}
}

metric_interface_t perf_ipc_metric = {.name = "perf-ipc",
                                      .type = METRIC_ABSOLUTE,
                                      .unit = unit,
                                      .callback_time = callback_time,
                                      .callback = NULL,
                                      .init = init,
                                      .fini = fini,
                                      .get_reading = get_reading,
                                      .get_error = get_error};
