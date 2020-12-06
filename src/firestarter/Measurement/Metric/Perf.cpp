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
#include <firestarter/Measurement/Metric/Perf.h>
#include <firestarter/Measurement/MetricInterface.h>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define PERF_EVENT_PARANOID "/proc/sys/kernel/perf_event_paranoid"

struct read_format {
  uint64_t value;
};

static std::string errorString = "";

static int cpu_cycles_fd = -1;
static int instructions_fd = -1;
static bool init_done = false;
static int init_value;

static struct read_format last[2];

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int fini(void) {
  if (!(cpu_cycles_fd < 0)) {
    close(cpu_cycles_fd);
    cpu_cycles_fd = -1;
  }
  if (!(instructions_fd < 0)) {
    close(instructions_fd);
    instructions_fd = -1;
  }
  return EXIT_SUCCESS;
}

static int init(void) {
  if (init_done) {
    return init_value;
  }

  if (!fs::exists(PERF_EVENT_PARANOID)) {
    // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    // The official way of knowing if perf_event_open() support is enabled
    // is checking for the existence of the file
    // /proc/sys/kernel/perf_event_paranoid.
    errorString =
        "syscall perf_event_open not supported or file " PERF_EVENT_PARANOID
        " does not exist";
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  struct perf_event_attr cpu_cycles_attr;
  std::memset(&cpu_cycles_attr, 0, sizeof(struct perf_event_attr));
  cpu_cycles_attr.type = PERF_TYPE_HARDWARE;
  cpu_cycles_attr.size = sizeof(struct perf_event_attr);
  cpu_cycles_attr.config = PERF_COUNT_HW_CPU_CYCLES;
  cpu_cycles_attr.inherit = 1;
  cpu_cycles_attr.exclude_kernel = 1;
  cpu_cycles_attr.exclude_hv = 1;
  cpu_cycles_attr.inherit_stat = 1;

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
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  struct perf_event_attr instructions_attr;
  std::memset(&instructions_attr, 0, sizeof(struct perf_event_attr));
  instructions_attr.type = PERF_TYPE_HARDWARE;
  instructions_attr.size = sizeof(struct perf_event_attr);
  instructions_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
  instructions_attr.inherit = 1;
  instructions_attr.exclude_kernel = 1;
  instructions_attr.exclude_hv = 1;
  instructions_attr.inherit_stat = 1;

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
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  ioctl(cpu_cycles_fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(instructions_fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(cpu_cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
  ioctl(instructions_fd, PERF_EVENT_IOC_ENABLE, 0);

  if (0 == read(cpu_cycles_fd, &last[0], sizeof(*last))) {
    fini();
    errorString = "PERF_COUNT_HW_CPU_CYCLES read failed in init";
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  if (0 == read(instructions_fd, &last[1], sizeof(*last))) {
    fini();
    errorString = "PERF_COUNT_HW_INSTRUCTIONS read failed in init";
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  init_value = EXIT_SUCCESS;
  init_done = true;
  return EXIT_SUCCESS;
}

static int get_reading(double *ipc_value, double *freq_value) {

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

  if (ipc_value != nullptr) {
    uint64_t diff[2];
    diff[0] = read_values[0].value - last[0].value;
    diff[1] = read_values[1].value - last[1].value;

    std::memcpy(last, read_values, sizeof(last));

    *ipc_value = (double)diff[1] / (double)diff[0];
  }

  if (freq_value != nullptr) {
    *freq_value = (double)read_values[0].value / 1e9;
  }

  return EXIT_SUCCESS;
}

static int get_reading_ipc(double *value) {
  return get_reading(value, nullptr);
}

static int get_reading_freq(double *value) {
  return get_reading(nullptr, value);
}

static const char *get_error(void) {
  const char *errorCString = errorString.c_str();
  return errorCString;
}
}

metric_interface_t perf_ipc_metric = {.name = "perf-ipc",
                                      .type = METRIC_ABSOLUTE,
                                      .unit = "",
                                      .callback_time = 0,
                                      .callback = nullptr,
                                      .init = init,
                                      .fini = fini,
                                      .get_reading = get_reading_ipc,
                                      .get_error = get_error};

metric_interface_t perf_freq_metric = {.name = "perf-freq",
                                       .type = METRIC_ACCUMALATIVE |
                                               METRIC_DIVIDE_BY_THREAD_COUNT,
                                       .unit = "GHz",
                                       .callback_time = 0,
                                       .callback = nullptr,
                                       .init = init,
                                       .fini = fini,
                                       .get_reading = get_reading_freq,
                                       .get_error = get_error};
