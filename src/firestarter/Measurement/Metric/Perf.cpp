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

extern "C" {
#include <firestarter/Measurement/Metric/Perf.h>
#include <firestarter/Measurement/MetricInterface.h>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define PERF_EVENT_PARANOID "/proc/sys/kernel/perf_event_paranoid"

struct read_format {
  uint64_t nr;
  struct {
    uint64_t value;
    uint64_t id;
  } values[2];
};

static std::string errorString = "";

static int cpu_cycles_fd = -1;
static int instructions_fd = -1;
static uint64_t cpu_cycles_id;
static uint64_t instructions_id;
static bool init_done = false;
static int32_t init_value;

static struct read_format last;

static long perf_event_open(struct perf_event_attr* hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int32_t fini(void) {
  if (!(cpu_cycles_fd < 0)) {
    close(cpu_cycles_fd);
    cpu_cycles_fd = -1;
  }
  if (!(instructions_fd < 0)) {
    close(instructions_fd);
    instructions_fd = -1;
  }
  init_done = false;
  return EXIT_SUCCESS;
}

static int32_t init(void) {
  if (init_done) {
    return init_value;
  }

  if (access(PERF_EVENT_PARANOID, F_OK) == -1) {
    // https://man7.org/linux/man-pages/man2/perf_event_open.2.html
    // The official way of knowing if perf_event_open() support is enabled
    // is checking for the existence of the file
    // /proc/sys/kernel/perf_event_paranoid.
    errorString = "syscall perf_event_open not supported or file " PERF_EVENT_PARANOID " does not exist";
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  struct perf_event_attr cpu_cycles_attr;
  std::memset(&cpu_cycles_attr, 0, sizeof(struct perf_event_attr));
  cpu_cycles_attr.type = PERF_TYPE_HARDWARE;
  cpu_cycles_attr.size = sizeof(struct perf_event_attr);
  cpu_cycles_attr.config = PERF_COUNT_HW_CPU_CYCLES;
  cpu_cycles_attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
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
  cpu_cycles_attr.inherit = 1;
  cpu_cycles_attr.exclude_kernel = 1;
  cpu_cycles_attr.exclude_hv = 1;

  if ((cpu_cycles_fd = perf_event_open(&cpu_cycles_attr,
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

  ioctl(cpu_cycles_fd, PERF_EVENT_IOC_ID, &cpu_cycles_id);

  struct perf_event_attr instructions_attr;
  std::memset(&instructions_attr, 0, sizeof(struct perf_event_attr));
  instructions_attr.type = PERF_TYPE_HARDWARE;
  instructions_attr.size = sizeof(struct perf_event_attr);
  instructions_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
  instructions_attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
  instructions_attr.inherit = 1;
  instructions_attr.exclude_kernel = 1;
  instructions_attr.exclude_hv = 1;

  if ((instructions_fd = perf_event_open(&instructions_attr,
                                         // pid == 0 and cpu == -1
                                         // This measures the calling process/thread on any CPU.
                                         0, -1,
                                         // The group_fd argument allows event groups to be created.  An event
                                         // group has one event which is the group leader.  The leader is
                                         // created first, with group_fd = -1.  The rest of the group members
                                         // are created with subsequent perf_event_open() calls with group_fd
                                         // being set to the file descriptor of the group leader.
                                         cpu_cycles_fd, 0)) < 0) {
    fini();
    errorString = "perf_event_open failed for PERF_COUNT_HW_INSTRUCTIONS";
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  ioctl(instructions_fd, PERF_EVENT_IOC_ID, &instructions_id);

  ioctl(cpu_cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(cpu_cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

  if (0 == read(cpu_cycles_fd, &last, sizeof(last))) {
    fini();
    errorString = "group read failed in init";
    init_value = EXIT_FAILURE;
    init_done = true;
    return EXIT_FAILURE;
  }

  init_value = EXIT_SUCCESS;
  init_done = true;
  return EXIT_SUCCESS;
}

static uint64_t value_from_id(struct read_format* values, uint64_t id) {
  for (decltype(values->nr) i = 0; i < values->nr; ++i) {
    if (id == values->values[i].id) {
      return values->values[i].value;
    }
  }

  return 0;
}

static int32_t get_reading(double* ipc_value, double* freq_value) {

  if (cpu_cycles_fd < 0 || instructions_fd < 0) {
    fini();
    return EXIT_FAILURE;
  }

  struct read_format read_values;

  if (0 == read(cpu_cycles_fd, &read_values, sizeof(read_values))) {
    fini();
    errorString = "group read failed";
    return EXIT_FAILURE;
  }

  if (ipc_value != nullptr) {
    uint64_t diff[2];
    diff[0] = value_from_id(&read_values, instructions_id) - value_from_id(&last, instructions_id);
    diff[1] = value_from_id(&read_values, cpu_cycles_id) - value_from_id(&last, cpu_cycles_id);

    std::memcpy(&last, &read_values, sizeof(last));

    *ipc_value = (double)diff[0] / (double)diff[1];
  }

  if (freq_value != nullptr) {
    *freq_value = (double)value_from_id(&read_values, cpu_cycles_id) / 1e9;
  }

  return EXIT_SUCCESS;
}

static int32_t get_reading_ipc(double* value) { return get_reading(value, nullptr); }

static int32_t get_reading_freq(double* value) { return get_reading(nullptr, value); }

static const char* get_error(void) {
  const char* errorCString = errorString.c_str();
  return errorCString;
}
}

metric_interface_t perf_ipc_metric = {
    .name = "perf-ipc",
    .type = {.absolute = 1,
             .accumalative = 0,
             .divide_by_thread_count = 0,
             .insert_callback = 0,
             .ignore_start_stop_delta = 0,
             .__reserved = 0},
    .unit = "IPC",
    .callback_time = 0,
    .callback = nullptr,
    .init = init,
    .fini = fini,
    .get_reading = get_reading_ipc,
    .get_error = get_error,
    .register_insert_callback = nullptr,
};

metric_interface_t perf_freq_metric = {
    .name = "perf-freq",
    .type = {.absolute = 0,
             .accumalative = 1,
             .divide_by_thread_count = 1,
             .insert_callback = 0,
             .ignore_start_stop_delta = 0,
             .__reserved = 0},
    .unit = "GHz",
    .callback_time = 0,
    .callback = nullptr,
    .init = init,
    .fini = fini,
    .get_reading = get_reading_freq,
    .get_error = get_error,
    .register_insert_callback = nullptr,
};
