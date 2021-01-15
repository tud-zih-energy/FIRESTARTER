![Build Status](https://github.com/tud-zih-energy/FIRESTARTER/workflows/Build/badge.svg)

# FIRESTARTER - A Processor Stress Test Utility

FIRESTARTER can be build under Linux, Windows and macOS with CMake.

GCC (>=7) or Clang (>=9) is supported.

CMake option | Description
:--- | :---
FIRESTARTER_BUILD_TYPE | Can be any of FIRESTARTER, FIRESTARTER_CUDA or FIRESTARTER_CUDA_ONLY. Default FIRESTARTER
FIRESTARTER_LINK_STATIC | Link FIRESTARTER as a static binary. Note, dlopen is not supported in static binaries. This option is not available on macOS or with CUDA enabled. Default ON
FIRESTARTER_BUILD_HWLOC | Build hwloc dependency. Default ON
FIRESTARTER_THREAD_AFFINITY | Enable FIRESTARTER to set affinity to hardware threads. Default ON

## Metrics

FIRESTARTER support to collect metrics during runtime.
Available metrics can be shown with `--list-metrics`.
Default metrics are perf-ipc, perf-freq, ipc-estimate and powercap-sysfs-rapl.

### Custom metrics

If one would like to use custom metrics, e.g. an external power measurement, `--metric-from-stdin=NAME` allows metric values to be passed via stdin in the following format:
`{NAME} {TIME SINCE EPOCH IN NS} {ABSOLUT METRIC VALUE (double)}\n`.
Make sure to use flush after each line.

## Measurement

FIRESTARTER has the option to output the colleted metric values by specifying `--measurement`.
Options `--start-delta` and `--stop-delta` specify a time in milliseconds in which metric values should be ignored.
After a run the output will be given in csv format to stdout.

## Optimization

FIRESTARTER has the option to optimize itself.
It currently supports the multiobjective algorithm NSGA2, selected by `--optimize=NSGA2`.
The optimization relies on the execution of FIRESTARTER with a combination of instruction groups, specified by `--run-instruction-groups`.
Available instruction groups can be listed with `--list-instruction-groups`.
During each test run of the duration specified by `-t | --timeout` metrics will collect information about the fitness.
The used metrics for optimization can be specified by `--optimization-metrics`.
An output file with the results will be written to `{HOSTNAME}_${STARTTIME}.json` if the option `--optimize-outfile` is not given.

# Reference

A detailed description can be found in the following paper. Please cite this if you use FIRESTARTER for scientific work.

Daniel Hackenberg, Roland Oldenburg, Daniel Molka, and Robert Schöne
[Introducing FIRESTARTER: A processor stress test utility](http://dx.doi.org/10.1109/IGCC.2013.6604507) (IGCC 2013)

Additional information: https://tu-dresden.de/zih/forschung/projekte/firestarter

# License

This program contains a slightly modified version of the implementation of the NSGA2 algorithm from [esa/pagmo2](https://github.com/esa/pagmo2) licensed under LGPL or GPL v3.

This program incorporates following libraries [asmjit/asmjit](https://github.com/asmjit/asmjit) licensed under zlib, [open-mpi/hwloc](https://github.com/open-mpi/hwloc) licensed under BSD 3-clause, [jarro2783/cxxopts](https://github.com/jarro2783/cxxopts) licensed under MIT, [nlohmann/json](https://github.com/nlohmann/json) licensed under MIT and [tud-zih-energy/nitro](https://github.com/tud-zih-energy/nitro) licensed under BSD 3-clause.

# Contact

Daniel Hackenberg < daniel dot hackenberg at tu-dresden.de >
