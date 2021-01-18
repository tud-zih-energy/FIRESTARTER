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
Options `--start-delta (default 5000ms)` and `--stop-delta (default 2000ms)` specify a time in milliseconds in which metric values should be ignored.
After a run the output will be given in csv format to stdout.

### Example

Measure all available metrics for 15 minutes disregarding the first 5 minutes and last two seconds (default to `--stop-delta`).
```
FIRESTARTER --measurement --start-delta=300000 -t 900
```

## Optimization

FIRESTARTER has the option to optimize itself.
It currently supports the multiobjective algorithm NSGA2, selected by `--optimize=NSGA2`.
The optimization relies on the execution of FIRESTARTER with a combination of instruction groups.
Available instruction groups can be listed with `--list-instruction-groups` or the default with `-a | --avail`.
Per default FIRESTARTER takes the instruction groups of the pre-optimized setting shown with `-a | --avail`.
The user may specify their own instruction groups with `--run-instruction-groups`.
The selected instruction groups will be used to preheat the CPU (default 240s, specify different value by setting `--preheat`).
During each test run of the duration specified by `-t | --timeout` metrics will collect information about the fitness.
Metrics used for optimization can be specified by `--optimization-metrics`.
The number of individuals (`--individuals`), as is the number of generations (`--generation`) is set 20 per default.
An output file with the results will be written to `{HOSTNAME}_${STARTTIME}.json` if the option `--optimize-outfile` is not given.

The NSGA2 algorithm, as described in [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://dl.acm.org/doi/10.1109/4235.996017), is a multiobjective algorithms allowing FIRESTARTER to optimize with two metrics.
This is relavant as highest power consumption can be achieved by both optimizing for a high IPC (instruction per cycle) and high power consumption.
Parameters of the algorithm can be tweaked with `--nsga2-cr` and `--nsga2-m`.

### Examples

Optimize FIRESTARTER with NSGA2 and `sysfs-powercap-rapl` and `perf-ipc` metric. The duration for the evaluation of a setting is 20s long. The default instruction-groups for the current platform will be used. (Show them with `-a | --avail`)
```
FIRESTARTER -t 20 --optimize=NSGA2 --optimization-metric sysfs-powercap-rapl,perf-ipc
```

If `perf-ipc` is not available use `ipc-estimate`
```
FIRESTARTER -t 20 --optimize=NSGA2 --optimization-metric sysfs-powercap-rapl,ipc-estimate
```

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
