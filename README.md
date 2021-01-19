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

The Linux version of FIRESTARTER supports to collect metrics during runtime.
Available metrics can be shown with `--list-metrics`.
Default metrics are `perf-ipc`, `perf-freq`, `ipc-estimate` and `sysfs-powercap-rapl`.

### Custom metrics

If one would like to use custom metrics, e.g. an external power measurement, `--metric-from-stdin=NAME` allows metric values to be passed via stdin in the following format:
`{NAME} {TIME SINCE EPOCH IN NS} {ABSOLUT METRIC VALUE (double)}\n`.
See [here](examples/test_metric.py) for a basic example.

## Measurement

The Linux version of FIRESTARTER has the option to output the collected metric values by specifying `--measurement`.
Options `--start-delta (default 5000ms)` and `--stop-delta (default 2000ms)` specify a time in milliseconds in which metric values should be ignored.
After a run, the output will be given in CSV format to stdout.

### Example

Measure all available metrics for 15 minutes disregarding the first 5 minutes and last two seconds (default to `--stop-delta`).
```
FIRESTARTER --measurement --start-delta=300000 -t 900
```

## Optimization

The Linux version of FIRESTARTER has the option to optimize itself using evolutionary algorithms.
It currently supports the multiobjective algorithm NSGA2, selected by `--optimize=NSGA2`.

The evolutionary algorithm evaluates individuals one after another.
Each evaluation of a given individual is `-t | --timeout` seconds long.
Selecting a long enough time for letting the power consumption stabilize, but not too long as this will leed to a much longer optimization timespan.
During this time metrics are collected and the selected metrics (`--optimization-metrics`) are used for assigning a fitness. (Specify multiple times of a multiobjective algorithm.)
Using a high precision power metric is essential for good optimization results.
If you use the `sysfs-powercap-rapl` metric make sure the RAPL values of your CPU microarchitecture have a close correlation to the actual power consumption, e.g. Intel® processors<sup>[1](#intel-trademark)</sup> since Haswell, see [An Energy Efficiency Feature Survey of the Intel Haswell Processor](http://dx.doi.org/10.1109/IPDPSW.2015.70).

Individuals are made of different instruction groups and their ratios to one another.
Without specifying the `--run-instruction-groups` option, preselected instruction groups will be used for optimization.
Setting this option allows the user to select different instruction group, e.g. for optimizing FIRESTARTER on a not yet optimized microarchitecture.
The format of this is the same as shown by `-a | --avail` (Show all supported microarchitectures and their optimized individual.)
All available instruction groups can be listed with `--list-instruction-groups`.

The number of individuals per generation (`--individuals`), as is the number of generations (`--generation`) is set 20 per default.

Before the optimization runs a user-defined period of preheating of the CPU is carried out.
Option `--preheat` is specified to a default of 240 seconds.

After the optimization finishes the acquired data will be written to `{HOSTNAME}_${STARTTIME}.json` if not specified otherwise with the option `--optimize-outfile`.
An [IPython Notebook](examples/Evaluation_Notebook/Evaluation_Notebook.ipynb) is provided for basic visualization.

### NSGA2

The NSGA2 algorithm, as described in [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://dl.acm.org/doi/10.1109/4235.996017), is a multiobjective algorithm allowing FIRESTARTER to optimize with two (or more) metrics.
This is relevant as the highest power consumption can be achieved by both optimizing for a high IPC (instruction per cycle) and high power consumption.
Parameters of the algorithm can be tweaked with `--nsga2-cr` and `--nsga2-m`.

### Examples

Optimize FIRESTARTER with NSGA2 and `sysfs-powercap-rapl` and `perf-ipc` metric. The duration for the evaluation of a setting is 20s long. The default instruction groups for the current platform will be used. (Show them with `-a | --avail`)
```
FIRESTARTER -t 20 --optimize=NSGA2 --optimization-metric sysfs-powercap-rapl,perf-ipc
```

If `perf-ipc` is not available use `ipc-estimate`
```
FIRESTARTER -t 20 --optimize=NSGA2 --optimization-metric sysfs-powercap-rapl,ipc-estimate
```

## Help output for Linux (non CUDA version)
```
Usage:
  FIRESTARTER [OPTION...]

  -h, --help                    Display usage information
  -v, --version                 Display version information
  -c, --copyright               Display copyright information
  -w, --warranty                Display warranty information
  -q, --quiet                   Set log level to Warning
  -r, --report                  Display additional information (overridden by
                                -q)
      --debug                   Print debug output
  -a, --avail                   List available functions
  -i, --function ID             Specify integer ID of the load-function to be
                                used (as listed by --avail) (default: 0)
  -t, --timeout TIMEOUT         Set the timeout (seconds) after which
                                FIRESTARTER terminates itself, default: no timeout
                                (default: 0)
  -l, --load LOAD               Set the percentage of high CPU load to LOAD
                                (%) default: 100, valid values: 0 <= LOAD <=
                                100, threads will be idle in the remaining time,
                                frequency of load changes is determined by
                                -p. (default: 100)
  -p, --period PERIOD           Set the interval length for CPUs to PERIOD
                                (usec), default: 100000, each interval contains
                                a high load and an idle phase, the percentage
                                of high load is defined by -l (default:
                                100000)
  -n, --threads COUNT           Specify the number of threads. Cannot be
                                combined with -b | --bind, which impicitly
                                specifies the number of threads (default: 0)
  -b, --bind CPULIST            Select certain CPUs. CPULIST format: "x,y,z",
                                "x-y", "x-y/step", and any combination of the
                                above. Cannot be combined with -n |
                                --threads. (default: "")
      --list-instruction-groups
                                List the available instruction groups for the
                                payload of the current platform.
      --run-instruction-groups GROUPS
                                Run the payload with the specified
                                instruction groups. GROUPS format: multiple INST:VAL
                                pairs comma-seperated (default: "")
      --set-line-count arg      Set the number of lines for a payload.
      --allow-unavailable-payload
                                This option is only for debugging. Do not use
                                it.
      --dump-registers [=DELAY(=10)]
                                Dump the working registers on the first
                                thread. Depending on the payload these are mm, xmm,
                                ymm or zmm. Only use it without a timeout and
                                100 percent load. DELAY between dumps in
                                secs.
      --dump-registers-outpath arg
                                Path for the dump of the output files. If
                                path is not given, current working directory will
                                be used. (default: "")
      --list-metrics            List the available metrics.
      --metric-from-stdin NAME  Add a metric NAME with values from stdin.
                                Format of input: "NAME TIME_SINCE_EPOCH VALUE\n".
                                TIME_SINCE_EPOCH is a int64 in nanoseconds.
                                VALUE is a double. (Do not forget to flush
                                lines!)
      --measurement             Start a measurement for the time specified by
                                -t | --timeout. (The timeout must be greater
                                than the start and stop deltas.) Cannot be
                                combined with --optimize.
      --measurement-interval arg
                                Interval of measurements in milliseconds.
                                (default: 100)
      --start-delta N           Cut of first N milliseconds of measurement.
                                (default: 5000)
      --stop-delta N            Cut of last N milliseconds of measurement.
                                (default: 2000)
      --preheat N               Preheat for N seconds. (default: 240)
      --optimize arg            Run the optimization with one of these
                                algorithms: NSGA2. Cannot be combined with
                                --measurement.
      --optimize-outfile arg    Dump the output of the optimization into this
                                file. (Default: $PWD/$HOSTNAME_$DATE.json)
      --optimization-metric arg
                                Use a metric for optimization. Metrics listed
                                with cli argument --list-metrics or specified
                                with --metric-from-stdin are valid.
      --individuals arg         Number of individuals for the population. For
                                NSGA2 specify at least 5 and a multiple of 4.
                                (default: 20)
      --generations arg         Number of generations. (default: 20)
      --nsga2-cr arg            Crossover probability. (Must be in range
                                [0,1[) (default: 0.6)
      --nsga2-m arg             Mutation probability. (Must be in range
                                [0,1]) (default: 0.4)

Examples:

  ./FIRESTARTER                 starts FIRESTARTER without timeout
  ./FIRESTARTER -t 300          starts a 5 minute run of FIRESTARTER
  ./FIRESTARTER -l 50 -t 600    starts a 10 minute run of FIRESTARTER with
                                50% high load and 50% idle time
  ./FIRESTARTER -l 75 -p 20000000
                                starts FIRESTARTER with an interval length
                                of 2 sec, 1.5s high load and 0.5s idle
  ./FIRESTARTER --measurement --start-delta=300000 -t 900
                                starts FIRESTARTER measuring all available
                                metrics for 15 minutes disregarding the first
                                5 minutes and last two seconds (default to `--stop-delta`)
  ./FIRESTARTER -t 20 --optimize=NSGA2 --optimization-metric sysfs-powercap-rapl,perf-ipc
                                starts FIRESTARTER optimizing with the sysfs-powercap-rapl
                                and perf-ipc metric. The duration is 20s long. The default
                                instruction groups for the current platform will be used.
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

---

<a name="intel-trademark">1</a>: Intel is a trademark of Intel Corporation or its subsidiaries.
