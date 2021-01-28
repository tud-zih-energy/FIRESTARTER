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

#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#include <cxxopts.hpp>

#include <string>

struct Config {
  inline static const std::vector<std::pair<std::string, std::string>>
      optionsMap = {{"information", "Information Options:\n"},
                    {"general", "General Options:\n"},
                    {"specialized-workloads", "Specialized workloads:\n"},
#ifdef FIRESTARTER_DEBUG_FEATURES
                    {"debug", "Debugging:\n"},
#endif
#if defined(linux) || defined(__linux__)
                    {"measurement", "Measurement:\n"},
                    {"optimization", "Optimization:\n"}
#endif
  };

  // default parameters
  std::chrono::seconds timeout;
  unsigned loadPercent;
  std::chrono::microseconds period;
  unsigned requestedNumThreads;
  std::string cpuBind = "";
  bool printFunctionSummary;
  unsigned functionId;
  bool listInstructionGroups;
  std::string instructionGroups;
  unsigned lineCount = 0;
  // debug features
  bool allowUnavailablePayload = false;
  bool dumpRegisters = false;
  std::chrono::seconds dumpRegistersTimeDelta = std::chrono::seconds(0);
  std::string dumpRegistersOutpath = "";
  // CUDA parameters
  int gpus = 0;
  unsigned gpuMatrixSize = 0;
  bool gpuUseFloat = false;
  bool gpuUseDouble = false;
  // linux features
  bool listMetrics = false;
  bool measurement = false;
  std::chrono::milliseconds startDelta = std::chrono::milliseconds(0);
  std::chrono::milliseconds stopDelta = std::chrono::milliseconds(0);
  std::chrono::milliseconds measurementInterval = std::chrono::milliseconds(0);
  std::vector<std::string> stdinMetrics;
  // linux and dynamic linked binary
  std::vector<std::string> metricPaths;

  // optimization
  bool optimize = false;
  std::chrono::seconds preheat;
  std::string optimizationAlgorithm;
  std::vector<std::string> optimizationMetrics;
  std::chrono::seconds evaluationDuration;
  unsigned individuals;
  std::string optimizeOutfile = "";
  unsigned generations;
  double nsga2_cr;
  double nsga2_m;

  Config(int argc, const char **argv);
};

void print_copyright() {
  firestarter::log::info()
      << "This program is free software: you can redistribute it and/or "
         "modify\n"
      << "it under the terms of the GNU General Public License as published "
         "by\n"
      << "the Free Software Foundation, either version 3 of the License, or\n"
      << "(at your option) any later version.\n"
      << "\n"
      << "You should have received a copy of the GNU General Public License\n"
      << "along with this program.  If not, see "
         "<http://www.gnu.org/licenses/>.\n";
}

void print_warranty() {
  firestarter::log::info()
      << "This program is distributed in the hope that it will be useful,\n"
      << "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
      << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
      << "GNU General Public License for more details.\n"
      << "\n"
      << "You should have received a copy of the GNU General Public License\n"
      << "along with this program.  If not, see "
         "<http://www.gnu.org/licenses/>.\n";
}

void print_help(cxxopts::Options const &parser, std::string const &section) {
  std::vector<std::pair<std::string, std::string>> options(
      Config::optionsMap.size());

  if (section.size() == 0) {
    std::copy(Config::optionsMap.begin(), Config::optionsMap.end(),
              options.begin());
  } else {
    auto findSection = [&](std::pair<std::string, std::string> const &pair) {
      return pair.first == section;
    };
    auto it = std::copy_if(Config::optionsMap.begin(), Config::optionsMap.end(),
                           options.begin(), findSection);
    options.resize(std::distance(options.begin(), it));
  }

  // clang-format off
  firestarter::log::info()
    << parser.help(options)
    << "Examples:\n"
    << "  ./FIRESTARTER                 starts FIRESTARTER without timeout\n"
    << "  ./FIRESTARTER -t 300          starts a 5 minute run of FIRESTARTER\n"
    << "  ./FIRESTARTER -l 50 -t 600    starts a 10 minute run of FIRESTARTER with\n"
    << "                                50\% high load and 50\% idle time\n"
#ifdef FIRESTARTER_BUILD_CUDA
    << "                                on CPUs and full load on GPUs\n"
#endif
    << "  ./FIRESTARTER -l 75 -p 20000000\n"
    << "                                starts FIRESTARTER with an interval length\n"
    << "                                of 2 sec, 1.5s high load"
#ifdef FIRESTARTER_BUILD_CUDA
    << "                                on CPUs and full load on GPUs\n"
#else
    << "\n"
#endif
#if defined(linux) || defined(__linux__) 
    << "  ./FIRESTARTER --measurement --start-delta=300000 -t 900\n"
    << "                                starts FIRESTARTER measuring all available\n"
    << "                                metrics for 15 minutes disregarding the first\n"
    << "                                5 minutes and last two seconds (default to `--stop-delta`)\n"
    << "  ./FIRESTARTER -t 20 --optimize=NSGA2 --optimization-metric sysfs-powercap-rapl,perf-ipc\n"
    << "                                starts FIRESTARTER optimizing with the sysfs-powercap-rapl\n"
    << "                                and perf-ipc metric. The duration is 20s long. The default\n"
    << "                                instruction groups for the current platform will be used.\n"
#endif
    ;
  // clang-format on
}

Config::Config(int argc, const char **argv) {

  cxxopts::Options parser(argv[0]);

  // clang-format off
  parser.add_options("information")
    ("h,help", "Display usage information. SECTION can be any of: information | general | specialized-workloads | debug\n| measurement | optimization",
      cxxopts::value<std::string>()->implicit_value(""), "SECTION")
    ("v,version", "Display version information")
    ("c,copyright", "Display copyright information")
    ("w,warranty", "Display warranty information")
    ("q,quiet", "Set log level to Warning")
    ("r,report", "Display additional information (overridden by -q)")
    ("debug", "Print debug output")
    ("a,avail", "List available functions");

  parser.add_options("general")
    ("i,function", "Specify integer ID of the load-function to be\nused (as listed by --avail)",
      cxxopts::value<unsigned>()->default_value("0"), "ID")
#ifdef FIRESTARTER_BUILD_CUDA
    ("f,usegpufloat", "Use single precision matrix multiplications\ninstead of default")
    ("d,usegpudouble", "Use double precision matrix multiplications\ninstead of default")
    ("g,gpus", "Number of gpus to use, default: -1 (all)",
      cxxopts::value<int>()->default_value("-1"))
    ("m,matrixsize", "Size of the matrix to calculate, default: 0 (maximum)",
      cxxopts::value<unsigned>()->default_value("0"))
#endif
    ("t,timeout", "Set the timeout (seconds) after which FIRESTARTER\nterminates itself, default: 0 (no timeout)",
      cxxopts::value<unsigned>()->default_value("0"), "TIMEOUT")
    ("l,load", "Set the percentage of high CPU load to LOAD\n(%) default: 100, valid values: 0 <= LOAD <=\n100, threads will be idle in the remaining time,\nfrequency of load changes is determined by -p."
#ifdef FIRESTARTER_BUILD_CUDA
     " This option does NOT influence the GPU\nworkload!"
#endif
     , cxxopts::value<unsigned>()->default_value("100"), "LOAD")
    ("p,period", "Set the interval length for CPUs to PERIOD\n(usec), default: 100000, each interval contains\na high load and an idle phase, the percentage\nof high load is defined by -l.",
      cxxopts::value<unsigned>()->default_value("100000"), "PERIOD")
    ("n,threads", "Specify the number of threads. Cannot be\ncombined with -b | --bind, which impicitly\nspecifies the number of threads.",
      cxxopts::value<unsigned>()->default_value("0"), "COUNT")
#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
    ("b,bind", "Select certain CPUs. CPULIST format: \"x,y,z\",\n\"x-y\", \"x-y/step\", and any combination of the\nabove. Cannot be combined with -n | --threads.",
      cxxopts::value<std::string>()->default_value(""), "CPULIST")
#endif
    ;

  parser.add_options("specialized-workloads")
    ("list-instruction-groups", "List the available instruction groups for the\npayload of the current platform.")
    ("run-instruction-groups", "Run the payload with the specified\ninstruction groups. GROUPS format: multiple INST:VAL\npairs comma-seperated.",
      cxxopts::value<std::string>()->default_value(""), "GROUPS")
    ("set-line-count", "Set the number of lines for a payload.",
      cxxopts::value<unsigned>());

#ifdef FIRESTARTER_DEBUG_FEATURES
  parser.add_options("debug")
    ("allow-unavailable-payload", "")
    ("dump-registers", "Dump the working registers on the first\nthread. Depending on the payload these are mm, xmm,\nymm or zmm. Only use it without a timeout and\n100 percent load. DELAY between dumps in secs.",
      cxxopts::value<unsigned>()->implicit_value("10"), "DELAY")
    ("dump-registers-outpath", "Path for the dump of the output files. If\nPATH is not given, current working directory will\nbe used.",
      cxxopts::value<std::string>()->default_value(""), "PATH");
#endif

#if defined(linux) || defined(__linux__)
  parser.add_options("measurement")
    ("list-metrics", "List the available metrics.")
#ifndef FIRESTARTER_LINK_STATIC
    ("metric-path", "Add a path to a shared library representing an interface for a metric. This option can be specified multiple times.",
      cxxopts::value<std::vector<std::string>>()->default_value(""))
#endif
    ("metric-from-stdin", "Add a metric NAME with values from stdin.\nFormat of input: \"NAME TIME_SINCE_EPOCH VALUE\\n\".\nTIME_SINCE_EPOCH is a int64 in nanoseconds. VALUE is a double. (Do not forget to flush\nlines!)",
      cxxopts::value<std::vector<std::string>>(), "NAME")
    ("measurement", "Start a measurement for the time specified by\n-t | --timeout. (The timeout must be greater\nthan the start and stop deltas.) Cannot be\ncombined with --optimize.")
    ("measurement-interval", "Interval of measurements in milliseconds, default: 100",
      cxxopts::value<unsigned>()->default_value("100"))
    ("start-delta", "Cut of first N milliseconds of measurement, default: 5000",
      cxxopts::value<unsigned>()->default_value("5000"), "N")
    ("stop-delta", "Cut of last N milliseconds of measurement, default: 2000",
      cxxopts::value<unsigned>()->default_value("2000"), "N")
    ("preheat", "Preheat for N seconds, default: 240",
      cxxopts::value<unsigned>()->default_value("240"), "N");

  parser.add_options("optimization")
    ("optimize", "Run the optimization with one of these algorithms: NSGA2.\nCannot be combined with --measurement.",
      cxxopts::value<std::string>())
    ("optimize-outfile", "Dump the output of the optimization into this\nfile, default: $PWD/$HOSTNAME_$DATE.json",
      cxxopts::value<std::string>())
    ("optimization-metric", "Use a metric for optimization. Metrics listed\nwith cli argument --list-metrics or specified\nwith --metric-from-stdin are valid.",
      cxxopts::value<std::vector<std::string>>())
    ("individuals", "Number of individuals for the population. For\nNSGA2 specify at least 5 and a multiple of 4,\ndefault: 20",
      cxxopts::value<unsigned>()->default_value("20"))
    ("generations", "Number of generations, default: 20",
      cxxopts::value<unsigned>()->default_value("20"))
    ("nsga2-cr", "Crossover probability. Must be in range [0,1[\ndefault: 0.6",
      cxxopts::value<double>()->default_value("0.6"))
    ("nsga2-m", "Mutation probability. Must be in range [0,1]\ndefault: 0.4",
      cxxopts::value<double>()->default_value("0.4"));
#endif
  // clang-format on

  try {
    auto options = parser.parse(argc, argv);

    if (options.count("quiet")) {
      firestarter::logging::filter<firestarter::logging::record>::set_severity(
          nitro::log::severity_level::warn);
    } else if (options.count("report")) {
      firestarter::logging::filter<firestarter::logging::record>::set_severity(
          nitro::log::severity_level::debug);
    } else if (options.count("debug")) {
      firestarter::logging::filter<firestarter::logging::record>::set_severity(
          nitro::log::severity_level::trace);
    } else {
      firestarter::logging::filter<firestarter::logging::record>::set_severity(
          nitro::log::severity_level::info);
    }

    if (options.count("version")) {
      std::exit(EXIT_SUCCESS);
    }

    if (options.count("copyright")) {
      print_copyright();
      std::exit(EXIT_SUCCESS);
    }

    if (options.count("warranty")) {
      print_warranty();
      std::exit(EXIT_SUCCESS);
    }

    firestarter::log::info()
        << "This program comes with ABSOLUTELY NO WARRANTY; for details run `"
        << argv[0] << " -w`.\n"
        << "This is free software, and you are welcome to redistribute it\n"
        << "under certain conditions; run `" << argv[0]
        << " -c` for details.\n";

    if (options.count("help")) {
      auto section = options["help"].as<std::string>();

      // section not found
      auto findSection = [&](std::pair<std::string, std::string> const &pair) {
        return pair.first == section;
      };
      if (std::find_if(optionsMap.begin(), optionsMap.end(), findSection) ==
              optionsMap.end() &&
          section.size() != 0) {
        throw std::invalid_argument("Section \"" + section +
                                    "\" not found in help.");
      }

      print_help(parser, section);
      std::exit(EXIT_SUCCESS);
    }

    timeout = std::chrono::seconds(options["timeout"].as<unsigned>());
    loadPercent = options["load"].as<unsigned>();
    period = std::chrono::microseconds(options["period"].as<unsigned>());

    if (loadPercent > 100) {
      throw std::invalid_argument("Option -l/--load may not be above 100.");
    }

#ifdef FIRESTARTER_DEBUG_FEATURES
    dumpRegisters = options.count("dump-registers");
    if (dumpRegisters) {
      dumpRegistersTimeDelta =
          std::chrono::seconds(options["dump-registers"].as<unsigned>());
      if (timeout != std::chrono::microseconds::zero() && loadPercent != 100) {
        throw std::invalid_argument("Option --dump-registers may only be used "
                                    "without a timeout and full load.");
      }
    }
    allowUnavailablePayload = options.count("allow-unavailable-payload");
#endif

    requestedNumThreads = options["threads"].as<unsigned>();

#if (defined(linux) || defined(__linux__)) &&                                  \
    defined(FIRESTARTER_THREAD_AFFINITY)
    cpuBind = options["bind"].as<std::string>();
    if (!cpuBind.empty()) {
      if (requestedNumThreads != 0) {
        throw std::invalid_argument(
            "Options -b/--bind and -n/--threads cannot be used together.");
      }
    }
#endif

#ifdef FIRESTARTER_BUILD_CUDA
    gpuUseFloat = options.count("usegpufloat");
    gpuUseDouble = options.count("usegpudouble");

    if (gpuUseFloat && gpuUseDouble) {
      throw std::invalid_argument("Options -f/--usegpufloat and "
                                  "-d/--usegpudouble cannot be used together.");
    }

    gpuMatrixSize = options["matrixsize"].as<unsigned>();
    if (gpuMatrixSize > 0 && gpuMatrixSize < 64) {
      throw std::invalid_argument(
          "Option -m/--matrixsize may not be below 64.");
    }

    gpus = options["gpus"].as<int>();
#endif

    printFunctionSummary = options.count("avail");

    functionId = options["function"].as<unsigned>();

    listInstructionGroups = options.count("list-instruction-groups");
    instructionGroups = options["run-instruction-groups"].as<std::string>();
    if (options.count("set-line-count")) {
      lineCount = options["set-line-count"].as<unsigned>();
    }

#if defined(linux) || defined(__linux__)
    startDelta =
        std::chrono::milliseconds(options["start-delta"].as<unsigned>());
    stopDelta = std::chrono::milliseconds(options["stop-delta"].as<unsigned>());
    measurementInterval = std::chrono::milliseconds(
        options["measurement-interval"].as<unsigned>());
#ifndef FIRESTARTER_LINK_STATIC
    metricPaths = options["metric-path"].as<std::vector<std::string>>();
#endif
    if (options.count("metric-from-stdin")) {
      stdinMetrics =
          options["metric-from-stdin"].as<std::vector<std::string>>();
    }
    measurement = options.count("measurement");
    listMetrics = options.count("list-metrics");

    if ((optimize = options.count("optimize"))) {
      if (measurement) {
        throw std::invalid_argument(
            "Options --measurement and --optimize cannot be used together.");
      }
      preheat = std::chrono::seconds(options["preheat"].as<unsigned>());
      optimizationAlgorithm = options["optimize"].as<std::string>();
      if (options.count("optimization-metric")) {
        optimizationMetrics =
            options["optimization-metric"].as<std::vector<std::string>>();
      }
      if (loadPercent != 100) {
        throw std::invalid_argument("Options -p | --period and -l | --load are "
                                    "not compatible with --optimize.");
      }
      if (timeout == std::chrono::seconds::zero()) {
        throw std::invalid_argument(
            "Option -t | --timeout must be specified for optimization.");
      }
      evaluationDuration = timeout;
      // this will deactivate the watchdog worker
      timeout = std::chrono::seconds::zero();
      individuals = options["individuals"].as<unsigned>();
      if (options.count("optimize-outfile")) {
        optimizeOutfile = options["optimize-outfile"].as<std::string>();
      }
      generations = options["generations"].as<unsigned>();
      nsga2_cr = options["nsga2-cr"].as<double>();
      nsga2_m = options["nsga2-m"].as<double>();

      if (optimizationAlgorithm != "NSGA2") {
        throw std::invalid_argument("Option --optimize must be any of: NSGA2");
      }
    }
#endif

  } catch (std::exception &e) {
    firestarter::log::error() << e.what() << "\n";
    print_help(parser, "");
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, const char **argv) {

  firestarter::log::info()
      << "FIRESTARTER - A Processor Stress Test Utility, Version "
      << _FIRESTARTER_VERSION_STRING << "\n"
      << "Copyright (C) " << _FIRESTARTER_BUILD_YEAR
      << " TU Dresden, Center for Information Services and High Performance "
         "Computing"
      << "\n";

  Config cfg{argc, argv};

  try {
    firestarter::Firestarter firestarter(
        argc, argv, cfg.timeout, cfg.loadPercent, cfg.period,
        cfg.requestedNumThreads, cfg.cpuBind, cfg.printFunctionSummary,
        cfg.functionId, cfg.listInstructionGroups, cfg.instructionGroups,
        cfg.lineCount, cfg.allowUnavailablePayload, cfg.dumpRegisters,
        cfg.dumpRegistersTimeDelta, cfg.dumpRegistersOutpath, cfg.gpus,
        cfg.gpuMatrixSize, cfg.gpuUseFloat, cfg.gpuUseDouble, cfg.listMetrics,
        cfg.measurement, cfg.startDelta, cfg.stopDelta, cfg.measurementInterval,
        cfg.metricPaths, cfg.stdinMetrics, cfg.optimize, cfg.preheat,
        cfg.optimizationAlgorithm, cfg.optimizationMetrics,
        cfg.evaluationDuration, cfg.individuals, cfg.optimizeOutfile,
        cfg.generations, cfg.nsga2_cr, cfg.nsga2_m);

    firestarter.mainThread();

  } catch (std::exception const &e) {
    firestarter::log::error() << e.what();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
