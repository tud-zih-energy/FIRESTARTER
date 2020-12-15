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
  unsigned lineCount;
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
  // linux and dynamic linked binary
  std::vector<std::string> metricPaths;

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

void print_help(cxxopts::Options const &parser) {
  // clang-format off
  firestarter::log::info()
    << parser.help()
		<< "\n"
    << "Examples:\n"
    << "\n"
    << "  ./FIRESTARTER                 starts FIRESTARTER without timeout\n"
    << "  ./FIRESTARTER -t 300          starts a 5 minute run of FIRESTARTER\n"
    << "  ./FIRESTARTER -l 50 -t 600    starts a 10 minute run of FIRESTARTER with\n"
		<< "                                50\% high load and 50\% idle time\n"
#ifdef FIRESTARTER_BUILD_CUDA
    << "                                on CPUs and full load on GPUs\n"
#endif
    << "  ./FIRESTARTER -l 75 -p 20000000\n"
    << "                                starts FIRESTARTER with an interval length\n"
    << "                                of 2 sec, 1.5s high load and 0.5s idle\n"
#ifdef FIRESTARTER_BUILD_CUDA
    << "                                on CPUs and full load on GPUs\n"
#endif
    ;
  // clang-format on
}

Config::Config(int argc, const char **argv) {

  cxxopts::Options parser(argv[0]);

  // clang-format off
  parser.add_options()
    ("h,help", "Display usage information")
    ("v,version", "Display version information")
    ("c,copyright", "Display copyright information")
    ("w,warranty", "Display warranty information")
    ("q,quiet", "Set log level to Warning")
    ("r,report", "Display additional information (overridden by -q)")
    ("debug", "Print debug output")
    ("a,avail", "List available functions")
    ("i,function", "Specify integer ID of the load-function to be used (as listed by --avail)",
      cxxopts::value<unsigned>()->default_value("0"), "ID")
#ifdef FIRESTARTER_BUILD_CUDA
    ("f,usegpufloat", "Use single precision matrix multiplications instead of default")
    ("d,usegpudouble", "Use double precision matrix multiplications instead of default")
    ("g,gpus", "Number of gpus to use (default: all)",
      cxxopts::value<int>()->default_value("-1"))
    ("m,matrixsize", "Size of the matrix to calculate, default is maximum",
      cxxopts::value<unsigned>()->default_value("0"))
#endif
    ("t,timeout", "Set the timeout (seconds) after which FIRESTARTER terminates itself, default: no timeout",
      cxxopts::value<unsigned>()->default_value("0"), "TIMEOUT")
    ("l,load", "Set the percentage of high CPU load to LOAD (%) default: 100, valid values: 0 <= LOAD <= 100, threads will be idle in the remaining time, frequency of load changes is determined by -p."
#ifdef FIRESTARTER_BUILD_CUDA
     " This option does NOT influence the GPU workload!"
#endif
     , cxxopts::value<unsigned>()->default_value("100"), "LOAD")
    ("p,period", "Set the interval length for CPUs to PERIOD (usec), default: 100000, each interval contains a high load and an idle phase, the percentage of high load is defined by -l",
      cxxopts::value<unsigned>()->default_value("100000"), "PERIOD")
    ("n,threads", "Specify the number of threads. Cannot be combined with -b | --bind, which impicitly specifies the number of threads",
      cxxopts::value<unsigned>()->default_value("0"), "COUNT")
#if (defined(linux) || defined(__linux__)) && defined(FIRESTARTER_THREAD_AFFINITY)
    ("b,bind", "Select certain CPUs. CPULIST format: \"x,y,z\", \"x-y\", \"x-y/step\", and any combination of the above. Cannot be comibned with -n | --threads.",
      cxxopts::value<std::string>()->default_value(""), "CPULIST")
#endif
    ("list-instruction-groups", "List the available instruction groups for the payload of the current platform.")
    ("run-instruction-groups", "Run the payload with the specified instruction groups. GROUPS format: multiple INST:VAL pairs comma-seperated",
      cxxopts::value<std::string>()->default_value(""), "GROUPS")
    ("set-line-count", "Set the number of lines for a payload.",
      cxxopts::value<unsigned>()->default_value("0"))
#ifdef FIRESTARTER_DEBUG_FEATURES
    ("allow-unavailable-payload", "This option is only for debugging. Do not use it.")
    ("dump-registers", "Dump the working registers on the first thread. Depending on the payload these are mm, xmm, ymm or zmm. Only use it without a timeout and 100 percent load. DELAY between dumps in secs.",
      cxxopts::value<unsigned>()->implicit_value("10"), "DELAY")
    ("dump-registers-outpath", "Path for the dump of the output files. If path is not given, current working directory will be used.",
      cxxopts::value<std::string>()->default_value(""))
#endif
#if defined(linux) || defined(__linux__)
    ("list-metrics", "List the available metrics.")
#ifndef FIRESTARTER_LINK_STATIC
    ("metric-path", "Add a path to a shared library representing an interface for a metric. This option can be specified multiple times.",
      cxxopts::value<std::vector<std::string>>()->default_value(""))
#endif
    ("measurement", "Start a measurement for the time specified by -t | --timeout. (The timeout must be greater than the start and stop deltas.)")
    ("measurement-interval", "Interval of measurements in milliseconds.",
      cxxopts::value<unsigned>()->default_value("100"))
    ("start-delta", "Cut of first N milliseconds of measurement.",
      cxxopts::value<unsigned>()->default_value("5000"), "N")
    ("stop-delta", "Cut of last N milliseconds of measurement.",
      cxxopts::value<unsigned>()->default_value("2000"), "N")
#endif
    ;
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
      print_help(parser);
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
    if (matrixSize > 0 && matrixSize < 64) {
      throw std::invalid_argument(
          "Option -m/--matrixsize may not be below 64.");
    }

    gpus = options["gpus"].as<int>();
#endif

    printFunctionSummary = options.count("avail");

    functionId = options["function"].as<unsigned>();

    listInstructionGroups = options.count("list-instruction-groups");
    instructionGroups = options["run-instruction-groups"].as<std::string>();
    lineCount = options["set-line-count"].as<unsigned>();

#if defined(linux) || defined(__linux__)
    startDelta =
        std::chrono::milliseconds(options["start-delta"].as<unsigned>());
    stopDelta = std::chrono::milliseconds(options["stop-delta"].as<unsigned>());
    measurementInterval = std::chrono::milliseconds(
        options["measurement-interval"].as<unsigned>());
#ifndef FIRESTARTER_LINK_STATIC
    metricPaths = options["metric-path"].as<std::vector<std::string>>();
#endif
    measurement = options.count("measurement");
    listMetrics = options.count("list-metrics");
#endif

  } catch (std::exception &e) {
    firestarter::log::error() << e.what() << "\n";
    print_help(parser);
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
        cfg.timeout, cfg.loadPercent, cfg.period, cfg.requestedNumThreads,
        cfg.cpuBind, cfg.printFunctionSummary, cfg.functionId,
        cfg.listInstructionGroups, cfg.instructionGroups, cfg.lineCount,
        cfg.allowUnavailablePayload, cfg.dumpRegisters,
        cfg.dumpRegistersTimeDelta, cfg.dumpRegistersOutpath, cfg.gpus,
        cfg.gpuMatrixSize, cfg.gpuUseFloat, cfg.gpuUseDouble, cfg.listMetrics,
        cfg.measurement, cfg.startDelta, cfg.stopDelta, cfg.measurementInterval,
        cfg.metricPaths);

    firestarter.mainThread();

  } catch (std::runtime_error const &e) {
    firestarter::log::error() << e.what();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
