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

#if defined(linux) || defined(__linux__)
#include <firestarter/Measurement/MeasurementWorker.hpp>
#endif

#include <cxxopts.hpp>

#include <string>
#include <thread>

int print_copyright() {

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

  return EXIT_SUCCESS;
}

int print_warranty() {

  firestarter::log::info()
      << "This program is distributed in the hope that it will be useful,\n"
      << "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
      << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
      << "GNU General Public License for more details.\n"
      << "\n"
      << "You should have received a copy of the GNU General Public License\n"
      << "along with this program.  If not, see "
         "<http://www.gnu.org/licenses/>.\n";

  return EXIT_SUCCESS;
}

int print_help(cxxopts::Options parser) {

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
#ifdef CUDA
    << "                                on CPUs and full load on GPUs\n"
#endif
    << "  ./FIRESTARTER -l 75 -p 20000000\n"
    << "                                starts FIRESTARTER with an interval length\n"
    << "                                of 2 sec, 1.5s high load and 0.5s idle\n"
#ifdef CUDA
    << "                                on CPUs and full load on GPUs\n"
#endif
    ;
  // clang-format on

  return EXIT_SUCCESS;
}

int main(int argc, const char **argv) {

  // TODO: get year number on build
  firestarter::log::info()
      << "FIRESTARTER - A Processor Stress Test Utility, Version "
      << _FIRESTARTER_VERSION_STRING << "\n"
      << "Copyright (C) " << _FIRESTARTER_BUILD_YEAR
      << " TU Dresden, Center for Information Services and High Performance "
         "Computing"
      << "\n";

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
    ("f,usegpufloat", "Use single precision matrix multiplications instead of default",
      cxxopts::value<bool>()->default_value("false"))
    ("d,usegpudouble", "Use double precision matrix multiplications instead of default",
      cxxopts::value<bool>()->default_value("false"))
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
    ("list-instruction-groups", "List the available instruction groups for the payload of the current platform.",
      cxxopts::value<bool>()->default_value("false"))
    ("run-instruction-groups", "Run the payload with the specified instruction groups. GROUPS format: multiple INST:VAL pairs comma-seperated",
      cxxopts::value<std::string>()->default_value(""), "GROUPS")
    ("set-line-count", "Set the number of lines for a payload.",
      cxxopts::value<unsigned>()->default_value("0"))
#ifdef FIRESTARTER_DEBUG_FEATURES
    ("allow-unavailable-payload", "This option is only for debugging. Do not use it.",
      cxxopts::value<bool>()->default_value("false"))
    ("dump-registers", "Dump the working registers on the first thread. Depending on the payload these are mm, xmm, ymm or zmm. Only use it without a timeout and 100 percent load. DELAY between dumps in secs.",
      cxxopts::value<unsigned>()->implicit_value("10"), "DELAY")
    ("dump-registers-outpath", "Path for the dump of the output files. If path is not given, current working directory will be used.",
      cxxopts::value<std::string>()->default_value(""))
#endif
#if defined(linux) || defined(__linux__)
    ("list-metrics", "List the available metrics.",
      cxxopts::value<bool>()->default_value("false"))
#ifndef FIRESTARTER_LINK_STATIC
    ("metric-path", "Add a path to a shared library representing an interface for a metric. This option can be specified multiple times.",
      cxxopts::value<std::vector<std::string>>()->default_value(""))
#endif
    ("measurement", "Start a measurement for the time specified by -t | --timeout. (The timeout must be greater than the start and stop deltas.",
      cxxopts::value<bool>()->default_value("false"))
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
      return EXIT_SUCCESS;
    }

    if (options.count("copyright")) {
      return print_copyright();
    }

    if (options.count("warranty")) {
      return print_warranty();
    }

    firestarter::log::info()
        << "This program comes with ABSOLUTELY NO WARRANTY; for details run `"
        << argv[0] << " -w`.\n"
        << "This is free software, and you are welcome to redistribute it\n"
        << "under certain conditions; run `" << argv[0]
        << " -c` for details.\n";

    if (options.count("help")) {
      return print_help(parser);
    }

    std::chrono::seconds timeout(options["timeout"].as<unsigned>());
    unsigned loadPercent = options["load"].as<unsigned>();
    std::chrono::microseconds period(options["period"].as<unsigned>());

    if (loadPercent > 100) {
      throw std::invalid_argument("Option -l/--load may not be above 100.");
    }

    std::chrono::microseconds load = (period * loadPercent) / 100;
    if (loadPercent == 100 || load == std::chrono::microseconds::zero()) {
      period = std::chrono::microseconds::zero();
    }

#ifdef FIRESTARTER_DEBUG_FEATURES
    bool dumpRegisters = options.count("dump-registers");
    if (dumpRegisters) {
      if (timeout != std::chrono::microseconds::zero() && loadPercent != 100) {
        throw std::invalid_argument("Option --dump-registers may only be used "
                                    "without a timeout and full load.");
      }
    }
#else
    bool dumpRegisters = false;
#endif

    unsigned requestedNumThreads = options["threads"].as<unsigned>();

    std::string cpuBind = "";
#if (defined(linux) || defined(__linux__)) &&                                  \
    defined(FIRESTARTER_THREAD_AFFINITY)
    if (!options["bind"].as<std::string>().empty()) {
      if (options["threads"].as<unsigned>() != 0) {
        throw std::invalid_argument(
            "Options -b/--bind and -n/--threads cannot be used together.");
      }

      cpuBind = options["bind"].as<std::string>();
    }
#endif

    int returnCode;
    firestarter::Firestarter firestarter;

#ifdef FIRESTARTER_BUILD_CUDA
    bool useGpuFloat = options["usegpufloat"].as<bool>();
    bool useGpuDouble = options["usegpudouble"].as<bool>();

    if (useGpuFloat && useGpuDouble) {
      throw std::invalid_argument("Options -f/--usegpufloat and "
                                  "-d/--usegpudouble cannot be used together.");
    }

    if (useGpuFloat) {
      firestarter.gpuStructPointer->use_double = 0;
    } else if (useGpuDouble) {
      firestarter.gpuStructPointer->use_double = 1;
    } else {
      firestarter.gpuStructPointer->use_double = 2;
    }

    unsigned matrixSize = options["matrixsize"].as<unsigned>();
    if (matrixSize > 0 && matrixSize < 64) {
      throw std::invalid_argument(
          "Option -m/--matrixsize may not be below 64.");
    }
    firestarter.gpuStructPointer->msize = matrixSize;

    firestarter.gpuStructPointer->use_device = options["gpus"].as<int>();
#endif

    if (EXIT_SUCCESS !=
        (returnCode = firestarter.environment().evaluateCpuAffinity(
             requestedNumThreads, cpuBind))) {
      return returnCode;
    }

    firestarter.environment().evaluateFunctions();

    if (options.count("avail")) {
      firestarter.environment().printFunctionSummary();
      return EXIT_SUCCESS;
    }

    unsigned functionId = options["function"].as<unsigned>();
    bool allowUnavailablePayload =
        options["allow-unavailable-payload"].as<bool>();

    if (EXIT_SUCCESS != (returnCode = firestarter.environment().selectFunction(
                             functionId, allowUnavailablePayload))) {
      return returnCode;
    }

    if (options["list-instruction-groups"].as<bool>()) {
      firestarter.environment().printAvailableInstructionGroups();
      return EXIT_SUCCESS;
    }

    std::string instructionGroups =
        options["run-instruction-groups"].as<std::string>();
    if (!instructionGroups.empty()) {
      if (EXIT_SUCCESS !=
          (returnCode = firestarter.environment().selectInstructionGroups(
               instructionGroups))) {
        return returnCode;
      }
    }

    unsigned lineCount = options["set-line-count"].as<unsigned>();
    if (lineCount != 0) {
      firestarter.environment().setLineCount(lineCount);
    }

#if defined(linux) || defined(__linux__)
    auto startDelta =
        std::chrono::milliseconds(options["start-delta"].as<unsigned>());
    auto stopDelta =
        std::chrono::milliseconds(options["stop-delta"].as<unsigned>());
    auto measurementInterval = std::chrono::milliseconds(
        options["measurement-interval"].as<unsigned>());
#ifndef FIRESTARTER_LINK_STATIC
    auto metricPath = options["metric-path"].as<std::vector<std::string>>();
#else
    auto metricPath = std::vector<std::string>();
#endif

    firestarter::measurement::MeasurementWorker *measurementWorker = nullptr;

    if (options["measurement"].as<bool>() ||
        options["list-metrics"].as<bool>()) {
      measurementWorker = new firestarter::measurement::MeasurementWorker(
          measurementInterval, firestarter.environment().requestedNumThreads(),
          metricPath);

      if (options["list-metrics"].as<bool>()) {
        firestarter::log::info() << measurementWorker->availableMetrics();
        delete measurementWorker;
        return EXIT_SUCCESS;
      }

      // TODO: select the metrics
      // init all metrics
      auto count =
          measurementWorker->initMetrics(measurementWorker->metricNames());

      if (count == 0) {
        firestarter::log::error() << "No metrics initialized";
        delete measurementWorker;
        return EXIT_FAILURE;
      }
    }
#endif

    firestarter.environment().printSelectedCodePathSummary();

    firestarter::log::info() << firestarter.environment().topology();

    firestarter.environment().printThreadSummary();

    // setup thread with either high or low load configured at the start
    // low loads has to know the length of the period
    if (EXIT_SUCCESS !=
        (returnCode = firestarter.initLoadWorkers(
             (loadPercent == 0), period.count(), dumpRegisters))) {
      return returnCode;
    }

#ifdef FIRESTARTER_BUILD_CUDA
    pthread_t gpu_thread;
    pthread_create(&gpu_thread, NULL, firestarter::cuda::init_gpu,
                   (void *)firestarter.gpuStructPointer);
    while (firestarter.gpuStructPointer->loadingdone != 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#endif

#if defined(linux) || defined(__linux__)
    // if measurement is enabled, start it here
    if (nullptr != measurementWorker) {
      measurementWorker->startMeasurement();
    }
#endif

    firestarter.signalWork();

#ifdef FIRESTARTER_DEBUG_FEATURES
    if (dumpRegisters) {
      auto dumpTimeDelta = options["dump-registers"].as<unsigned>();
      if (EXIT_SUCCESS !=
          (returnCode = firestarter.initDumpRegisterWorker(
               std::chrono::seconds(dumpTimeDelta),
               options["dump-registers-outpath"].as<std::string>()))) {
        return returnCode;
      }
    }
#endif

    // worker thread for load control
    firestarter.watchdogWorker(period, load, timeout);

    // wait for watchdog to timeout or until user terminates
    firestarter.joinLoadWorkers();
#ifdef FIRESTARTER_DEBUG_FEATURES
    if (dumpRegisters) {
      firestarter.joinDumpRegisterWorker();
    }
#endif

    firestarter.printPerformanceReport();

#if defined(linux) || defined(__linux__)
    // if measurment is enabled, stop it here
    if (nullptr != measurementWorker) {
      // TODO: clear this up
      firestarter::log::info()
          << "metric,num_timepoints,duration_ms,average,stddev";
      for (auto const &[name, sum] :
           measurementWorker->getValues(startDelta, stopDelta)) {
        firestarter::log::info()
            << std::quoted(name) << "," << sum.num_timepoints << ","
            << sum.duration.count() << "," << sum.average << "," << sum.stddev;
      }

      delete measurementWorker;
    }
#endif

  } catch (std::exception &e) {
    firestarter::log::error() << e.what() << "\n";
    return print_help(parser);
  }

  return EXIT_SUCCESS;
}
