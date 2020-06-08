#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#include <cxxopts.hpp>

#include <string>

int print_copyright(void) {

  firestarter::log::info()
      << "\n"
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

int print_warranty(void) {

  firestarter::log::info()
      << "\n"
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

  firestarter::log::info() << parser.help();

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {

  // TODO: get year number on build
  firestarter::log::info()
      << "FIRESTARTER - A Processor Stress Test Utility, Version "
      << _FIRESTARTER_VERSION_STRING << "\n"
      << "Copyright (C) " << _FIRESTARTER_BUILD_YEAR
      << " TU Dresden, Center for Information Services and High Performance "
         "Computing"
      << "\n";

  cxxopts::Options parser(argv[0]);

  parser.add_options()("h,help", "Display usage information")(
      "v,version", "Display version information")(
      "c,copyright", "Display copyright information")(
      "w,warranty", "Display warranty information")(
      "d,debug", "Display debug output")("a,avail", "List available functions")(
      "i,function",
      "Specify integer ID of the load-function to be used (as listed by "
      "--avail)",
      cxxopts::value<unsigned>()->default_value("0"),
      "ID")("t,timeout",
            "Set the timeout (seconds) after which FIRESTARTER terminates "
            "itself, default: no timeout",
            cxxopts::value<unsigned>()->default_value("0"), "TIMEOUT")(
      "l,load",
      "Set the percentage of high CPU load to LOAD (%) default: 100, valid "
      "values: 0 <= LOAD <= 100, thredas will be idle in the remaining time, "
      "frequenc of load changes is determined by -p",
      cxxopts::value<unsigned>()->default_value("100"),
      "LOAD")("p,period",
              "Set the interval length for CPUs to PERIOD (usec), default: "
              "100000, each interval contains a high load and an idle phase, "
              "the percentage of high load is defined by -l",
              cxxopts::value<unsigned>()->default_value("100000"), "PERIOD")(
      "n,threads",
      "Specify the number of threads. Cannot be combined with -b | "
      "--bind, which impicitly specifies the number of threads",
      cxxopts::value<unsigned>()->default_value("0"), "COUNT")
#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
      ("b,bind",
       "Select certain CPUs. CPULIST format: \"x,y,z\", \"x-y\", \"x-y/step\", "
       "and any combination of the above. Cannot be comibned with -n | "
       "--threads.",
       cxxopts::value<std::string>()->default_value(""), "CPULIST")
#endif
      ;
  // TODO:
  // r report
  //
  // TODO: cuda
  // f: usegpufloat
  // g: gpus
  // m: matrixsize

  try {
    auto options = parser.parse(argc, argv);

    if (options.count("debug")) {
      firestarter::logging::filter<firestarter::logging::record>::set_severity(
          nitro::log::severity_level::debug);
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
    if (load == period || load == std::chrono::microseconds::zero()) {
      period = std::chrono::microseconds::zero();
    }

    unsigned requestedNumThreads = options["threads"].as<unsigned>();

    std::string cpuBind = "";
#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
    if (!options["bind"].as<std::string>().empty()) {
      if (options["threads"].as<unsigned>() != 0) {
        throw std::invalid_argument(
            "Options -b/--bind and -n/--threads cannot be used together.");
      }

      cpuBind = options["bind"].as<std::string>();
    }
#endif

    int returnCode;
    auto firestarter = new firestarter::Firestarter();

    if (EXIT_SUCCESS !=
        (returnCode = firestarter->environment->evaluateEnvironment())) {
      delete firestarter;
      return returnCode;
    }

    if (EXIT_SUCCESS !=
        (returnCode = firestarter->environment->evaluateCpuAffinity(
             requestedNumThreads, cpuBind))) {
      delete firestarter;
      return returnCode;
    }

    firestarter->environment->evaluateFunctions();

    if (options.count("avail")) {
      firestarter->environment->printFunctionSummary();
      return EXIT_SUCCESS;
    }

    firestarter->environment->printEnvironmentSummary();

    unsigned functionId = options["function"].as<unsigned>();

    if (EXIT_SUCCESS !=
        (returnCode = firestarter->environment->selectFunction(functionId))) {
      delete firestarter;
      return returnCode;
    }

    firestarter->environment->printThreadSummary();

    // setup thread with either high or low load configured at the start
    // low loads has to know the length of the period
    if (EXIT_SUCCESS != (returnCode = firestarter->initThreads(
                             (loadPercent == 0), period.count()))) {
      delete firestarter;
      return returnCode;
    }

    firestarter->signalWork();

    // worker thread for load control
    firestarter->watchdogWorker(period, load, timeout);

    // wait for watchdog to timeout or until user terminates
    firestarter->joinThreads();

    firestarter->printPerformanceReport();

  } catch (std::exception &e) {
    firestarter::log::error() << "Error: " << e.what() << "\n";
    return print_help(parser);
  }

  return EXIT_SUCCESS;
}
