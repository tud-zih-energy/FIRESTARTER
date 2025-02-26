/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/Config/Config.hpp"
#include "firestarter/Config/CpuBind.hpp"
#include "firestarter/Config/InstructionGroups.hpp"
#include "firestarter/Config/MetricName.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/SafeExit.hpp"

#include <algorithm>
#include <cstdlib>
#include <cxxopts.hpp>
#include <exception>
#include <iterator>
#include <nitro/log/severity.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void printCopyright() {
  firestarter::log::info() << "This program is free software: you can redistribute it and/or "
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

void printWarranty() {
  firestarter::log::info() << "This program is distributed in the hope that it will be useful,\n"
                           << "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
                           << "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
                           << "GNU General Public License for more details.\n"
                           << "\n"
                           << "You should have received a copy of the GNU General Public License\n"
                           << "along with this program.  If not, see "
                              "<http://www.gnu.org/licenses/>.\n";
}

void printHelp(cxxopts::Options const& Parser, std::string const& Section = "") {
  std::vector<std::pair<std::string, std::string>> Options = {{"information", "Information Options:\n"},
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

  // Select the specific option if sections is no empty
  if (!Section.empty()) {
    // section not found
    auto FindSection = [&Section](std::pair<std::string, std::string> const& Pair) { return Pair.first == Section; };
    auto SectionsIt = std::find_if(Options.begin(), Options.end(), FindSection);
    if (SectionsIt == Options.end()) {
      throw std::invalid_argument("Section \"" + Section + "\" not found in help.");
    }
    Options = {*SectionsIt};
  }

  // clang-format off
  firestarter::log::info()
    << Parser.help(Options)
    << "Examples:\n"
    << "  ./FIRESTARTER                 starts FIRESTARTER without timeout\n"
    << "  ./FIRESTARTER -t 300          starts a 5 minute run of FIRESTARTER\n"
    << "  ./FIRESTARTER -l 50 -t 600    starts a 10 minute run of FIRESTARTER with\n"
    << "                                50\% high load and 50\% idle time\n"
    << (firestarter::OptionalFeatures.gpuEnabled() ? 
       "                                on CPUs and full load on GPUs\n"
     : "")
    << "  ./FIRESTARTER -l 75 -p 20000000\n"
    << "                                starts FIRESTARTER with an interval length\n"
    << "                                of 2 sec, 1.5s high load"
    << (firestarter::OptionalFeatures.gpuEnabled() ? 
       " on CPUs and full load on GPUs\n"
     : "\n")
    << (firestarter::OptionalFeatures.OptimizationEnabled ?
       "  ./FIRESTARTER --measurement --start-delta=300000 -t 900\n"
       "                                starts FIRESTARTER measuring all available\n"
       "                                metrics for 15 minutes disregarding the first\n"
       "                                5 minutes and last two seconds (default to `--stop-delta`)\n"
       "  ./FIRESTARTER -t 20 --optimize=NSGA2 --optimization-metric sysfs-powercap-rapl,perf-ipc\n"
       "                                starts FIRESTARTER optimizing with the sysfs-powercap-rapl\n"
       "                                and perf-ipc metric. The duration is 20s long. The default\n"
       "                                instruction groups for the current platform will be used.\n"
     : "")
    ;
  // clang-format on
}

} // namespace

namespace firestarter {

Config::Config(int Argc, const char** Argv)
    : Argv(Argv)
    , Argc(Argc) {
  const auto* ExecutableName = *Argv;

  cxxopts::Options Parser(ExecutableName);

  const auto HelpDescription =
      std::string("Display usage information. SECTION can be any of: information | general | specialized-workloads") +
      (firestarter::OptionalFeatures.DebugFeatureEnabled ? " | debug" : "") +
      (firestarter::OptionalFeatures.OptimizationEnabled ? "\n| measurement | optimization" : "");

  const auto LoadDescription =
      std::string("Set the percentage of high CPU load to LOAD\n(%) default: 100, valid values: 0 <= LOAD <=\n100, "
                  "threads will be idle in the remaining time,\nfrequency of load changes is determined by -p.") +
      (firestarter::OptionalFeatures.gpuEnabled() ? " This option does NOT influence the GPU\nworkload!" : "");

  // clang-format off
  Parser.add_options("information")
    ("h,help", HelpDescription,
      cxxopts::value<std::string>()->implicit_value(""), "SECTION")
    ("v,version", "Display version information")
    ("c,copyright", "Display copyright information")
    ("w,warranty", "Display warranty information")
    ("q,quiet", "Set log level to Warning")
    ("r,report", "Display additional information (overridden by -q)")
    ("debug", "Print debug output")
    ("a,avail", "List available functions");

  Parser.add_options("general")
    ("i,function", "Specify integer ID of the load-function to be\nused (as listed by --avail)",
      cxxopts::value<unsigned>(), "ID");

  if (firestarter::OptionalFeatures.gpuEnabled()) {
    Parser.add_options("general")
      ("f,usegpufloat", "Use single precision matrix multiplications\ninstead of default")
      ("d,usegpudouble", "Use double precision matrix multiplications\ninstead of default")
      ("g,gpus", "Number of gpus to use, default: -1 (all)",
        cxxopts::value<int>()->default_value("-1"))
      ("m,matrixsize", "Size of the matrix to calculate, default: 0 (maximum)",
        cxxopts::value<unsigned>()->default_value("0"));
  }

  Parser.add_options("general")
    ("t,timeout", "Set the timeout (seconds) after which FIRESTARTER\nterminates itself, default: 0 (no timeout)",
      cxxopts::value<unsigned>()->default_value("0"), "TIMEOUT")
    ("l,load", LoadDescription,
      cxxopts::value<unsigned>()->default_value("100"), "LOAD")
    ("p,period", "Set the interval length for CPUs to PERIOD\n(usec), default: 100000, each interval contains\na high load and an idle phase, the percentage\nof high load is defined by -l.",
      cxxopts::value<unsigned>()->default_value("100000"), "PERIOD")
    ("n,threads", "Specify the number of threads. Cannot be\ncombined with -b | --bind, which impicitly\nspecifies the number of threads.",
      cxxopts::value<unsigned>(), "COUNT")
    ("b,bind", "Select certain CPUs. CPULIST format: \"x,y,z\",\n\"x-y\", \"x-y/step\", and any combination of the\nabove. Cannot be combined with -n | --threads.",
      cxxopts::value<std::string>(), "CPULIST")
    ("error-detection", "Enable error detection. This aborts execution when the calculated data is corruped by errors. FIRESTARTER must run with 2 or more threads for this feature. Cannot be used with -l | --load and --optimize.");

  Parser.add_options("specialized-workloads")
    ("list-instruction-groups", "List the available instruction groups for the\npayload of the current platform.")
    ("run-instruction-groups", "Run the payload with the specified\ninstruction groups. GROUPS format: multiple INST:VAL\npairs comma-seperated.",
      cxxopts::value<std::string>(), "GROUPS")
    ("set-line-count", "Set the number of lines for a payload.",
      cxxopts::value<unsigned>());

  if (firestarter::OptionalFeatures.DebugFeatureEnabled) {
    Parser.add_options("debug")
      ("allow-unavailable-payload", "")
      ("dump-registers", "Dump the working registers on the first\nthread. Depending on the payload these are mm, xmm,\nymm or zmm. Only use it without a timeout and\n100 percent load. DELAY between dumps in secs. Cannot be used with --error-detection.",
        cxxopts::value<unsigned>()->implicit_value("10"), "DELAY")
      ("dump-registers-outpath", "Path for the dump of the output files. If\nPATH is not given, current working directory will\nbe used.",
        cxxopts::value<std::string>()->default_value(""), "PATH");
  }

  if (firestarter::OptionalFeatures.OptimizationEnabled) {
    Parser.add_options("measurement")
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
  
    Parser.add_options("optimization")
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
  }
  // clang-format on

  try {
    auto Options = Parser.parse(Argc, Argv);

    if (static_cast<bool>(Options.count("quiet"))) {
      firestarter::logging::Filter<firestarter::logging::record>::set_severity(nitro::log::severity_level::warn);
    } else if (static_cast<bool>(Options.count("report"))) {
      firestarter::logging::Filter<firestarter::logging::record>::set_severity(nitro::log::severity_level::debug);
    } else if (static_cast<bool>(Options.count("debug"))) {
      firestarter::logging::Filter<firestarter::logging::record>::set_severity(nitro::log::severity_level::trace);
    } else {
      firestarter::logging::Filter<firestarter::logging::record>::set_severity(nitro::log::severity_level::info);
    }

    if (static_cast<bool>(Options.count("version"))) {
      safeExit(EXIT_SUCCESS);
    }

    if (static_cast<bool>(Options.count("copyright"))) {
      printCopyright();
      safeExit(EXIT_SUCCESS);
    }

    if (static_cast<bool>(Options.count("warranty"))) {
      printWarranty();
      safeExit(EXIT_SUCCESS);
    }

    firestarter::log::info() << "This program comes with ABSOLUTELY NO WARRANTY; for details run `" << ExecutableName
                             << " -w`.\n"
                             << "This is free software, and you are welcome to redistribute it\n"
                             << "under certain conditions; run `" << ExecutableName << " -c` for details.\n";

    if (static_cast<bool>(Options.count("help"))) {
      auto Section = Options["help"].as<std::string>();

      printHelp(Parser, Section);
      safeExit(EXIT_SUCCESS);
    }

    Timeout = std::chrono::seconds(Options["timeout"].as<unsigned>());
    const auto LoadPercent = Options["load"].as<unsigned>();
    Period = std::chrono::microseconds(Options["period"].as<unsigned>());

    if (LoadPercent > 100) {
      throw std::invalid_argument("Option -l/--load may not be above 100.");
    }

    Load = (Period * LoadPercent) / 100;
    if (LoadPercent == 100 || Load == std::chrono::microseconds::zero()) {
      Period = std::chrono::microseconds::zero();
    }

    ErrorDetection = static_cast<bool>(Options.count("error-detection"));
    if (ErrorDetection && LoadPercent != 100) {
      throw std::invalid_argument("Option --error-detection may only be used "
                                  "with -l/--load equal 100.");
    }

    if (firestarter::OptionalFeatures.DebugFeatureEnabled) {
      AllowUnavailablePayload = static_cast<bool>(Options.count("allow-unavailable-payload"));
      DumpRegisters = static_cast<bool>(Options.count("dump-registers"));
      if (DumpRegisters) {
        DumpRegistersTimeDelta = std::chrono::seconds(Options["dump-registers"].as<unsigned>());
        if (Timeout != std::chrono::microseconds::zero() && LoadPercent != 100) {
          throw std::invalid_argument("Option --dump-registers may only be used "
                                      "without a timeout and full load.");
        }
        if (ErrorDetection) {
          throw std::invalid_argument("Options --dump-registers and --error-detection cannot be used "
                                      "together.");
        }
      }
    }

    if (static_cast<bool>(Options.count("threads"))) {
      RequestedNumThreads = Options["threads"].as<unsigned>();
    }

    if (static_cast<bool>(Options.count("bind"))) {
      CpuBinding = CpuBind::fromString(Options["bind"].as<std::string>());
    }

    if (RequestedNumThreads && CpuBinding) {
      throw std::invalid_argument("Options -b/--bind and -n/--threads cannot be used together.");
    }

    if (firestarter::OptionalFeatures.gpuEnabled()) {
      GpuUseFloat = static_cast<bool>(Options.count("usegpufloat"));
      GpuUseDouble = static_cast<bool>(Options.count("usegpudouble"));

      if (GpuUseFloat && GpuUseDouble) {
        throw std::invalid_argument("Options -f/--usegpufloat and "
                                    "-d/--usegpudouble cannot be used together.");
      }

      GpuMatrixSize = Options["matrixsize"].as<unsigned>();
      if (GpuMatrixSize > 0 && GpuMatrixSize < 64) {
        throw std::invalid_argument("Option -m/--matrixsize may not be below 64.");
      }

      Gpus = Options["gpus"].as<int>();
    }

    PrintFunctionSummary = static_cast<bool>(Options.count("avail"));

    if (static_cast<bool>(Options.count("function"))) {
      FunctionId = Options["function"].as<unsigned>();
    }

    ListInstructionGroups = static_cast<bool>(Options.count("list-instruction-groups"));
    if (static_cast<bool>(Options.count("run-instruction-groups"))) {
      Groups = InstructionGroups::fromString(Options["run-instruction-groups"].as<std::string>());
    }
    if (static_cast<bool>(Options.count("set-line-count"))) {
      LineCount = Options["set-line-count"].as<unsigned>();
    }

    if (firestarter::OptionalFeatures.OptimizationEnabled) {
      StartDelta = std::chrono::milliseconds(Options["start-delta"].as<unsigned>());
      StopDelta = std::chrono::milliseconds(Options["stop-delta"].as<unsigned>());
      MeasurementInterval = std::chrono::milliseconds(Options["measurement-interval"].as<unsigned>());
#ifndef FIRESTARTER_LINK_STATIC
      MetricPaths = Options["metric-path"].as<std::vector<std::string>>();
#endif
      if (static_cast<bool>(Options.count("metric-from-stdin"))) {
        StdinMetrics = Options["metric-from-stdin"].as<std::vector<std::string>>();
      }
      Measurement = static_cast<bool>(Options.count("measurement"));
      ListMetrics = static_cast<bool>(Options.count("list-metrics"));
      Optimize = static_cast<bool>(Options.count("optimize"));

      if (Optimize) {
        if (ErrorDetection) {
          throw std::invalid_argument("Options --error-detection and --optimize "
                                      "cannot be used together.");
        }
        if (Measurement) {
          throw std::invalid_argument("Options --measurement and --optimize cannot be used together.");
        }
        Preheat = std::chrono::seconds(Options["preheat"].as<unsigned>());
        OptimizationAlgorithm = Options["optimize"].as<std::string>();
        if (static_cast<bool>(Options.count("optimization-metric"))) {
          const auto Metrics = Options["optimization-metric"].as<std::vector<std::string>>();
          std::transform(Metrics.cbegin(), Metrics.cend(), std::back_inserter(OptimizationMetrics),
                         [](const std::string& Metric) { return MetricName::fromString(Metric); });
        }
        if (LoadPercent != 100) {
          throw std::invalid_argument("Options -p | --period and -l | --load are "
                                      "not compatible with --optimize.");
        }
        if (Timeout == std::chrono::seconds::zero()) {
          throw std::invalid_argument("Option -t | --timeout must be specified for optimization.");
        }
        EvaluationDuration = Timeout;
        // this will deactivate the watchdog worker
        Timeout = std::chrono::seconds::zero();
        Individuals = Options["individuals"].as<unsigned>();
        if (static_cast<bool>(Options.count("optimize-outfile"))) {
          OptimizeOutfile = Options["optimize-outfile"].as<std::string>();
        }
        Generations = Options["generations"].as<unsigned>();
        Nsga2Cr = Options["nsga2-cr"].as<double>();
        Nsga2M = Options["nsga2-m"].as<double>();

        if (OptimizationAlgorithm != "NSGA2") {
          throw std::invalid_argument("Option --optimize must be any of: NSGA2");
        }
      }
    }
  } catch (std::exception& E) {
    printHelp(Parser);
    firestarter::log::error() << E.what() << "\n";
  }
}
} // namespace firestarter