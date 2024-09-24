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

#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ctime>
#include <firestarter/Json/Summary.hpp>
#include <firestarter/Logging/Log.hpp>
#include <firestarter/Measurement/Summary.hpp>
#include <firestarter/Optimizer/Individual.hpp>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

extern "C" {
#include <unistd.h>
}

namespace firestarter::optimizer {

struct History {
private:
  // https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of/17074810#17074810
  template <typename T, typename CompareT>
  static auto sortPermutation(const std::vector<T>& Vec, CompareT& Compare) -> std::vector<std::size_t> {
    std::vector<std::size_t> P(Vec.size());
    std::iota(P.begin(), P.end(), 0);
    std::sort(P.begin(), P.end(), [&](std::size_t I, std::size_t J) { return Compare(Vec[I], Vec[J]); });
    return P;
  }

  static void padding(std::stringstream& Ss, std::size_t Width, std::size_t Taken, char C) {
    for (std::size_t I = 0; I < (std::max)(Width, Taken) - Taken; ++I) {
      Ss << C;
    }
  }

  inline static int MaxElementPrintCount = 20;
  inline static std::size_t MinColumnWidth = 10;

  inline static std::vector<Individual> X = {};
  inline static std::vector<std::map<std::string, firestarter::measurement::Summary>> F = {};

public:
  static void append(std::vector<unsigned> const& Ind,
                     std::map<std::string, firestarter::measurement::Summary> const& Metric) {
    X.push_back(Ind);
    F.push_back(Metric);
  }

  static auto find(std::vector<unsigned> const& Individual)
      -> std::optional<std::map<std::string, firestarter::measurement::Summary>> {
    auto FindEqual = [Individual](auto const& ind) { return ind == Individual; };
    auto Ind = std::find_if(X.begin(), X.end(), FindEqual);
    if (Ind == X.end()) {
      return {};
    }
    auto Dist = std::distance(X.begin(), Ind);
    return F[Dist];
  }

  static void printBest(std::vector<std::string> const& OptimizationMetrics,
                        std::vector<std::string> const& PayloadItems) {
    // TODO: print paretto front

    // print the best 20 individuals for each metric in a format
    // where the user can give it to --run-instruction-groups directly
    std::map<std::string, std::size_t> columnWidth;

    for (auto const& metric : OptimizationMetrics) {
      columnWidth[metric] = (std::max)(metric.size(), MinColumnWidth);
      firestarter::log::trace() << metric << ": " << columnWidth[metric];
    }

    for (auto const& metric : OptimizationMetrics) {
      using SummaryMap = std::map<std::string, firestarter::measurement::Summary>;
      auto compareIndividual = [&metric](SummaryMap const& mapA, SummaryMap const& mapB) {
        auto summaryA = mapA.find(metric);
        auto summaryB = mapB.find(metric);

        if (summaryA == mapA.end() || summaryB == mapB.end()) {
          summaryA = mapA.find(metric.substr(1));
          summaryB = mapB.find(metric.substr(1));
          assert(summaryA != mapA.end());
          assert(summaryB != mapB.end());
          return summaryA->second.Average < summaryB->second.Average;
        }

        assert(summaryA != mapA.end());
        assert(summaryB != mapB.end());
        return summaryA->second.Average > summaryB->second.Average;
      };

      auto perm = sortPermutation(F, compareIndividual);

      auto formatIndividual = [&PayloadItems](std::vector<unsigned> const& individual) {
        std::string result = "";
        assert(PayloadItems.size() == individual.size());

        for (std::size_t i = 0; i < individual.size(); ++i) {
          // skip zero values
          if (individual[i] == 0) {
            continue;
          }

          if (result.size() != 0) {
            result += ",";
          }
          result += PayloadItems[i] + ":" + std::to_string(individual[i]);
        }

        return result;
      };

      auto begin = perm.begin();
      auto end = perm.end();

      // stop printing at a max of MaxElementPrintCount
      if (std::distance(begin, end) > MaxElementPrintCount) {
        end = perm.begin();
        std::advance(end, MaxElementPrintCount);
      }

      // print each of the best elements
      std::size_t max = 0;
      for (auto it = begin; it != end; ++it) {
        max = (std::max)(max, formatIndividual(X[*it]).size());
      }

      std::stringstream firstLine;
      std::stringstream secondLine;
      std::string ind = "INDIVIDUAL";

      firstLine << "  " << ind;
      padding(firstLine, max, ind.size(), ' ');

      secondLine << "  ";
      padding(secondLine, (std::max)(max, ind.size()), 0, '-');

      for (auto const& metric : OptimizationMetrics) {
        auto width = columnWidth[metric];

        firstLine << " | ";
        secondLine << "---";

        firstLine << metric;
        padding(firstLine, width, metric.size(), ' ');
        padding(secondLine, width, 0, '-');
      }

      std::stringstream ss;

      ss << "\n Best individuals sorted by metric " << metric << " "
         << ((metric[0] == '-') ? "ascending" : "descending") << ":\n"
         << firstLine.str() << "\n"
         << secondLine.str() << "\n";

      // print INDIVIDUAL | metric 1 | metric 2 | ... | metric N
      for (auto it = begin; it != end; ++it) {
        auto const fitness = F[*it];
        auto const ind = formatIndividual(X[*it]);

        ss << "  " << ind;
        padding(ss, max, ind.size(), ' ');

        for (auto const& metric : OptimizationMetrics) {
          auto width = columnWidth[metric];
          std::string value;

          auto fitnessOfMetric = fitness.find(metric);
          auto invertedMetric = metric.substr(1);
          auto fitnessOfInvertedMetric = fitness.find(invertedMetric);

          if (fitnessOfMetric != fitness.end()) {
            value = std::to_string(fitnessOfMetric->second.Average);
          } else if (fitnessOfInvertedMetric != fitness.end()) {
            value = std::to_string(fitnessOfInvertedMetric->second.Average);
          } else {
            assert(false);
          }

          ss << " | " << value;
          padding(ss, width, value.size(), ' ');
        }
        ss << "\n";
      }

      ss << "\n";

      firestarter::log::info() << ss.str();
    }

    firestarter::log::info() << "To run FIRESTARTER with the best individual of a given metric "
                                "use the command line argument "
                                "`--run-instruction-groups=INDIVIDUAL`";
  }

  static void save(std::string const& Path, std::string const& StartTime, std::vector<std::string> const& PayloadItems,
                   const int Argc, const char** Argv) {
    using json = nlohmann::json;

    json J = json::object();

    J["individuals"] = json::array();
    for (auto const& Ind : X) {
      J["individuals"].push_back(Ind);
    }

    J["metrics"] = json::array();
    for (auto const& Eval : F) {
      J["metrics"].push_back(Eval);
    }

    // get the hostname
    char CHostname[256];
    std::string Hostname;
    if (0 != gethostname(CHostname, sizeof(CHostname))) {
      Hostname = "unknown";
    } else {
      Hostname = CHostname;
    }

    J["hostname"] = Hostname;

    J["startTime"] = StartTime;
    J["endTime"] = getTime();

    // save the payload items
    J["payloadItems"] = json::array();
    for (auto const& Item : PayloadItems) {
      J["payloadItems"].push_back(Item);
    }

    // save the arguments
    J["args"] = json::array();
    for (int I = 0; I < Argc; ++I) {
      J["args"].push_back(Argv[I]);
    }

    // dump the output
    std::string S = J.dump();

    firestarter::log::trace() << S;

    std::string Outpath = Path;
    if (Outpath.empty()) {
      char* Pwd = get_current_dir_name();
      if (Pwd) {
        Outpath = Pwd;
        free(Pwd);
      } else {
        firestarter::log::warn() << "Could not find $PWD.";
        Outpath = "/tmp";
      }
      Outpath += "/" + Hostname + "_" + StartTime + ".json";
    }

    firestarter::log::info() << "\nDumping output json in " << Outpath;

    std::ofstream Fp(Outpath);

    if (Fp.bad()) {
      firestarter::log::error() << "Could not open " << Outpath;
      return;
    }

    Fp << S;

    Fp.close();
  }

  static auto getTime() -> std::string {
    auto T = std::time(nullptr);
    auto Tm = *std::localtime(&T);
    std::stringstream Ss;
    Ss << std::put_time(&Tm, "%F_%T%z");
    return Ss.str();
  }
};
} // namespace firestarter::optimizer
