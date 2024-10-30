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

#include "../Json/Summary.hpp" // IWYU pragma: keep
#include "../Logging/Log.hpp"
#include "../Measurement/Summary.hpp"
#include "Individual.hpp"
#include "firestarter/WindowsCompat.hpp" // IWYU pragma: keep
#include <algorithm>
#include <cassert>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <vector>

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

  static constexpr const int MaxElementPrintCount = 20;
  static constexpr const std::size_t MinColumnWidth = 10;

  // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
  inline static std::vector<Individual> X = {};
  inline static std::vector<std::map<std::string, firestarter::measurement::Summary>> F = {};
  // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

public:
  static void append(std::vector<unsigned> const& Ind,
                     std::map<std::string, firestarter::measurement::Summary> const& Metric) {
    X.push_back(Ind);
    F.push_back(Metric);
  }

  static auto find(std::vector<unsigned> const& Individual)
      -> std::optional<std::map<std::string, firestarter::measurement::Summary>> {
    auto FindEqual = [&Individual](auto const& Ind) { return Ind == Individual; };
    auto Ind = std::find_if(X.begin(), X.end(), FindEqual);
    if (Ind == X.end()) {
      return {};
    }
    auto Dist = std::distance(X.begin(), Ind);
    return F[Dist];
  }

  static void printBest(std::vector<std::string> const& OptimizationMetrics,
                        std::vector<std::string> const& PayloadItems) {
    // TODO(Issue #76): print paretto front

    // print the best 20 individuals for each metric in a format
    // where the user can give it to --run-instruction-groups directly
    std::map<std::string, std::size_t> ColumnWidth;

    for (auto const& Metric : OptimizationMetrics) {
      ColumnWidth[Metric] = (std::max)(Metric.size(), MinColumnWidth);
      firestarter::log::trace() << Metric << ": " << ColumnWidth[Metric];
    }

    for (auto const& Metric : OptimizationMetrics) {
      using SummaryMap = std::map<std::string, firestarter::measurement::Summary>;
      auto CompareIndividual = [&Metric](SummaryMap const& MapA, SummaryMap const& MapB) {
        auto SummaryA = MapA.find(Metric);
        auto SummaryB = MapB.find(Metric);

        if (SummaryA == MapA.end() || SummaryB == MapB.end()) {
          SummaryA = MapA.find(Metric.substr(1));
          SummaryB = MapB.find(Metric.substr(1));
          assert(SummaryA != MapA.end());
          assert(SummaryB != MapB.end());
          return SummaryA->second.Average < SummaryB->second.Average;
        }

        assert(SummaryA != MapA.end());
        assert(SummaryB != MapB.end());
        return SummaryA->second.Average > SummaryB->second.Average;
      };

      auto Perm = sortPermutation(F, CompareIndividual);

      auto FormatIndividual = [&PayloadItems](std::vector<unsigned> const& Individual) {
        std::string Result;
        assert(PayloadItems.size() == Individual.size());

        for (std::size_t I = 0; I < Individual.size(); ++I) {
          // skip zero values
          if (Individual[I] == 0) {
            continue;
          }

          if (!Result.empty()) {
            Result += ",";
          }
          Result += PayloadItems[I] + ":" + std::to_string(Individual[I]);
        }

        return Result;
      };

      auto Begin = Perm.begin();
      auto End = Perm.end();

      // stop printing at a max of MaxElementPrintCount
      if (std::distance(Begin, End) > MaxElementPrintCount) {
        End = Perm.begin();
        std::advance(End, MaxElementPrintCount);
      }

      // print each of the best elements
      std::size_t Max = 0;
      for (auto It = Begin; It != End; ++It) {
        Max = (std::max)(Max, FormatIndividual(X[*It]).size());
      }

      std::stringstream FirstLine;
      std::stringstream SecondLine;
      std::string const Ind = "INDIVIDUAL";

      FirstLine << "  " << Ind;
      padding(FirstLine, Max, Ind.size(), ' ');

      SecondLine << "  ";
      padding(SecondLine, (std::max)(Max, Ind.size()), 0, '-');

      for (auto const& Metric : OptimizationMetrics) {
        auto Width = ColumnWidth[Metric];

        FirstLine << " | ";
        SecondLine << "---";

        FirstLine << Metric;
        padding(FirstLine, Width, Metric.size(), ' ');
        padding(SecondLine, Width, 0, '-');
      }

      std::stringstream Ss;

      Ss << "\n Best individuals sorted by metric " << Metric << " "
         << ((Metric[0] == '-') ? "ascending" : "descending") << ":\n"
         << FirstLine.str() << "\n"
         << SecondLine.str() << "\n";

      // print INDIVIDUAL | metric 1 | metric 2 | ... | metric N
      for (auto It = Begin; It != End; ++It) {
        auto const& Fitness = F[*It];
        auto const Ind = FormatIndividual(X[*It]);

        Ss << "  " << Ind;
        padding(Ss, Max, Ind.size(), ' ');

        for (auto const& Metric : OptimizationMetrics) {
          auto Width = ColumnWidth[Metric];
          std::string Value;

          auto FitnessOfMetric = Fitness.find(Metric);
          auto InvertedMetric = Metric.substr(1);
          auto FitnessOfInvertedMetric = Fitness.find(InvertedMetric);

          if (FitnessOfMetric != Fitness.end()) {
            Value = std::to_string(FitnessOfMetric->second.Average);
          } else if (FitnessOfInvertedMetric != Fitness.end()) {
            Value = std::to_string(FitnessOfInvertedMetric->second.Average);
          } else {
            assert(false);
          }

          Ss << " | " << Value;
          padding(Ss, Width, Value.size(), ' ');
        }
        Ss << "\n";
      }

      Ss << "\n";

      firestarter::log::info() << Ss.str();
    }

    firestarter::log::info() << "To run FIRESTARTER with the best individual of a given metric "
                                "use the command line argument "
                                "`--run-instruction-groups=INDIVIDUAL`";
  }

  /// Save the history to a file. This function is not threadsafe as is calls History::getTime.
  /// \arg Path The folder in which the outfile shall be created. If it is empty the current directory name or /tmp will
  /// be choosen.
  /// \arg StartTime The start time as a string which is saved in the json datastructure.
  /// \arg PayloadItems The Vector of meta instructions which map to the vector of individuals.
  /// \arg Argc The Argc of the executed programm.
  /// \arg Argv The Argv of the executed programm.
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

    // Initialize a string with length of 256 filled with null characters
    auto Hostname = std::string(256, 0);
    // get the hostname
    if (0 != gethostname(Hostname.data(), Hostname.size())) {
      Hostname = "unknown";
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
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      J["args"].push_back(Argv[I]);
    }

    // dump the output
    const auto S = J.dump();

    firestarter::log::trace() << S;

    std::string Outpath = Path;
    if (Outpath.empty()) {
      // Wrapp get_current_dir_name in a unique ptr, as it needs to get deleted by free when it is not used anymore.
      const std::unique_ptr<char, void (*)(void*)> WrappedPwd = {get_current_dir_name(), free};
      if (WrappedPwd) {
        Outpath = *WrappedPwd;
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

  /// Get the current time in the local timezone as a string formatted by "%F_%T%z". This function is NOT threadsafe.
  /// \returns The current time in local timezone as a formatted string.
  static auto getTime() -> std::string {
    const auto T = std::time(nullptr);
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const auto* Tm = std::localtime(&T);
    std::stringstream Ss;
    Ss << std::put_time(Tm, "%F_%T%z");
    return Ss.str();
  }
};
} // namespace firestarter::optimizer
