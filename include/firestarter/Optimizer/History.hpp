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

#pragma once

#include <firestarter/Json/Summary.hpp>
#include <firestarter/Logging/Log.hpp>
#include <firestarter/Measurement/Summary.hpp>
#include <firestarter/Optimizer/Individual.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <tuple>
#include <vector>

extern "C" {
#include <unistd.h>
}

namespace firestarter::optimizer {

struct History {
private:
  // https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of/17074810#17074810
  template <typename T, typename Compare>
  inline static std::vector<std::size_t>
  sortPermutation(const std::vector<T> &vec, Compare &compare) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) {
      return compare(vec[i], vec[j]);
    });
    return p;
  }

  inline static void padding(std::stringstream &ss, std::size_t width,
                             std::size_t taken, char c) {
    for (std::size_t i = 0; i < (std::max)(width, taken) - taken; ++i) {
      ss << c;
    }
  }

  inline static int MAX_ELEMENT_PRINT_COUNT = 20;
  inline static std::size_t MIN_COLUMN_WIDTH = 10;

  inline static std::vector<Individual> _x = {};
  inline static std::vector<
      std::map<std::string, firestarter::measurement::Summary>>
      _f = {};

public:
  inline static void append(
      std::vector<unsigned> const &ind,
      std::map<std::string, firestarter::measurement::Summary> const &metric) {
    _x.push_back(ind);
    _f.push_back(metric);
  }

  inline static std::optional<
      std::map<std::string, firestarter::measurement::Summary>>
  find(std::vector<unsigned> const &individual) {
    auto findEqual = [individual](auto const &ind) {
      return ind == individual;
    };
    auto ind = std::find_if(_x.begin(), _x.end(), findEqual);
    if (ind == _x.end()) {
      return {};
    }
    auto dist = std::distance(_x.begin(), ind);
    return _f[dist];
  }

  inline static void
  printBest(std::vector<std::string> const &optimizationMetrics,
            std::vector<std::string> const &payloadItems) {
    // TODO: print paretto front

    // print the best 20 individuals for each metric in a format
    // where the user can give it to --run-instruction-groups directly
    std::map<std::string, std::size_t> columnWidth;

    for (auto const &metric : optimizationMetrics) {
      columnWidth[metric] = (std::max)(metric.size(), MIN_COLUMN_WIDTH);
      firestarter::log::trace() << metric << ": " << columnWidth[metric];
    }

    for (auto const &metric : optimizationMetrics) {
      using SummaryMap =
          std::map<std::string, firestarter::measurement::Summary>;
      auto compareIndividual = [&metric](SummaryMap const &mapA,
                                         SummaryMap const &mapB) {
        auto summaryA = mapA.find(metric);
        auto summaryB = mapB.find(metric);

        if (summaryA == mapA.end() || summaryB == mapB.end()) {
          summaryA = mapA.find(metric.substr(1));
          summaryB = mapB.find(metric.substr(1));
          assert(summaryA != mapA.end());
          assert(summaryB != mapB.end());
          return summaryA->second.average < summaryB->second.average;
        }

        assert(summaryA != mapA.end());
        assert(summaryB != mapB.end());
        return summaryA->second.average > summaryB->second.average;
      };

      auto perm = sortPermutation(_f, compareIndividual);

      auto formatIndividual =
          [&payloadItems](std::vector<unsigned> const &individual) {
            std::string result = "";
            assert(payloadItems.size() == individual.size());

            for (std::size_t i = 0; i < individual.size(); ++i) {
              // skip zero values
              if (individual[i] == 0) {
                continue;
              }

              if (result.size() != 0) {
                result += ",";
              }
              result += payloadItems[i] + ":" + std::to_string(individual[i]);
            }

            return result;
          };

      auto begin = perm.begin();
      auto end = perm.end();

      // stop printing at a max of MAX_ELEMENT_PRINT_COUNT
      if (std::distance(begin, end) > MAX_ELEMENT_PRINT_COUNT) {
        end = perm.begin();
        std::advance(end, MAX_ELEMENT_PRINT_COUNT);
      }

      // print each of the best elements
      std::size_t max = 0;
      for (auto it = begin; it != end; ++it) {
        max = (std::max)(max, formatIndividual(_x[*it]).size());
      }

      std::stringstream firstLine;
      std::stringstream secondLine;
      std::string ind = "INDIVIDUAL";

      firstLine << "  " << ind;
      padding(firstLine, max, ind.size(), ' ');

      secondLine << "  ";
      padding(secondLine, (std::max)(max, ind.size()), 0, '-');

      for (auto const &metric : optimizationMetrics) {
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
        auto const fitness = _f[*it];
        auto const ind = formatIndividual(_x[*it]);

        ss << "  " << ind;
        padding(ss, max, ind.size(), ' ');

        for (auto const &metric : optimizationMetrics) {
          auto width = columnWidth[metric];
          std::string value;

          auto fitnessOfMetric = fitness.find(metric);
          auto invertedMetric = metric.substr(1);
          auto fitnessOfInvertedMetric = fitness.find(invertedMetric);

          if (fitnessOfMetric != fitness.end()) {
            value = std::to_string(fitnessOfMetric->second.average);
          } else if (fitnessOfInvertedMetric != fitness.end()) {
            value = std::to_string(fitnessOfInvertedMetric->second.average);
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

    firestarter::log::info()
        << "To run FIRESTARTER with the best individual of a given metric "
           "use the command line argument "
           "`--run-instruction-groups=INDIVIDUAL`";
  }

  inline static void save(std::string const &path, std::string const &startTime,
                          std::vector<std::string> const &payloadItems,
                          const int argc, const char **argv) {
    using json = nlohmann::json;

    json j = json::object();

    j["individuals"] = json::array();
    for (auto const &ind : _x) {
      j["individuals"].push_back(ind);
    }

    j["metrics"] = json::array();
    for (auto const &eval : _f) {
      j["metrics"].push_back(eval);
    }

    // get the hostname
    char cHostname[256];
    std::string hostname;
    if (0 != gethostname(cHostname, sizeof(cHostname))) {
      hostname = "unknown";
    } else {
      hostname = cHostname;
    }

    j["hostname"] = hostname;

    j["startTime"] = startTime;
    j["endTime"] = getTime();

    // save the payload items
    j["payloadItems"] = json::array();
    for (auto const &item : payloadItems) {
      j["payloadItems"].push_back(item);
    }

    // save the arguments
    j["args"] = json::array();
    for (int i = 0; i < argc; ++i) {
      j["args"].push_back(argv[i]);
    }

    // dump the output
    std::string s = j.dump();

    firestarter::log::trace() << s;

    std::string outpath = path;
    if (outpath.empty()) {
      char *pwd = get_current_dir_name();
      if (pwd) {
        outpath = pwd;
        free(pwd);
      } else {
        firestarter::log::warn() << "Could not find $PWD.";
        outpath = "/tmp";
      }
      outpath += "/" + hostname + "_" + startTime + ".json";
    }

    firestarter::log::info() << "\nDumping output json in " << outpath;

    std::ofstream fp(outpath);

    if (fp.bad()) {
      firestarter::log::error() << "Could not open " << outpath;
      return;
    }

    fp << s;

    fp.close();
  }

  inline static std::string getTime() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%F_%T%z");
    return ss.str();
  }
};
} // namespace firestarter::optimizer
