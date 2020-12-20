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
#include <firestarter/Measurement/Summary.hpp>
#include <firestarter/Optimizer/Individual.hpp>

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

  inline static void save(std::string const &path) {
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
      outpath += "/" + hostname + "_" + getTime() + ".json";
    }

    firestarter::log::info() << "Dumping output json in " << outpath;

    std::ofstream fp(outpath);

    if (fp.bad()) {
      firestarter::log::error() << "Could not open " << outpath;
      return;
    }

    fp << s;

    fp.close();
  }

private:
  inline static std::string getTime() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%F_%T%z");
    return ss.str();
  }

  inline static std::vector<Individual> _x = {};
  inline static std::vector<
      std::map<std::string, firestarter::measurement::Summary>>
      _f = {};
};
} // namespace firestarter::optimizer
