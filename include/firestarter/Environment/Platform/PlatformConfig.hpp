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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_PLATFORMCONFIG_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_PLATFORMCONFIG_H

#include <firestarter/Environment/Payload/Payload.hpp>
#include <firestarter/Logging/Log.hpp>

#include <algorithm>
#include <initializer_list>
#include <map>
#include <sstream>
#include <string>

namespace firestarter::environment::platform {

class PlatformConfig {
private:
  std::string _name;
  std::list<unsigned> _threads;
  payload::Payload *_payload;

protected:
  unsigned _instructionCacheSize;
  std::list<unsigned> _dataCacheBufferSize;
  unsigned _ramBufferSize;

public:
  PlatformConfig(std::string name, std::list<unsigned> threads,
                 unsigned instructionCacheSize,
                 std::initializer_list<unsigned> dataCacheBufferSize,
                 unsigned ramBufferSize, payload::Payload *payload)
      : _name(name), _threads(threads), _payload(payload),
        _instructionCacheSize(instructionCacheSize),
        _dataCacheBufferSize(dataCacheBufferSize),
        _ramBufferSize(ramBufferSize){};
  ~PlatformConfig(){};

  const std::string &name = _name;
  const unsigned &instructionCacheSize = _instructionCacheSize;
  const std::list<unsigned> &dataCacheBufferSize = _dataCacheBufferSize;
  const unsigned &ramBufferSize = _ramBufferSize;
  payload::Payload *const &payload = _payload;

  std::map<unsigned, std::string> getThreadMap(void) {
    std::map<unsigned, std::string> threadMap;

    for (auto const &thread : _threads) {
      std::stringstream functionName;
      functionName << "FUNC_" << name << "_" << payload->name << "_" << thread
                   << "T";
      threadMap[thread] = functionName.str();
    }

    return threadMap;
  }

  bool isAvailable(void) { return payload->isAvailable(); }

  virtual bool isDefault(void) = 0;

  virtual std::vector<std::pair<std::string, unsigned>>
  getDefaultPayloadSettings(void) = 0;

  std::string getDefaultPayloadSettingsString(void) {
    std::stringstream ss;

    for (auto const &[name, value] : this->getDefaultPayloadSettings()) {
      ss << name << ":" << value << ",";
    }

    auto str = ss.str();
    if (str.size() > 0) {
      str.pop_back();
    }

    return str;
  }
};

} // namespace firestarter::environment::platform

#endif
