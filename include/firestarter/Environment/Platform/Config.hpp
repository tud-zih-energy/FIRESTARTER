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

#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_CONFIG_HPP
#define INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_CONFIG_HPP

#include <firestarter/Environment/Platform/PlatformConfig.hpp>

namespace firestarter::environment::platform {

class Config {
public:
  Config(PlatformConfig *platformConfig, unsigned thread)
      : _platformConfig(platformConfig), _payload(nullptr), _thread(thread),
        _payloadSettings(platformConfig->getDefaultPayloadSettings()){};
  Config(const Config &c)
      : _platformConfig(c.platformConfig),
        _payload(c.platformConfig->payload->clone()), _thread(c.thread),
        _payloadSettings(c.payloadSettings){};
  ~Config(void);

  PlatformConfig *const &platformConfig = _platformConfig;
  payload::Payload *const &payload = _payload;
  const unsigned &thread = _thread;
  const std::vector<std::pair<std::string, unsigned>> &payloadSettings =
      _payloadSettings;

  void setPayloadSettings(
      std::vector<std::pair<std::string, unsigned>> payloadSettings) {
    this->_payloadSettings = payloadSettings;
  }

private:
  PlatformConfig *_platformConfig;
  payload::Payload *_payload;
  unsigned _thread;
  std::vector<std::pair<std::string, unsigned>> _payloadSettings;
};

} // namespace firestarter::environment::platform

#endif
