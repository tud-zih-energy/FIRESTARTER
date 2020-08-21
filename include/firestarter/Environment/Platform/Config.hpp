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
