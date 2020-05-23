#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_CONFIG_HPP
#define INCLUDE_FIRESTARTER_ENVIRONMENT_PLATFORM_CONFIG_HPP

#include <firestarter/Environment/Platform/PlatformConfig.hpp>

namespace firestarter::environment::platform {

class Config {
public:
  Config(PlatformConfig *platformConfig, unsigned thread)
      : _platformConfig(platformConfig), _thread(thread){};
  ~Config(void);

  PlatformConfig *const &platformConfig = _platformConfig;
  const unsigned &thread = _thread;

private:
  PlatformConfig *_platformConfig;
  unsigned _thread;
};

} // namespace firestarter::environment::platform

#endif
