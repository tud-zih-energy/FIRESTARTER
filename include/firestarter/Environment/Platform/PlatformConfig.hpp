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
  std::list<unsigned> _dataCacheBufferSize;
  unsigned _ramBufferSize;

public:
  PlatformConfig(std::string name, std::list<unsigned> threads,
                 std::initializer_list<unsigned> dataCacheBufferSize,
                 unsigned ramBufferSize, payload::Payload *payload)
      : _name(name), _threads(threads),
        _dataCacheBufferSize(dataCacheBufferSize),
        _ramBufferSize(ramBufferSize), _payload(payload){};
  ~PlatformConfig(){};

  const std::string &name = _name;
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

  void printCodePathSummary(unsigned thread) {
    log::info() << "\n"
                << "  Taking " << payload->name << " path optimized for "
                << name << " - " << thread << " thread(s) per core\n"
                << "  Used buffersizes per thread:";
    unsigned i = 1;
    for (auto const &bytes : dataCacheBufferSize) {
      log::info() << "    - L" << i << "-Cache: " << bytes / thread << " Bytes";
      i++;
    }

    log::info() << "    - Memory: " << ramBufferSize / thread << " Bytes";
  }

  bool isAvailable(void) { return payload->isAvailable(); }

  virtual bool isDefault(void) = 0;

  virtual std::map<std::string, unsigned> getDefaultPayloadSettings(void) = 0;
};

} // namespace firestarter::environment::platform

#endif