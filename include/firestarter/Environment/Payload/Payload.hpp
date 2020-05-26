#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_PAYLOAD_PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_PAYLOAD_PAYLOAD_H

#include <initializer_list>
#include <list>
#include <map>
#include <string>

namespace firestarter::environment::payload {

class Payload {
private:
  std::string _name;

public:
  Payload(std::string name) : _name(name){};
  ~Payload(){};

  const std::string &name = _name;

  virtual bool isAvailable(void) = 0;

  virtual void lowLoadFunction(...) = 0;

  virtual int compilePayload(std::map<std::string, unsigned> proportion) = 0;
  virtual std::list<std::string> getAvailableInstructions(void) = 0;
  virtual void init(unsigned long long *memoryAddr,
                    unsigned long long bufferSize) = 0;
  virtual void highLoadFunction(...) = 0;

  virtual Payload *clone(void) = 0;
};

} // namespace firestarter::environment::payload

#endif
