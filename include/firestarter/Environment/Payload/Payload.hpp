#ifndef INCLUDE_FIRESTARTER_ENVIRONMENT_PAYLOAD_PAYLOAD_H
#define INCLUDE_FIRESTARTER_ENVIRONMENT_PAYLOAD_PAYLOAD_H

#include <initializer_list>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace firestarter::environment::payload {

class Payload {
private:
  std::string _name;
  unsigned getSequenceStartCount(const std::vector<std::string> sequence,
                                 const std::string start);

protected:
  std::vector<std::string>
  generateSequence(const std::map<std::string, unsigned> proportion);
  unsigned getL2SequenceCount(const std::vector<std::string> sequence) {
    return getSequenceStartCount(sequence, "L2");
  };
  unsigned getL3SequenceCount(const std::vector<std::string> sequence) {
    return getSequenceStartCount(sequence, "L3");
  };
  unsigned getRAMSequenceCount(const std::vector<std::string> sequence) {
    return getSequenceStartCount(sequence, "RAM");
  };

  unsigned
  getNumberOfSequenceRepetitions(const std::vector<std::string> sequence,
                                 const unsigned numberOfLines) {
    return numberOfLines / sequence.size();
  };

  unsigned getL2LoopCount(const std::vector<std::string> sequence,
                          const unsigned numberOfLines, const unsigned size);
  unsigned getL3LoopCount(const std::vector<std::string> sequence,
                          const unsigned numberOfLines, const unsigned size);
  unsigned getRAMLoopCount(const std::vector<std::string> sequence,
                           const unsigned numberOfLines, const unsigned size);

public:
  Payload(std::string name) : _name(name){};
  ~Payload(){};

  const std::string &name = _name;

  virtual bool isAvailable(void) = 0;

  virtual void lowLoadFunction(volatile unsigned long long *addrHigh,
                               unsigned long long period) = 0;

  virtual int compilePayload(std::map<std::string, unsigned> proportion,
                             std::list<unsigned> dataCacheBufferSize,
                             unsigned ramBufferSize, unsigned thread,
                             unsigned numberOfLines) = 0;
  virtual std::list<std::string> getAvailableInstructions(void) = 0;
  virtual void init(unsigned long long *memoryAddr,
                    unsigned long long bufferSize) = 0;
  virtual unsigned long long
  highLoadFunction(unsigned long long *addrMem,
                   volatile unsigned long long *addrHigh,
                   unsigned long long iterations) = 0;

  virtual Payload *clone(void) = 0;
};

} // namespace firestarter::environment::payload

#endif
