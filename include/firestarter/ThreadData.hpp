#ifndef INCLUDE_FIRESTARTER_THREADDATA_HPP
#define INCLUDE_FIRESTARTER_THREADDATA_HPP

#define THREAD_WAIT 1
#define THREAD_WORK 2
#define THREAD_INIT 3
#define THREAD_STOP 4
#define THREAD_INIT_FAILURE 0xffffffff

#include <mutex>

namespace firestarter {

class ThreadData {
public:
  ThreadData(int id) : _id(id){};
  ~ThreadData(){};

  int getId(void) { return _id; }

  int comm = THREAD_WAIT;
  bool ack = false;
  std::mutex mutex;

private:
  int _id;
};

} // namespace firestarter

#endif
