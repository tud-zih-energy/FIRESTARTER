#pragma once
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>


namespace firestarter::oneapi {

class oneAPI {
private:
  std::thread _initThread;
  std::condition_variable _waitForInitCv;
  std::mutex _waitForInitCvMutex;

public:
  //Cuda(volatile unsigned long long *loadVar, bool useFloat, bool useDouble,
  //     unsigned matrixSize, int gpus);

  oneAPI(volatile unsigned long long *loadVar, unsigned matrixsize, int gpus);

};

} // namespace firestarter::oneAPI
