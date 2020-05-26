#include <firestarter/Firestarter.hpp>
#include <firestarter/Logging/Log.hpp>

#ifdef ENABLE_VTRACING
#include <vt_user.h>
#endif
#ifdef ENABLE_SCOREP
#include <SCOREP_User.h>
#endif

#include <cstdlib>

using namespace firestarter;

int Firestarter::init(void) {

  int returnCode;

  if (EXIT_SUCCESS != (returnCode = this->environment->setCpuAffinity(0))) {
    return EXIT_FAILURE;
  }

  this->threads = static_cast<pthread_t *>(std::aligned_alloc(
      64, this->environment->requestedNumThreads * sizeof(pthread_t)));

  bool ack;

  for (int i = 0; i < this->environment->requestedNumThreads; i++) {
    auto td = new ThreadData(i, this->environment);

    auto dataCacheSizeIt =
        td->config->platformConfig->dataCacheBufferSize.begin();
    auto ramBufferSize = td->config->platformConfig->ramBufferSize;

    td->buffersizeMem = (*dataCacheSizeIt + *std::next(dataCacheSizeIt, 1) +
                         *std::next(dataCacheSizeIt, 2) + ramBufferSize) /
                        td->config->thread / sizeof(unsigned long long);

    this->threadData.push_back(std::ref(td));

    auto t =
        pthread_create(&this->threads[i], NULL, threadWorker, std::ref(td));

    // TODO: set thread data high address
    td->mutex.lock();
    td->comm = THREAD_INIT;
    td->mutex.unlock();

    do {
      td->mutex.lock();
      ack = td->ack;
      td->mutex.unlock();
    } while (!ack);

    td->mutex.lock();
    td->ack = false;
    td->mutex.unlock();

#if 1
    td->mutex.lock();
    td->comm = THREAD_WORK;
    td->mutex.unlock();

    do {
      td->mutex.lock();
      ack = td->ack;
      td->mutex.unlock();
    } while (!ack);

    td->mutex.lock();
    td->ack = false;
    td->mutex.unlock();
#endif
  }

  log::debug() << "Started all work functions.";

  int i = 0;
  for (ThreadData *td : this->threadData) {
    pthread_join(this->threads[i], NULL);
    i++;
  }

  log::debug() << "Stopped all threads.";

  return EXIT_SUCCESS;
}

void *Firestarter::threadWorker(void *threadData) {

  int old = THREAD_WAIT;

  auto td = (ThreadData *)threadData;

  for (;;) {
    td->mutex.lock();
    int comm = td->comm;
    td->mutex.unlock();

    if (comm != old) {
      old = comm;

      td->mutex.lock();
      td->ack = true;
      td->mutex.unlock();
    } else {
      continue;
    }

    switch (comm) {
    // allocate and initialize memory
    case THREAD_INIT:
      // set affinity
      td->environment->setCpuAffinity(td->id);

      // compile payload
      td->config->payload->compilePayload(
          td->config->platformConfig->getDefaultPayloadSettings());

      // allocate memory
      td->addrMem = static_cast<unsigned long long *>(std::aligned_alloc(
          64, td->buffersizeMem * sizeof(unsigned long long)));
      // TODO: handle error

      // call init function
      td->config->payload->init(td->addrMem, td->buffersizeMem);
      break;
    // perform stress test
    case THREAD_WORK:
      // record threads start timestamp
      td->start_tsc = td->environment->timestamp();

      // will be terminated by watchdog
      for (;;) {
        // call high load function
#ifdef ENABLE_VTRACING
        VT_USER_START("HIGH_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_BEGIN("HIGH",
                                         SCOREP_USER_REGION_TYPE_COMMON);
#endif
        td->config->payload->highLoadFunction(td->addrHigh, td->period);

        // call low load function
#ifdef ENABLE_VTRACING
        VT_USER_END("HIGH_LOAD_FUNC");
        VT_USER_START("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("HIGH");
        SCOREP_USER_REGION_BY_NAME_BEGIN("LOW", SCOREP_USER_REGION_TYPE_COMMON);
#endif
        td->config->payload->lowLoadFunction(td->addrHigh, td->period);
#ifdef ENABLE_VTRACING
        VT_USER_END("LOW_LOAD_FUNC");
#endif
#ifdef ENABLE_SCOREP
        SCOREP_USER_REGION_BY_NAME_END("LOW");
#endif

        // terminate if master signals end of run and record stop timestamp
        if (*td->addrHigh == LOAD_STOP) {
          td->stop_tsc = td->environment->timestamp();

          pthread_exit(NULL);
        }
      }
      break;
    case THREAD_WAIT:
      break;
    case THREAD_STOP:
    default:
      pthread_exit(0);
    }
  }
}
