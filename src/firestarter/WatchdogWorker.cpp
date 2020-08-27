#include <firestarter/Firestarter.hpp>

#include <cerrno>
#include <csignal>

#define NSEC_PER_SEC 1000000000

#define DO_SLEEP(sleepret, target, remaining)                                  \
  do {                                                                         \
    if (watchdog_terminate) {                                                  \
      log::info() << "\nCaught shutdown signal, ending now ...";               \
      return EXIT_SUCCESS;                                                     \
    }                                                                          \
    sleepret = nanosleep(&target, &remaining);                                 \
    while (sleepret == -1 && errno == EINTR && !watchdog_terminate) {          \
      sleepret = nanosleep(&remaining, &remaining);                            \
    }                                                                          \
    if (sleepret == -1) {                                                      \
      switch (errno) {                                                         \
      case EFAULT:                                                             \
        log::error()                                                           \
            << "Found a bug in FIRESTARTER, error on copying for nanosleep";   \
        break;                                                                 \
      case EINVAL:                                                             \
        log::error()                                                           \
            << "Found a bug in FIRESTARTER, invalid setting for nanosleep";    \
        break;                                                                 \
      case EINTR:                                                              \
        log::info() << "\nCaught shutdown signal, ending now ...";             \
        break;                                                                 \
      default:                                                                 \
        log::error() << "Error calling nanosleep: " << errno;                  \
        break;                                                                 \
      }                                                                        \
      set_load(LOAD_STOP);                                                     \
      if (errno == EINTR) {                                                    \
        return EXIT_SUCCESS;                                                   \
      } else {                                                                 \
        return EXIT_FAILURE;                                                   \
      }                                                                        \
    }                                                                          \
  } while (0)

using namespace firestarter;

namespace {
static volatile pthread_t watchdog_thread;
static volatile bool watchdog_terminate = false;
static volatile unsigned long long *loadvar;
} // namespace

void set_load(unsigned long long value) {
  // signal load change to workers
  *loadvar = value;
  __asm__ __volatile__("mfence;");
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void sigalrm_handler(int signum) {}
#pragma GCC diagnostic pop
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
static void sigterm_handler(int signum) {
  // required for cases load = {0,100}, which do no enter the loop
  set_load(LOAD_STOP);
  // exit loop
  // used in case of 0 < load < 100
  watchdog_terminate = true;

  pthread_kill(watchdog_thread, SIGALRM);
}
#pragma GCC diagnostic pop
#pragma clang diagnostic pop

int Firestarter::watchdogWorker(std::chrono::microseconds period,
                                std::chrono::microseconds load,
                                std::chrono::seconds timeout) {

  using clock = std::chrono::high_resolution_clock;
  using nsec = std::chrono::nanoseconds;
  using usec = std::chrono::microseconds;
  using sec = std::chrono::seconds;

  loadvar = &this->loadVar;

  // setup signal handlers
  sigset_t signal_mask;

  sigemptyset(&signal_mask);
  sigaddset(&signal_mask, SIGINT);
  sigaddset(&signal_mask, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signal_mask, NULL);

  watchdog_thread = pthread_self();

  std::signal(SIGALRM, sigalrm_handler);

  std::signal(SIGTERM, sigterm_handler);
  std::signal(SIGINT, sigterm_handler);

  // values for wrapper to pthreads nanosleep
  int sleepret;
  struct timespec target, remaining;

  // calculate idle time to be the rest of the period
  auto idle = period - load;

  // elapsed time
  nsec time(0);

  // do no enter the loop if we do not have to set the load level periodically,
  // at 0 or 100 load.
  if (period > usec::zero()) {
    // this first time is critical as the period will be alligend from this
    // point
    std::chrono::time_point<clock> startTime = clock::now();

    // this loop will set the load level periodically.
    for (;;) {
      std::chrono::time_point<clock> currentTime = clock::now();

      // get the time already advanced in the current timeslice
      // this can happen if a load function does not terminates just on time
      nsec advance = std::chrono::duration_cast<nsec>(currentTime - startTime) %
                     std::chrono::duration_cast<nsec>(period);

      // subtract the advaned time from our timeslice by spilting it based on
      // the load level
      nsec load_reduction =
          (std::chrono::duration_cast<nsec>(load).count() * advance) /
          std::chrono::duration_cast<nsec>(period).count();
      nsec idle_reduction = advance - load_reduction;

      // signal high load level
      set_load(LOAD_HIGH);

      // calculate values for nanosleep
      nsec load_nsec = load - load_reduction;
      target.tv_nsec = load_nsec.count() % NSEC_PER_SEC;
      target.tv_sec = load_nsec.count() / NSEC_PER_SEC;

      // wait for time to be ellapsed with high load
#ifdef ENABLE_VTRACING
      VT_USER_START("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_BEGIN("WD_HIGH",
                                       SCOREP_USER_REGION_TYPE_COMMON);
#endif
      DO_SLEEP(sleepret, target, remaining);
#ifdef ENABLE_VTRACING
      VT_USER_END("WD_HIGH");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_END("WD_HIGH");
#endif

      // terminate if an interrupt by the user was fired
      if (watchdog_terminate) {
        set_load(LOAD_STOP);

        return EXIT_SUCCESS;
      }

      // signal low load
      set_load(LOAD_LOW);

      // calculate values for nanosleep
      nsec idle_nsec = idle - idle_reduction;
      target.tv_nsec = idle_nsec.count() % NSEC_PER_SEC;
      target.tv_sec = idle_nsec.count() / NSEC_PER_SEC;

      // wait for time to be ellapsed with low load
#ifdef ENABLE_VTRACING
      VT_USER_START("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_BEGIN("WD_LOW",
                                       SCOREP_USER_REGION_TYPE_COMMON);
#endif
      DO_SLEEP(sleepret, target, remaining);
#ifdef ENABLE_VTRACING
      VT_USER_END("WD_LOW");
#endif
#ifdef ENABLE_SCOREP
      SCOREP_USER_REGION_BY_NAME_END("WD_LOW");
#endif

      // increment elapsed time
      time += period;

      // exit when termination signal is received or timeout is reached
      if (watchdog_terminate || (timeout > sec::zero() && (time > timeout))) {
        set_load(LOAD_STOP);

        return EXIT_SUCCESS;
      }
    }
  }

  // if timeout is set, sleep for this time and stop execution.
  // else return and wait for sigterm handler to request threads to stop.
  if (timeout > sec::zero()) {
    target.tv_nsec = 0;
    target.tv_sec = timeout.count();

    DO_SLEEP(sleepret, target, remaining);

    set_load(LOAD_STOP);

    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}
