#include <firestarter/firestarter.hpp>
#include <firestarter/log.hpp>

#include <thread>

using namespace firestarter;

void Firestarter::run(void) {
	for (int i=0; i<this->numThreads; i++) {
		auto threadData = new ThreadData(i);
		this->threadData.push_back(std::ref(threadData));

		std::thread t(&Firestarter::threadWorker, this, std::ref(threadData));
		t.detach();

		// TODO: set thread data high address
		threadData->comm = THREAD_WORK;
		while(!threadData->ack)
			;
		threadData->ack = 0;
	}

	for (ThreadData *td : this->threadData) {
		td->comm = THREAD_STOP;
		while (!td->ack)
			;
		td->ack = 0;
	}
}

void Firestarter::threadWorker(ThreadData *threadData) {
	int id = threadData->getId();

	for (;;) {
		switch (threadData->comm) {
		case THREAD_INIT:
			break;
		case THREAD_WORK:
			threadData->ack = 1;
			break;
		case THREAD_WAIT:
			break;
		case THREAD_STOP:
		default:
			threadData->ack = 1;
			delete threadData;
			return;
		}
	}
}
