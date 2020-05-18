#include <firestarter/Environment/Environment.hpp>
#include <firestarter/Logging/Log.hpp>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>

using namespace firestarter::environment;

std::unique_ptr<llvm::MemoryBuffer> Environment::getScalingGovernor(void) {
	return this->getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
}

int Environment::getCpuClockrate(void) {
	auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
	if (nullptr == procCpuinfo) {
		return EXIT_FAILURE;
	}

	llvm::SmallVector<llvm::StringRef, 512> lines;
	llvm::SmallVector<llvm::StringRef, 2> clockrateVector;
	procCpuinfo->getBuffer().split(lines, "\n");
	
	for (size_t i = 0; i < lines.size(); i++) {
		if (lines[i].startswith("cpu MHz")) {
			lines[i].split(clockrateVector, ':');
			break;
		}
	}

	std::string clockrate;

	if (clockrateVector.size() == 2) {
		clockrate = clockrateVector[1].str();
		clockrate.erase(0, 1);
	} else {
		firestarter::log::fatal() << "Can't determine clockrate from /proc/cpuinfo";
	}

	std::unique_ptr<llvm::MemoryBuffer> scalingGovernor;
	if (nullptr == (scalingGovernor = this->getScalingGovernor())) {
		return EXIT_FAILURE;
	}

	std::string governor = scalingGovernor->getBuffer().str();
	
	auto scalingCurFreq = this->getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
	auto cpuinfoCurFreq = this->getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq");
	auto scalingMaxFreq = this->getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
	auto cpuinfoMaxFreq = this->getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");

	if (governor.compare("performance") || governor.compare("powersave")) {
		if (nullptr == scalingCurFreq) {
			if (nullptr != cpuinfoCurFreq) {
				clockrate = cpuinfoCurFreq->getBuffer().str();
			}
		} else {
			clockrate = scalingCurFreq->getBuffer().str();
		}
	} else {
		if (nullptr == scalingMaxFreq) {
			if(nullptr != cpuinfoMaxFreq) {
				clockrate = cpuinfoMaxFreq->getBuffer().str();
			}
		} else {
			clockrate = scalingMaxFreq->getBuffer().str();
		}
	}

	this->clockrate = std::stoi(clockrate);
	this->clockrate *= 1000;

	return EXIT_SUCCESS;
}
