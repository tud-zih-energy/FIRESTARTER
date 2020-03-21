#include <firestarter/log.hpp>
#include <firestarter/firestarter.hpp>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>

#include <ctime>

extern "C" {
#include <firestarter/x86.h>

#include <sys/time.h>
}

using namespace firestarter;

std::unique_ptr<llvm::MemoryBuffer> Firestarter::getScalingGovernor(void) {
	return this->getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
}

int Firestarter::genericGetCpuClockrate(void) {
	auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
	if (nullptr == procCpuinfo) {
		return EXIT_FAILURE;
	}

	llvm::SmallVector<llvm::StringRef, _HW_DETECT_MAX_OUTPUT> lines;
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


#ifdef __ARCH_X86

int Firestarter::hasInvariantRdtsc()
{
    unsigned long long a=0,b=0,c=0,d=0;
    int res=0;

    if (has_rdtsc()) {

        /* TSCs are usable if CPU supports only one frequency in C0 (no speedstep/Cool'n'Quite)
           or if multiple frequencies are available and the constant/invariant TSC feature flag is set */

				if (0 == this->vendor.compare("GenuineIntel")) {
            /*check if Powermanagement and invariant TSC are supported*/
            if (has_cpuid())
            {
                a=1;
                cpuid(&a,&b,&c,&d);
                /* no Frequency control */
                if ((!(d&(1<<22)))&&(!(c&(1<<7)))) res=1;
                a=0x80000000;
                cpuid(&a,&b,&c,&d);
                if (a >=0x80000007)
                {
                    a=0x80000007;
                    cpuid(&a,&b,&c,&d);
                    /* invariant TSC */
                    if (d&(1<<8)) res =1;
                }
            }
        }

				if (0 == this->vendor.compare("AuthenticAMD")) {
            /*check if Powermanagement and invariant TSC are supported*/
            if (has_cpuid())
            {
                a=0x80000000;
                cpuid(&a,&b,&c,&d);
                if (a >=0x80000007)
                {
                    a=0x80000007;
                    cpuid(&a,&b,&c,&d);

                    /* no Frequency control */
                    if ((!(d&(1<<7)))&&(!(d&(1<<1)))) res=1;
                    /* invariant TSC */
                    if (d&(1<<8)) res =1;
                }
                /* assuming no frequency control if cpuid does not provide the extended function to test for it */
                else res=1;
            }
        }
    }

    return res;
}

// measures clockrate using the Time-Stamp-Counter
// only constant TSCs will be used (i.e. power management indepent TSCs)
// save frequency in highest P-State or use generic fallback if no invarient TSC is available
int Firestarter::getCpuClockrate(void) {
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::microseconds microseconds;

	unsigned long long start1_tsc,start2_tsc,end1_tsc,end2_tsc;
//	unsigned long long end_time-start_time;
	unsigned long long clock_lower_bound,clock_upper_bound,clock;
	unsigned long long clockrate=0;
	int i,num_measurements=0,min_measurements;

	unsigned long long start_time,end_time;
	struct timeval ts;

	//Clock::time_point start_time, end_time;

	auto scalingGovernor = this->getScalingGovernor();
	if (nullptr == scalingGovernor) {
		return EXIT_FAILURE;
	}

	std::string governor = scalingGovernor->getBuffer().str();

	/* non invariant TSCs can be used if CPUs run at fixed frequency */
	if (!this->hasInvariantRdtsc() && governor.compare("performance") && governor.compare("powersave")) {
		return this->genericGetCpuClockrate();
	}

	min_measurements=5;

	if (!has_rdtsc()) {
		return this->genericGetCpuClockrate();
	}

	i = 3;

	do {
			//start timestamp
			start1_tsc=timestamp();
        gettimeofday(&ts,NULL);
			//start_time = Clock::now();
			start2_tsc=timestamp();
        start_time=ts.tv_sec*1000000+ts.tv_usec;

			//waiting
			do {
					end1_tsc=timestamp();
			}
			while (end1_tsc<start2_tsc+1000000*i);   /* busy waiting */

			//end timestamp
			do{
				end1_tsc=timestamp();
          gettimeofday(&ts,NULL);
				//end_time = Clock::now();
				end2_tsc=timestamp();
          end_time=ts.tv_sec*1000000+ts.tv_usec;
				//end_time-start_time = std::chrono::duration_cast<microseconds>(end_time - start_time).count();
			}
			while (start_time == end_time);

			clock_lower_bound=(((end1_tsc-start2_tsc)*1000000)/(end_time-start_time));
			clock_upper_bound=(((end2_tsc-start1_tsc)*1000000)/(end_time-start_time));

			// if both values differ significantly, the measurement could have been interrupted between 2 rdtsc's
			if (((double)clock_lower_bound>(((double)clock_upper_bound)*0.999))&&((end_time-start_time)>2000))
			{
					num_measurements++;
					clock=(clock_lower_bound+clock_upper_bound)/2;
					if(clockrate==0) clockrate=clock;
					else if (clock<clockrate) clockrate=clock;
			}
			i+=2;
	} while (((end_time-start_time)<10000)||(num_measurements<min_measurements));

	this->clockrate = clockrate;

	return EXIT_SUCCESS;
}

#elif __ARCH_UNKNOWN

int Firestarter::getCpuClockrate(void) {
	return this->genericGetCpuClockrate();
}
#endif
