#include <firestarter/Logging/Log.hpp>
#include <firestarter/Firestarter.hpp>

#include <cstdlib>

using namespace firestarter;

#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)

extern "C" {
#include <sched.h>
#include <errno.h>
}

#define ADD_CPU_SET(cpu,cpuset) \
	do { \
		if (cpu_allowed(cpu)) { \
			CPU_SET(cpu, &cpuset); \
		} else { \
			if (cpu >= this->numThreads ) { \
				log::error() << "Error: The given bind argument (-b/--bind) includes CPU " << cpu << " that is not available on this system."; \
			} \
			else { \
				log::error() << "Error: The given bind argument (-b/--bind) cannot be implemented with the cpuset given from the OS\n" \
					<< "This can be caused by the taskset tool, cgroups, the batch system, or similar mechanisms.\n" \
					<< "Please fix the argument to match the restrictions.";\
			} \
			return EACCES; \
		} \
	} while (0)

int cpu_set(int id) {
	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(id, &mask);

	return sched_setaffinity(0, sizeof(cpu_set_t), &mask);
}

int cpu_allowed(int id) {
	cpu_set_t mask;

	CPU_ZERO(&mask);

	if (!sched_getaffinity(0, sizeof(cpu_set_t), &mask)) {
		return CPU_ISSET(id, &mask);
	}

	return 0;
}

// this code is from the C version of FIRESTARTER
// the parsing is ugly and complicated to read.
// TODO: replace this code with a nice regex
int Firestarter::parse_cpulist(cpu_set_t *cpusetPtr, const char *fsbind, unsigned *requestedNumThreads) {
	char *p,*q,*r,*s,*t;
	int i=0,p_val=0,r_val=0,s_val=0,error=0;

	cpu_set_t cpuset;
	std::memcpy(&cpuset, cpusetPtr, sizeof(cpu_set_t));

	errno=0;
	p=strdup(fsbind);
	while(p!=NULL) {
		q=strstr(p,",");
		if (q) {
			*q='\0';
			q++;
		}
		s=strstr(p,"/");
		if (s) {
			*s='\0';
			s++;
			s_val=(int)strtol(s,&t,10);
			if ((errno) || ((strcmp(t,"\0") && (t[0] !=','))) ) error++;
		}
		r=strstr(p,"-");
		if (r) {
			*r='\0';
			r++;
			r_val=(int)strtol(r,&t,10);
			if ((errno) || ((strcmp(t,"\0") && (t[0] !=',') && (t[0] !='/'))) ) error++;
		}
		p_val=(int)strtol(p,&t,10);
		if ((errno) || (p_val < 0) || (strcmp(t,"\0"))) error++;
		if (error) {
			log::error() << "Error: invalid symbols in CPU list: " << std::string(fsbind);
			return 127;
		}
		if ((s) && (s_val<=0)) {
			log::error() << "Error: s has to be >= 0 in x-y/s expressions of CPU list: " << std::string(fsbind);
			return 127;
		}
		if ((r) && (r_val < p_val)) {
			log::error() << "Error: y has to be >= x in x-y expressions of CPU list: " << std::string(fsbind);
			return 127;
		}
		if ((s)&&(r)) for (i=p_val; (int)i<=r_val; i+=s_val) {
			ADD_CPU_SET(i, cpuset);
			(*requestedNumThreads)++;
		}
		else if (r) for (i=p_val; (int)i<=r_val; i++) {
			ADD_CPU_SET(i, cpuset);
			(*requestedNumThreads)++;
		}
		else {
			ADD_CPU_SET(p_val, cpuset);
			(*requestedNumThreads)++;
		}
		p=q;
	}
	free(p);

	std::memcpy(cpusetPtr, &cpuset, sizeof(cpu_set_t));

	return EXIT_SUCCESS;
}
#endif

int Firestarter::setCpuAffinity(unsigned requestedNumThreads, std::string cpuBind) {

	if (requestedNumThreads > 0 && requestedNumThreads > this->numThreads) {
		log::warn() << "Warning: not enough CPUs for requested number of threads";
	}

#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
	cpu_set_t cpuset;

	CPU_ZERO(&cpuset);

	if (cpuBind.empty()) {
		// no cpu binding defined

		// use all CPUs if not defined otherwise
		if (requestedNumThreads == 0) {
			for (int i=0; i<this->numThreads; i++) {
				if (cpu_allowed(i)) {
					CPU_SET(i, &cpuset);
					requestedNumThreads++;
				}
			}
		} else {
			// if -n / --threads is set
			int current_cpu = 0;
			for (int i=0; i< this->numThreads; i++) {
				// search for available cpu
				while(!cpu_allowed(current_cpu)) {
					current_cpu++;

					// if rearhed end of avail cpus or max(int)
					if (current_cpu >= this->numThreads || current_cpu < 0) {
						log::error() << "Error: Your are requesting more threads than there are CPUs available in the given cpuset.\n"
							<< "This can be caused by the taskset tool, cgrous, the batch system, or similar mechanisms.\n"
							<< "Please fix the -n/--threads argument to match the restrictions.";
						return EACCES;
					}
				}
				ADD_CPU_SET(current_cpu, cpuset);

				// next cpu for next thread (or one of the following)
				current_cpu++;
			}
		}

	} else {
		// parse CPULIST for binding
		int returnCode;
		if (EXIT_SUCCESS != (returnCode = this->parse_cpulist(&cpuset, cpuBind.c_str(), &requestedNumThreads))) {
			return returnCode;
		}
	}
#else
	if (requestedNumThreads == 0) {
		requestedNumThreads = this->numThreads;
	}
#endif

	if (requestedNumThreads == 0) {
		log::error() << "Error: found no usable CPUs!";
		return 127;
	}
#if (defined(linux) || defined(__linux__)) && defined(AFFINITY)
	else {
		for (int i=0; i<this->numThreads; i++) {
			if (CPU_ISSET(i, &cpuset)) {
				this->cpuBind.push_back(i);
			}
		}
	}
#endif

	if (requestedNumThreads > this->numThreads) {
		requestedNumThreads = this->numThreads;
	}

	this->requestedNumThreads = requestedNumThreads;

	return EXIT_SUCCESS;
}

int Firestarter::getCoreIdFromPU(unsigned long long pu) {
	int width;
	hwloc_obj_t obj;

	width = hwloc_get_nbobjs_by_type(this->topology, HWLOC_OBJ_PU);

	if (width >= 1) {
		for (int i=0; i<width; i++) {
			obj = hwloc_get_obj_by_type(this->topology, HWLOC_OBJ_PU, i);
			if (obj->os_index == pu) {
				for (; obj; obj=obj->parent) {
					if (obj->type == HWLOC_OBJ_CORE) {
						return obj->logical_index;
					}
				}
			}
		}
	}

	return -1;
}

int Firestarter::getPkgIdFromPU(unsigned long long pu) {
	int width;
	hwloc_obj_t obj;

	width = hwloc_get_nbobjs_by_type(this->topology, HWLOC_OBJ_PU);

	if (width >= 1) {
		for (int i=0; i<width; i++) {
			obj = hwloc_get_obj_by_type(this->topology, HWLOC_OBJ_PU, i);
			if (obj->os_index == pu) {
				for (; obj; obj=obj->parent) {
					if (obj->type == HWLOC_OBJ_PACKAGE) {
						return obj->logical_index;
					}
				}
			}
		}
	}
	
	return -1;
}
