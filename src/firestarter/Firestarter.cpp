#include <firestarter/Firestarter.hpp>

using namespace firestarter;

Firestarter::Firestarter(void) {

	hwloc_topology_init(&this->topology);

	// do not filter icaches
	hwloc_topology_set_cache_types_filter(this->topology, HWLOC_TYPE_FILTER_KEEP_ALL);

	hwloc_topology_load(this->topology);
}

Firestarter::~Firestarter(void) {

	hwloc_topology_destroy(this->topology);
}
