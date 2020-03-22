#include <firestarter/log.hpp>
#include <firestarter/firestarter.hpp>

#include <llvm/Support/Host.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>

#include <thread>

extern "C" {
#include <hwloc.h>
}

using namespace firestarter;

void Firestarter::printEnvironmentSummary(void) {

	log::info()
		<< "  system summary:\n"
		<< "    number of processors:        " << this->numPackages << "\n"
		<< "    number of cores per package: " << this->numPhysicalCoresPerPackage << "\n"
		<< "    number of threads per core:  " << this->numThreads / this->numPhysicalCoresPerPackage / this->numPackages << "\n"
		<< "    total number of threads:     " << this->numThreads << "\n";

	std::stringstream ss;

	for (auto &ent : this->cpuFeatures) {
		if (ent.getValue()) {
			ss << ent.getKey().str() << " ";
		}
	}

	log::info()
		<< "  processor characteristics:\n"
		<< "    architecture:       " << this->architecture << "\n"
		<< "    vendor:             " << this->vendor << "\n"
		<< "    processor-name:     " << this->processorName << "\n"
		<< "    model:              " << this->model << "\n"
		<< "    frequency:          " << this->clockrate / 1000000 << " MHz\n"
		<< "    supported features: " << ss.str() << "\n";
}

std::unique_ptr<llvm::MemoryBuffer> Firestarter::getFileAsStream(std::string filePath, bool showError) {
	std::error_code e;
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileStream = llvm::MemoryBuffer::getFileAsStream(filePath);
	if ((e = fileStream.getError()) && showError) {
		firestarter::log::fatal() << filePath << e.message();
		return nullptr;
	}

	return std::move(*fileStream);
}

int Firestarter::evaluateEnvironment(void) {

	int depth;
	hwloc_topology_t topology;

	hwloc_topology_init(&topology);
	hwloc_topology_load(topology);

	depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);

	if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
		this->numPackages = 1;
	} else {
		this->numPackages = hwloc_get_nbobjs_by_depth(topology, depth);
	}

	hwloc_topology_destroy(topology);

	this->numPhysicalCoresPerPackage = llvm::sys::getHostNumPhysicalCores() / this->numPackages;
	this->numThreads = std::thread::hardware_concurrency();

	llvm::sys::getHostCPUFeatures(this->cpuFeatures);

	auto procCpuinfo = this->getFileAsStream("/proc/cpuinfo");
	if (nullptr == procCpuinfo) {
		return EXIT_FAILURE;
	}

	llvm::SmallVector<llvm::StringRef, 32> lines;
	llvm::SmallVector<llvm::StringRef, 2> vendor;
	llvm::SmallVector<llvm::StringRef, 2> modelName;
	procCpuinfo->getBuffer().split(lines, "\n");
	
	for (size_t i = 0; i < lines.size(); i++) {
		if (lines[i].startswith("vendor_id")) {
			lines[i].split(vendor, ':');
		}
		if (lines[i].startswith("model name")) {
			lines[i].split(modelName, ':');
			break;
		}
	}

	if (modelName.size() == 2) {
		this->processorName = modelName[1].str();
		this->processorName.erase(0, 1);
	}

	if (vendor.size() == 2) {
		this->vendor = vendor[1].str();
		this->vendor.erase(0, 1);
	}

	llvm::Triple PT(llvm::sys::getProcessTriple());

	this->architecture = PT.getArchName().str();
	this->model = llvm::sys::getHostCPUName().str();

	// TODO: define this function to be invarient of current architecture
	if (EXIT_SUCCESS != this->getCpuClockrate()) {
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
