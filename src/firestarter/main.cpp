#include <firestarter/log.hpp>
#include <firestarter/global.hpp>

#include <nitro/broken_options/parser.hpp>

#include <llvm/Support/Host.h>
#include <llvm/ADT/Triple.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/MemoryBuffer.h>

#include <thread>

int print_copyright(void) {

	firestarter::log::info() << "\n"
		<< "This program is free software: you can redistribute it and/or modify\n"
		<< "it under the terms of the GNU General Public License as published by\n"
		<< "the Free Software Foundation, either version 3 of the License, or\n"
		<< "(at your option) any later version.\n"
		<< "\n"
		<< "You should have received a copy of the GNU General Public License\n"
		<< "along with this program.  If not, see <http://www.gnu.org/licenses/>.\n";

	return EXIT_SUCCESS;
}

int print_warranty(void) {

	firestarter::log::info() << "\n"
		<< "This program is distributed in the hope that it will be useful,\n"
		<< "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
		<< "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
		<< "GNU General Public License for more details.\n"
		<< "\n"
		<< "You should have received a copy of the GNU General Public License\n"
		<< "along with this program.  If not, see <http://www.gnu.org/licenses/>.\n";

	return EXIT_SUCCESS;
}

int print_help(nitro::broken_options::parser parser) {

	std::stringstream ss;
	parser.usage(ss);
	firestarter::log::info() << ss.str() << "\n";

	return EXIT_SUCCESS;
}

int evaluate_environment(firestarter::data_t *data) {

	data->numPhysicalCores = llvm::sys::getHostNumPhysicalCores();
	data->numThreads = std::thread::hardware_concurrency();

	llvm::sys::getHostCPUFeatures(data->cpuFeatures);

	std::error_code e;
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> cpuinfo = llvm::MemoryBuffer::getFileAsStream("/proc/cpuinfo");
	if ((e = cpuinfo.getError())) {
		firestarter::log::fatal() << "Can't read /proc/cpuinfo: " << e.message();
		return EXIT_FAILURE;
	}

	llvm::SmallVector<llvm::StringRef, 32> lines;
	llvm::SmallVector<llvm::StringRef, 2> vendor;
	llvm::SmallVector<llvm::StringRef, 2> modelName;
	llvm::SmallVector<llvm::StringRef, 2> clockrateVector;
	(*cpuinfo)->getBuffer().split(lines, "\n");
	
	for (size_t i = 0; i < lines.size(); i++) {
		if (lines[i].startswith("vendor_id")) {
			lines[i].split(vendor, ':');
		}
		if (lines[i].startswith("model name")) {
			lines[i].split(modelName, ':');
		}
		if (lines[i].startswith("cpu MHz")) {
			lines[i].split(clockrateVector, ':');
			break;
		}
	}

	if (modelName.size() == 2) {
		data->processorName = modelName[1].str();
		data->processorName.erase(0, 1);
	}

	if (vendor.size() == 2) {
		data->vendor = vendor[1].str();
		data->vendor.erase(0, 1);
	}

	std::string clockrate;

	if (clockrateVector.size() == 2) {
		clockrate = clockrateVector[1].str();
		clockrate.erase(0, 1);
	} else {
		firestarter::log::fatal() << "Can't determine clockrate from /proc/cpuinfo";
	}

	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> scalingGovernor = llvm::MemoryBuffer::getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
	if ((e = scalingGovernor.getError())) {
		firestarter::log::fatal() << "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor" << e.message();
		return EXIT_FAILURE;
	}

	// TODO: determine clockrate on x86 with TSC
	// move this to a function an file
	// getCpuClockrate
	std::string governor = (*scalingGovernor)->getBuffer().str();
	
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> scallingCurFreq = llvm::MemoryBuffer::getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> cpuinfoCurFreq = llvm::MemoryBuffer::getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq");
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> scallingMaxFreq = llvm::MemoryBuffer::getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq");
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> cpuinfoMaxFreq = llvm::MemoryBuffer::getFileAsStream("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");

	if (governor.compare("performance") || governor.compare("powersave")) {
		if (scallingCurFreq.getError()) {
			if (!cpuinfoCurFreq.getError()) {
				clockrate = (*cpuinfoCurFreq)->getBuffer().str();
			}
		} else {
			clockrate = (*scallingCurFreq)->getBuffer().str();
		}
	} else {
		if (scallingMaxFreq.getError()) {
			if(!cpuinfoMaxFreq.getError()) {
				clockrate = (*cpuinfoMaxFreq)->getBuffer().str();
			}
		} else {
			clockrate = (*scallingMaxFreq)->getBuffer().str();
		}
	}

	data->clockrate = std::stoi(clockrate);
	data->clockrate *= 1000;

	llvm::Triple PT(llvm::sys::getProcessTriple());

	data->architecture = PT.getArchName().str();
	data->model = llvm::sys::getHostCPUName().str();

	return EXIT_SUCCESS;
}

void print_environment_summary(firestarter::data_t *data) {

	// TODO: add number of processors, change to number of cores per package
	firestarter::log::info()
		<< "  system summary:\n"
		<< "    number of processors: " << "\n"
		<< "    number of cores:            " << data->numPhysicalCores << "\n"
		<< "    number of threads per core: " << data->numThreads / data->numPhysicalCores << "\n"
		<< "    total number of threads:    " << data->numThreads << "\n";

	std::stringstream ss;

	for (auto &ent : data->cpuFeatures) {
		if (ent.getValue()) {
			ss << ent.getKey().str() << " ";
		}
	}

	firestarter::log::info()
		<< "  processor characteristics:\n"
		<< "    architecture:       " << data->architecture << "\n"
		<< "    vendor:             " << data->vendor << "\n"
		<< "    processor-name:     " << data->processorName << "\n"
		<< "    model:              " << data->model << "\n"
		<< "    frequency:          " << data->clockrate / 1000000 << "MHz\n"
		<< "    supported features: " << ss.str() << "\n";
}

int main(int argc, char **argv) {

	// TODO: get year number on build
	firestarter::log::info()
		<< "FIRESTARTER - A Processor Stress Test Utility, Version " << _FIRESTARTER_VERSION_STRING << "\n"
		<< "Copyright (C) " << _FIRESTARTER_BUILD_YEAR << " TU Dresden, Center for Information Services and High Performance Computing" << "\n";

	nitro::broken_options::parser parser(argv[0]);

	parser.toggle("help", "Display usage information\n").short_name("h");
	parser.toggle("version", "Display version information\n").short_name("v");
	parser.toggle("copyright", "Display copyright information\n").short_name("c");
	parser.toggle("warranty", "Display warranty information\n").short_name("w");
	parser.toggle("debug", "Display debug output\n").short_name("d");

	try {
		auto options = parser.parse(argc, argv);

		if (options.given("debug")) {
			firestarter::logging::filter<firestarter::logging::record>::set_severity(
					nitro::log::severity_level::debug);
		} else {
			firestarter::logging::filter<firestarter::logging::record>::set_severity(
					nitro::log::severity_level::info);
		}

		if (options.given("version")) {
			return EXIT_SUCCESS;
		}

		if (options.given("copyright")) {
			return print_copyright();
		}

		if (options.given("warranty")) {
			return print_warranty();
		}

		firestarter::log::info()
			<< "This program comes with ABSOLUTELY NO WARRANTY; for details run `" << argv[0] << " -w`.\n"
			<< "This is free software, and you are welcome to redistribute it\n"
			<< "under certain conditions; run `" << argv[0] << " -c` for details.\n";

		if (options.given("help")) {
			return print_help(parser);
		}
	} catch(nitro::except::exception& e) {
		firestarter::log::info() << e.what() << "\n";
		return print_help(parser);
	}

	firestarter::data_t data;

	int returnCode;
	if (EXIT_SUCCESS != (returnCode = evaluate_environment(&data))) {
		return returnCode;
	}

	print_environment_summary(&data);

	return EXIT_SUCCESS;
}
