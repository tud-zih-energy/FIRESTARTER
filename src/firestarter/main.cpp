#include <firestarter/log.hpp>
#include <firestarter/cpuinfo.hpp>
#include <firestarter/perf.hpp>
#include <firestarter/evalplatform.hpp>

#include <nitro/broken_options/parser.hpp>

int print_help(nitro::broken_options::parser parser) {

	std::stringstream ss;
	parser.usage(ss);
	std::cout << ss.str() << std::endl;

	return 1;
}

int main(int argc, char **argv) {

	std::cout
		<< "FIRESTARTER - A Processor Stress Test Utility, Version " << _FIRESTARTER_VERSION_STRING << std::endl
		<< "Copyright (C) 2018 TU Dresden, Center for Information Services and High Performance Computing" << std::endl
		<< std::endl;

	nitro::broken_options::parser parser;

	parser.toggle("debug", "Toggle debug output").short_name("d");
	parser.toggle("help", "Print this help").short_name("h");

	try {
		auto options = parser.parse(argc, argv);

		if (options.given("debug")) {
			firestarter::logging::filter<firestarter::logging::record>::set_severity(
					nitro::log::severity_level::debug);
		} else {
			firestarter::logging::filter<firestarter::logging::record>::set_severity(
					nitro::log::severity_level::info);
		}

		if (options.given("help")) {
			return print_help(parser);
		}
	} catch(nitro::except::exception& e) {
		std::cout << e.what() << std::endl;
		return print_help(parser);
	}

	std::cout << sys::getHostCPUName().str() << std::endl;
	std::cout << sys::getHostNumPhysicalCores() << std::endl;
	std::cout << std::thread::hardware_concurrency() << std::endl;

	llvm::StringMap<bool> features;
	if (sys::getHostCPUFeatures(features)) {
		for (auto &ent : features) {
			if (ent.getValue()) {
				std::cout << ent.getKey().str() << std::endl;
			}
		}
	}

	return 0;
}
