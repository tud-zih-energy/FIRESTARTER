#include <firestarter/log.hpp>
#include <firestarter/firestarter.hpp>

#include <cxxopts.hpp>

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

int print_help(cxxopts::Options parser) {

	firestarter::log::info() << parser.help();

	return EXIT_SUCCESS;
}

int main(int argc, char **argv) {

	// TODO: get year number on build
	firestarter::log::info()
		<< "FIRESTARTER - A Processor Stress Test Utility, Version " << _FIRESTARTER_VERSION_STRING << "\n"
		<< "Copyright (C) " << _FIRESTARTER_BUILD_YEAR << " TU Dresden, Center for Information Services and High Performance Computing" << "\n";

	cxxopts::Options parser(argv[0]);

	parser.add_options()
		("h,help", "Display usage information")
		("v,version", "Display version information")
		("c,copyright", "Display copyright information")
		("w,warranty", "Display warranty information")
		("d,debug", "Display debug output")
		("n,threads", "Specify the number of threads. Cannot be combined with -b | --bind, which impicitly specifies the number of threads",
		 cxxopts::value<unsigned>()->default_value("0"), "N")
		;
	// TODO:
	// a: list functions
	// i: use specific function
	// r report
	// t timeout
	// l load
	// p perdoid
	// n count
	// b cpulist (only linux)
	//
	// TODO: cuda
	// f: usegpufloat
	// g: gpus
	// m: matrixsize

	int numThreads;

	try {
		auto options = parser.parse(argc, argv);

		if (options.count("debug")) {
			firestarter::logging::filter<firestarter::logging::record>::set_severity(
					nitro::log::severity_level::debug);
		} else {
			firestarter::logging::filter<firestarter::logging::record>::set_severity(
					nitro::log::severity_level::info);
		}

		if (options.count("version")) {
			return EXIT_SUCCESS;
		}

		if (options.count("copyright")) {
			return print_copyright();
		}

		if (options.count("warranty")) {
			return print_warranty();
		}

		firestarter::log::info()
			<< "This program comes with ABSOLUTELY NO WARRANTY; for details run `" << argv[0] << " -w`.\n"
			<< "This is free software, and you are welcome to redistribute it\n"
			<< "under certain conditions; run `" << argv[0] << " -c` for details.\n";

		if (options.count("help")) {
			return print_help(parser);
		}

		numThreads = options["threads"].as<unsigned>();

	} catch(std::exception& e) {
		firestarter::log::info() << e.what() << "\n";
		return print_help(parser);
	}

	auto firestarter = new firestarter::Firestarter();

	int returnCode;
	if (EXIT_SUCCESS != (returnCode = firestarter->evaluateEnvironment())) {
		delete firestarter;
		return returnCode;
	}

	firestarter->printEnvironmentSummary();

	firestarter->run();

	return EXIT_SUCCESS;
}
