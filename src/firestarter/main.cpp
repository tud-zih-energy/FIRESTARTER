#include <firestarter/log.hpp>
#include <firestarter/firestarter.hpp>

#include <nitro/broken_options/parser.hpp>

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

	auto firestarter = new firestarter::Firestarter();

	int returnCode;
	if (EXIT_SUCCESS != (returnCode = firestarter->evaluateEnvironment())) {
		delete firestarter;
		return returnCode;
	}

	firestarter->printEnvironmentSummary();

	return EXIT_SUCCESS;
}
