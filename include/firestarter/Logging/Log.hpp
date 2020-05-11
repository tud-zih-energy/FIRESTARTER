#ifndef INCLUDE_FIRESTARTER_LOG_HPP
#define INCLUDE_FIRESTARTER_LOG_HPP

#include <nitro/log/log.hpp>

#include <nitro/log/attribute/message.hpp>
#include <nitro/log/attribute/severity.hpp>

#include <nitro/log/sink/stdout.hpp>

#include <nitro/log/filter/severity_filter.hpp>

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

namespace firestarter {

	namespace logging {

		using record = nitro::log::record<nitro::log::severity_attribute, nitro::log::message_attribute>;

		template <typename Record>
		class formater {
			public:
				std::string format(Record& r) {
					std::stringstream s;
					s << r.message();

					return s.str();
				}
		};
		
		template <typename Record>
		using filter = nitro::log::filter::severity_filter<Record>;

	}

	using log = nitro::log::logger<logging::record, logging::formater,
                                 nitro::log::sink::stdout, logging::filter>;

}

#endif
