#ifndef INCLUDE_FIRESTARTER_LOG_HPP
#define INCLUDE_FIRESTARTER_LOG_HPP

#include <nitro/log/log.hpp>

#include <nitro/log/attribute/message.hpp>
#include <nitro/log/attribute/omp_thread_id.hpp>
#include <nitro/log/attribute/severity.hpp>
#include <nitro/log/attribute/timestamp.hpp>

#include <nitro/log/sink/stdout_omp.hpp>

#include <nitro/log/filter/severity_filter.hpp>

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

namespace firestarter {

	namespace logging {

		using record = nitro::log::record<nitro::log::severity_attribute, nitro::log::message_attribute,
			                                nitro::log::omp_thread_id_attribute, nitro::log::timestamp_attribute>;

		template <typename Record>
		class formater {
			public:
				std::string format(Record& r) {
					std::stringstream s;
					s << "[" << r.severity() << "]: " << r.message();

					return s.str();
				}
		};
		
		template <typename Record>
		using filter = nitro::log::filter::severity_filter<Record>;

	}

	using log = nitro::log::logger<logging::record, logging::formater, nitro::log::sink::stdout_omp,
				                         logging::filter>;

}

#endif
