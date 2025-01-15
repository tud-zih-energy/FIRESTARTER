/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/\>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#include "firestarter/Firestarter.hpp"
#include "firestarter/Logging/Log.hpp"

auto main(int argc, const char** argv) -> int {
  firestarter::log::info() << "FIRESTARTER - A Processor Stress Test Utility, Version " << _FIRESTARTER_VERSION_STRING
                           << "\n"
                           << "Copyright (C) " << _FIRESTARTER_BUILD_YEAR
                           << " TU Dresden, Center for Information Services and High Performance "
                              "Computing"
                           << "\n";
#ifdef _FIRESTARTER_VERSION_TEMPERED
  firestarter::log::info() << "*The version and/or year was explicitely set during build and does not "
                           << "necessarily represent the actual version.\n"
                           << "This helps maintainers to keep track of versions, e.g., on a cluster."
                           << "\n";
#endif

  try {
    firestarter::Config Cfg{argc, argv};

    firestarter::Firestarter Firestarter(std::move(Cfg));

    Firestarter.mainThread();

  } catch (std::exception const& E) {
    firestarter::log::error() << E.what();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}