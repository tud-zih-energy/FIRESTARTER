/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/X86/X86FunctionSelection.hpp"
#include "firestarter/X86/X86ProcessorInformation.hpp"
#include <memory>

auto main(int /*argc*/, const char** /*argv*/) -> int {
  firestarter::logging::Filter<firestarter::logging::record>::set_severity(nitro::log::severity_level::info);

  firestarter::x86::X86FunctionSelection Env;
  const auto ProcessorInfos = std::make_shared<firestarter::x86::X86ProcessorInformation>();

  Env.printFunctionSummary(ProcessorInfos, /*ForceYes=*/true);

  return EXIT_SUCCESS;
}