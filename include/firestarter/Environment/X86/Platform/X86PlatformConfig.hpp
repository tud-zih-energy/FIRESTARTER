/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020 TU Dresden, Center for Information Services and High
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

#pragma once

#include <firestarter/Environment/Platform/PlatformConfig.hpp>
#include <firestarter/Environment/X86/Payload/X86Payload.hpp>

namespace firestarter::environment::x86::platform {

class X86PlatformConfig : public environment::platform::PlatformConfig {
private:
  unsigned _family;
  std::list<unsigned> _models;
  unsigned _currentFamily;
  unsigned _currentModel;
  unsigned _currentThreads;

public:
  X86PlatformConfig(std::string name, unsigned family,
                    std::initializer_list<unsigned> models,
                    std::initializer_list<unsigned> threads,
                    unsigned instructionCacheSize,
                    std::initializer_list<unsigned> dataCacheBufferSize,
                    unsigned ramBuffersize, unsigned lines,
                    unsigned currentFamily, unsigned currentModel,
                    unsigned currentThreads, payload::X86Payload *payload)
      : PlatformConfig(name, threads, instructionCacheSize, dataCacheBufferSize,
                       ramBuffersize, lines, payload),
        _family(family), _models(models), _currentFamily(currentFamily),
        _currentModel(currentModel), _currentThreads(currentThreads) {}

  bool isDefault() const override {
    return _family == _currentFamily &&
           (std::find(_models.begin(), _models.end(), _currentModel) !=
            _models.end()) &&
           isAvailable();
  }
};

} // namespace firestarter::environment::x86::platform
