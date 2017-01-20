###############################################################################
# FIRESTARTER - A Processor Stress Test Utility
# Copyright (C) 2017 TU Dresden, Center for Information Services and High
# Performance Computing
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contact: daniel.hackenberg@tu-dresden.de
###############################################################################

def version_info(file,major_version,minor_version,version_info,build_date):
    file.write("#define VERSION_MAJOR  {}\n".format(major_version))
    file.write("#define VERSION_MINOR  {}\n".format(minor_version))
    # "" for python2, "\"\"" for python3
    if version_info is "" or "\"\"":
        file.write("#define VERSION_INFO   \"\" //additional information, e.g. \"BETA\"\n")
    else:
        file.write("#define VERSION_INFO   \"{}\" //additional information, e.g. \"BETA\"\n".format(' '+version_info.replace("\"","")))
    file.write("#define BUILDDATE      \"{}\"\n".format(build_date))

