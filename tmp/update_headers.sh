#!/bin/sh
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

# contact email address
CONTACT=daniel.hackenberg@tu-dresden.de
YEAR=`date +%Y`

# update copyright year definition in firestarter_global.h and main_win64.c
sed -i 's/COPYRIGHT_YEAR [0-9][0-9]*/COPYRIGHT_YEAR '$YEAR'/' source_files/firestarter_global.h
sed -i 's/COPYRIGHT_YEAR [0-9][0-9]*/COPYRIGHT_YEAR '$YEAR'/' source_files/main_win64.c

# update headers TODO increase line number if header gets larger
for i in source_files/* templates/* USAGE COPYING CHANGELOG code-generator.py config.cfg update_headers.sh
do
  sed -i '1,20 s/Copyright (C) [0-9][0-9]*/Copyright (C) '$YEAR'/' $i
  sed -i '1,20 s/Contact: [a-zA-Z][-a-zA-Z0-9.@\ ]*/Contact: '$CONTACT'/' $i
done

