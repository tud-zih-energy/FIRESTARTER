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

def function_definitions(file,architectures):
    id = 1
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                def_name = "FUNC_{}_{}_{}_{}T".format(each.arch, each.model, isa, threads)
                file.write("#define {} {}\n".format(def_name.upper().ljust(30), id))
                id = id + 1

def init_functions(file,architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                func_name = "init_{}_{}_{}_{}t".format(each.arch, each.model, isa, threads)
                file.write("int {}(threaddata_t* threaddata) __attribute__((noinline));\n".format(func_name))
                file.write("int {}(threaddata_t* threaddata);\n\n".format(func_name))


def stress_test_functions(file,architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                func_name = "asm_work_{}_{}_{}_{}t".format(each.arch, each.model, isa, threads)
                file.write("int {}(threaddata_t* threaddata) __attribute__((noinline));\n".format(func_name))
                file.write("int {}(threaddata_t* threaddata);\n\n".format(func_name))

