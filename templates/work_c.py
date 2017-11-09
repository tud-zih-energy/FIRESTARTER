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

def switch_init(file,architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                func_name = "{}_{}_{}_{}t".format(each.arch, each.model, isa, threads)
                file.write("                        case FUNC_"+func_name.upper()+":\n")
                file.write("                            tmp = init_"+func_name+"(mydata);\n")
                file.write("                            break;\n")

def switch_asm_work(file,architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                func_name = "{}_{}_{}_{}t".format(each.arch, each.model, isa, threads)
                file.write("                            case FUNC_"+func_name.upper()+":\n")
                file.write("                                tmp = asm_work_"+func_name+"(mydata);\n")
                file.write("                                break;\n")

