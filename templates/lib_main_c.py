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

def switch_init(file, architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                func_name = "{}_{}_{}_{}t".format(each.arch, each.model, isa, threads)
                file.write("        case FUNC_"+func_name.upper()+":\n")
                file.write("            return init_"+func_name+";\n")
                file.write("            break;\n")

def switch_asm_work(file, architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                func_name = "{}_{}_{}_{}t".format(each.arch, each.model, isa, threads)
                file.write("        case FUNC_" + func_name.upper() + ":\n")
                file.write("            return asm_work_" + func_name + ";\n")
                file.write("            break;\n")

def evaluate_environment_set_function_cases(file, architectures, families):
    for cpu_fam in families:
        file.write("          case "+cpu_fam+":\n")
        file.write("            switch (cpuinfo->model) {\n")
        for each in architectures:
            if each.cpu_family == cpu_fam:
                for cpu_model in each.cpu_model:
                    file.write("              case "+cpu_model+":\n")
                for isa in each.isa:
                    func_name = 'FUNC_'+each.arch+'_'+each.model+'_'+isa
                    file.write("                if (feature_available(\""+isa.upper()+"\")) {\n")
                    file.write("                    result = "+func_name.upper()+";\n")
                    file.write("                }\n")
                file.write("                break;\n")
        file.write("            }\n")
        file.write("            break;\n")

def evaluate_environment_set_buffersize(file, architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                mem_size = int(each.l1_size) // int(threads) * 2 + int(each.l2_size) // int(threads) + int(each.l3_size) // int(threads) + int(each.ram_size) // int(threads)
                func_name = 'FUNC_'+each.arch+'_'+each.model+'_'+isa+'_'+threads+'T'
                file.write("    case "+func_name.upper()+":\n")
                file.write("        return {} + 64 + 2 * sizeof(unsigned long long);\n".format(mem_size))
                file.write("        break;\n")
