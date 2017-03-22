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

def list_functions(file,architectures,templates):
    id = 0
    for each in architectures:
        for isa in each.isa:
            for tmpl in templates:
                if ("ISA_"+isa.upper() == tmpl.name) and (tmpl.win64_incl == 1):
                    for threads in each.threads:
                        id = id + 1
                        func_name = 'func_'+each.arch+'_'+each.model+'_'+isa+'_'+threads+'T'
                        file.write("    if (has_feature("+tmpl.feature_req.upper().replace('.','_')+")) printf(\"  %4.4s | %.30s | yes\\n\",\""+str(id)+"\",\""+func_name.upper()+"                             \");\n")
                        file.write("    else printf(\"  %4.4s | %.30s | no \\n\",\""+str(id)+"\",\""+func_name.upper()+"                             \");\n")

def get_function_cases(file,architectures,templates):
    id = 0
    for each in architectures:
        for isa in each.isa:
            for tmpl in templates:
                if ("ISA_"+isa.upper() == tmpl.name) and (tmpl.win64_incl == 1):
                    for threads in each.threads:
                        id = id + 1
                        func_name = 'func_'+each.arch+'_'+each.model+'_'+isa+'_'+threads+'t'
                        file.write("       case "+str(id)+":\n")
                        file.write("         func = "+func_name.upper()+";\n")
                        file.write("         break;\n")

def WorkerThread_select_function(file,architectures,templates):
    for each in architectures:
        for isa in each.isa:
            for tmpl in templates:
                if ("ISA_"+isa.upper() == tmpl.name) and (tmpl.win64_incl == 1):
                    for threads in each.threads:
                        func_name = each.arch+'_'+each.model+'_'+isa+'_'+threads+'t'
                        file.write("    case FUNC_"+func_name.upper()+":\n")
                        if (int(threads) == 1):
                            file.write("      p =  _mm_malloc(13406208*8,4096);\n")
                        if (int(threads) == 2):
                            file.write("      p = _mm_malloc(6703104*8,4096);\n")
                        if (int(threads) == 4):
                            file.write("      p = _mm_malloc(3351552*8,4096);\n")
                        file.write("      data->addrMem = (unsigned long long) p;\n")
                        file.write("      init_"+func_name+"(data);\n")
                        file.write("      while(*((unsigned long long*)(data->addrHigh)) != LOAD_STOP){\n")
                        file.write("        asm_work_"+func_name+"(data);\n")
                        file.write("        low_load_function(data->addrHigh, PERIOD);\n")
                        file.write("      }\n")
                        file.write("      break;\n")

def main_function_info(file,architectures,templates):
    for each in architectures:
        for isa in each.isa:
            for tmpl in templates:
                if ("ISA_"+isa.upper() == tmpl.name) and (tmpl.win64_incl == 1):
                    for threads in each.threads:
                        func_name = 'func_'+each.arch+'_'+each.model+'_'+isa+'_'+threads+'t'
                        file.write("    case "+func_name.upper()+":\n")
                        file.write("      printf(\"\\nTaking "+isa.upper()+" code path optimized for "+each.name+" - "+threads+" thread(s) per core\\n\");\n")
                        file.write("      break;\n")

def main_set_function_cases(file,architectures,families,templates):
    for cpu_fam in families:
        file.write("          case "+cpu_fam+":\n")
        file.write("            switch (model) {\n")
        for each in architectures:
            for isa in each.isa:
                for tmpl in templates:
                    if ("ISA_"+isa.upper() == tmpl.name) and (tmpl.win64_incl == 1):

                        if each.cpu_family == cpu_fam:
                            for cpu_model in each.cpu_model:
                                file.write("              case "+cpu_model+":\n")
                            for isa in each.isa:
                                func_name = 'FUNC_'+each.arch+'_'+each.model+'_'+isa+'_'
                                file.write("                if (has_feature("+isa.upper().replace('.','_')+")) {\n")
                                for threads in each.threads:
                                    file.write("                    if (threads_per_core == "+threads+") func = "+func_name.upper()+threads.upper()+"T;\n")
                                file.write("                    if (func == FUNC_NOT_DEFINED) {\n")
                                file.write("                        fprintf(stderr, \"Warning: no code path for %lu threads per core!\\n\",threads_per_core);\n")
                                file.write("                    }\n")
                                file.write("                }\n")
                            file.write("                if (func == FUNC_NOT_DEFINED) {\n")
                            file.write("                    fprintf(stderr, \"\\nWarning: "+isa.upper()+" is requiered for architecture \\\""+each.arch.upper()+"\\\", but is not supported!\\n\");\n")
                            file.write("                }\n")
                            file.write("                break;\n")
        file.write("            default:\n")
        file.write("                fprintf(stderr, \"\\nWarning: %s family %i, model %i is not supported by this version of FIRESTARTER!\\n         Check project website for updates.\\n\",vendor,family,model);\n")
        file.write("            }\n")
        file.write("            break;\n")

def main_select_fallback_function(file,templates):
    for templ in templates:
        if templ.win64_incl == 1:
            file.write("  /* use "+templ.feature_req.upper()+" as fallback if available*/\n")
            file.write("  if ((func == FUNC_NOT_DEFINED)&&(has_feature("+templ.feature_req.upper().replace('.','_')+"))) {\n")
            file.write("      /* use function for correct number of threads per core if available */\n")
            for fb in templ.fallback:
                # TODO: this implementation does only support fallbacks with single digit number of threads
                file.write("      if(threads_per_core == "+fb[-2:-1]+") {\n")
                file.write("          func = "+fb.upper()+";\n")
                file.write("          fprintf(stderr, \"Warning: using function "+fb.upper()+" as fallback.\\n\");\n")
                file.write("          fprintf(stderr, \"         You can use the parameter --function to try other functions.\\n\");\n")
                file.write("      }\n")
            # TODO: this implementation does only support fallbacks with single digit number of threads
            file.write("      /* use function for "+templ.fallback[0][-2:-1]+" threads per core if no function for actual number of thread per core exists*/\n")
            file.write("      if (func == FUNC_NOT_DEFINED)\n")
            file.write("      {\n")
            file.write("          func = "+templ.fallback[0].upper()+";\n")
            file.write("          fprintf(stderr, \"Warning: using function "+templ.fallback[0].upper()+" as fallback.\\n\");\n")
            file.write("          fprintf(stderr, \"         You can use the parameter --function to try other functions.\\n\");\n")
            file.write("      }\n")
            file.write("  }\n")

