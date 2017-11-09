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

def list_functions(file,architectures):
    id = 0
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                id = id + 1
                func_name = 'FUNC_'+each.arch+'_'+each.model+'_'+isa+'_'+threads+'T'
                file.write("  if (feature_available(\""+isa.upper()+"\")) printf(\"  %4.4s | %.30s | yes\\n\",\""+str(id)+"\",\""+func_name.upper()+"                             \");\n")
                file.write("  else printf(\"  %4.4s | %.30s | no\\n\",\""+str(id)+"\",\""+func_name.upper()+"                             \");\n")

def get_function_cases(file,architectures):
    id = 0
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                id = id + 1
                func_name = 'FUNC_'+each.arch+'_'+each.model+'_'+isa+'_'+threads+'T'
                file.write("       case "+str(id)+":\n")
                file.write("         if (feature_available(\""+isa.upper()+"\")) func = "+func_name.upper()+";\n")
                file.write("         else{\n")
                file.write("           fprintf(stderr, \"\\nError: Function "+str(id)+" (\\\""+func_name.upper()+"\\\") requires "+isa.upper()+", which is not supported by the processor.\\n\\n\");\n")
                file.write("         }\n")
                file.write("         break;\n")

def evaluate_environment_set_function_cases(file,architectures,families):
    for cpu_fam in families:
        file.write("          case "+cpu_fam+":\n")
        file.write("            switch (cpuinfo->model) {\n")
        for each in architectures:
            if each.cpu_family == cpu_fam:
                for cpu_model in each.cpu_model:
                    file.write("              case "+cpu_model+":\n")
                for isa in each.isa:
                    func_name = 'FUNC_'+each.arch+'_'+each.model+'_'+isa+'_'
                    file.write("                if (feature_available(\""+isa.upper()+"\")) {\n")
                    for threads in each.threads:
                        file.write("                    if (num_threads_per_core() == "+threads+") FUNCTION = "+func_name.upper()+threads.upper()+"T;\n")
                    file.write("                    if (FUNCTION == FUNC_NOT_DEFINED) {\n")
                    file.write("                        fprintf(stderr, \"Warning: no code path for %i threads per core!\\n\",num_threads_per_core());\n")
                    file.write("                    }\n")
                    file.write("                }\n")
                file.write("                if (FUNCTION == FUNC_NOT_DEFINED) {\n")
                file.write("                    fprintf(stderr, \"\\nWarning: "+isa.upper()+" is requiered for architecture \\\""+each.arch.upper()+"\\\", but is not supported!\\n\");\n")
                file.write("                }\n")
                file.write("                break;\n")
        file.write("            default:\n")
        file.write("                fprintf(stderr, \"\\nWarning: %s family %i, model %i is not supported by this version of FIRESTARTER!\\n         Check project website for updates.\\n\",cpuinfo->vendor,cpuinfo->family,cpuinfo->model);\n")
        file.write("            }\n")
        file.write("            break;\n")

def evaluate_environment_select_fallback_function(file,templates):
    for templ in templates:
        file.write("    /* use "+templ.feature_req.upper()+" as fallback if available*/\n")
        file.write("    if ((FUNCTION == FUNC_NOT_DEFINED)&&(feature_available(\""+templ.feature_req.upper()+"\"))) {\n")
        file.write("        /* use function for correct number of threads per core if available */\n")
        for fb in templ.fallback:
            # TODO: this implementation does only support fallbacks with single digit number of threads
            file.write("        if(num_threads_per_core() == "+fb[-2:-1]+") {\n")
            file.write("            FUNCTION = "+fb.upper()+";\n")
            file.write("            fprintf(stderr, \"Warning: using function "+fb.upper()+" as fallback.\\n\");\n")
            file.write("            fprintf(stderr, \"         You can use the parameter --function to try other functions.\\n\");\n")
            file.write("        }\n")
        # TODO: this implementation does only support fallbacks with single digit number of threads
        file.write("        /* use function for "+templ.fallback[0][-2:-1]+" threads per core if no function for actual number of thread per core exists*/\n")
        file.write("        if (FUNCTION == FUNC_NOT_DEFINED)\n")
        file.write("        {\n")
        file.write("            FUNCTION = "+templ.fallback[0].upper()+";\n")
        file.write("            fprintf(stderr, \"Warning: using function "+templ.fallback[0].upper()+" as fallback.\\n\");\n")
        file.write("            fprintf(stderr, \"         You can use the parameter --function to try other functions.\\n\");\n")
        file.write("        }\n")
        file.write("    }\n")

def evaluate_environment_set_buffersize(file,architectures):
    for each in architectures:
        for isa in each.isa:
            for threads in each.threads:
                func_name = 'FUNC_'+each.arch+'_'+each.model+'_'+isa+'_'+threads+'T'
                file.write("    case "+func_name.upper()+":\n")
                size = int(each.l1_size) // int(threads)
                file.write("        BUFFERSIZE[0] = "+str(size)+";\n")
                size = int(each.l2_size) // int(threads)
                file.write("        BUFFERSIZE[1] = "+str(size)+";\n")
                size = int(each.l3_size) // int(threads)
                file.write("        BUFFERSIZE[2] = "+str(size)+";\n")
                size = int(each.ram_size) // int(threads)
                file.write("        RAMBUFFERSIZE = "+str(size)+";\n")
                file.write("        if (verbose) {\n")
                file.write("            printf(\"\\n  Taking "+isa.upper()+" path optimized for "+each.name+" - "+threads+" thread(s) per core\");\n")
                file.write("            printf(\"\\n  Used buffersizes per thread:\\n\");\n")
                file.write("            for (i = 0; i < MAX_CACHELEVELS; i++) if (BUFFERSIZE[i] > 0) printf(\"    - L%d-Cache: %d Bytes\\n\", i + 1, BUFFERSIZE[i]);\n")
                file.write("            printf(\"    - Memory: %llu Bytes\\n\\n\", RAMBUFFERSIZE);\n")
                file.write("        }\n")
                file.write("        break;\n")

def main_getopt(file,version):
    opts_noarg="chvwqar"
    opts_arg="i:t:l:p:n:"
    if version.enable_cuda == 1:
        opts_noarg=opts_noarg+"f"
        opts_arg=opts_arg+"m:g:"
    file.write("        #if (defined(linux) || defined(__linux__)) && defined (AFFINITY)\n")
    file.write("        c = getopt_long(argc, argv, \""+opts_noarg+"b:"+opts_arg+"\", long_options, NULL);\n")
    file.write("        #else\n")
    file.write("        c = getopt_long(argc, argv, \""+opts_noarg+opts_arg+"\", long_options, NULL);\n")
    file.write("        #endif\n")

