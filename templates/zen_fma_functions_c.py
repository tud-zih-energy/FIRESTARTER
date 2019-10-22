###############################################################################
# FIRESTARTER - A Processor Stress Test Utility
# Copyright (C) 2019 TU Dresden, Center for Information Services and High
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

from __future__ import division
import sys
from . import util

def init_functions(file,architectures):
    isa='zen_fma'
    for each in architectures:
        for item in each.isa:
            if item == isa:
                for threads in each.threads:
                    l1_size = each.l1_size // int(threads)
                    l2_size = each.l2_size // int(threads)
                    l3_size = each.l3_size // int(threads)
                    ram_size = each.ram_size // int(threads)
                    lines = each.lines // int(threads)
                    func_name = each.arch+'_'+each.model+'_'+isa+'_'+threads+'t'
                    file.write("int init_"+func_name+"(threaddata_t* threaddata) __attribute__((noinline));\n")
                    file.write("int init_"+func_name+"(threaddata_t* threaddata)\n")
                    file.write("{\n")
                    file.write("    unsigned long long addrMem = threaddata->addrMem;\n")
                    file.write("    int i;\n")
                    file.write("\n")
# old version: one large loop that initializes indivisual elements
#                    buffersize = (l1_size+l2_size+l3_size+ram_size) // 8
#                    file.write("    // for (i = 0; i<"+str(buffersize)+"; i++) ((double*)addrMem)[i] = 0.25 + (double)(i%9267) * 0.24738995982e-4;\n")
#                    file.write("    for (i = 0; i<"+str(buffersize)+"; i++) ((double*)addrMem)[i] = 0.25 + (double)(i&0x1FFF) * 0.27948995982e-4;\n")
                    buffersize = (l1_size+l2_size+l3_size+ram_size)
                    file.write("    for (i = 0; i < INIT_BLOCKSIZE; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;\n")
                    file.write("    for (i = INIT_BLOCKSIZE; i <= "+str(buffersize)+" - INIT_BLOCKSIZE; i+= INIT_BLOCKSIZE) memcpy((void*)(addrMem+i),(void*)(addrMem+i-INIT_BLOCKSIZE),INIT_BLOCKSIZE);\n")
                    file.write("    for (; i <= "+str(buffersize)+"-8; i+=8) *((double*)(addrMem+i)) = 0.25 + (double)i * 0.27948995982e-4;\n")
                    file.write("\n")
                    flops_total=0
                    bytes_total=0
                    sequence_length=0
                    for i in range(0,len(each.instr_groups)):
                        flops=0
                        bytes=0
                        sequence_length+=int(each.proportion[i])
                        if each.instr_groups[i] == 'REG':
                            flops=8 # one 256 bit FMA operations
                        elif each.instr_groups[i] == 'L1_L':
                            flops=8 # one 256 bit FMA operations
                        elif each.instr_groups[i] == 'L1_S':
                            flops=8 # one 256 bit FMA operation
                        elif each.instr_groups[i] == 'L1_LS':
                            flops=8 # one 256 bit FMA operation
                        elif each.instr_groups[i] == 'L2_L':
                            flops=8 # one 256 bit FMA operations
                        elif each.instr_groups[i] == 'L2_S':
                            flops=8 # one 256 bit FMA operation
                        elif each.instr_groups[i] == 'L2_LS':
                            flops=8 # one 256 bit FMA operation
                        elif each.instr_groups[i] == 'L3_L':
                            flops=8 # one 256 bit FMA operations
                        elif each.instr_groups[i] == 'L3_S':
                            flops=8 # one 256 bit FMA operation
                        elif each.instr_groups[i] == 'L3_LS':
                            flops=8 # one 256 bit FMA operation
                        elif each.instr_groups[i] == 'L3_P':
                            flops=8 # one 256 bit FMA operation
                        elif each.instr_groups[i] == 'RAM_L':
                            flops=8 # one 256 bit FMA operations
                            bytes=64 # load one cache line
                        elif each.instr_groups[i] == 'RAM_S':
                            flops=8 # one 256 bit FMA operation
                            bytes=128 # one cache line, RFO + store
                        elif each.instr_groups[i] == 'RAM_LS':
                            flops=8 # one 256 bit FMA operation
                            bytes=128 # one cache line, load/RFO + store (one load and one RFO per cache line)
                        elif each.instr_groups[i] == 'RAM_P':
                            flops=8 # one 256 bit FMA operation
                            bytes=64 # prefetch one cache line
                        else:
                            print("Error: instruction group \""+each.instr_groups[i]+"\" undefined in "+isa+"_functions_c.init_functions")
                            sys.exit(2)
                        flops_total+=flops*int(each.proportion[i])
                        bytes_total+=bytes*int(each.proportion[i])
                    repeat= lines // sequence_length
                    flops_total*=repeat
                    bytes_total*=repeat
                    file.write("    threaddata->flops="+str(flops_total)+";\n")
                    file.write("    threaddata->bytes="+str(bytes_total)+";\n")
                    file.write("\n")
                    file.write("    return EXIT_SUCCESS;\n")
                    file.write("}\n")

def work_functions(file,architectures,version):
    isa='zen_fma'

    # register definitions
    pointer_reg     = 'rax' # do not modify (used for in/output)
    l1_addr         = 'rbx' # do not modify (used for in/output)
    l2_addr         = 'rcx' # do not modify (used for in/output)
    l3_addr         = 'r8'
    ram_addr        = 'r9'
    l2_count_reg    = 'r10'
    l3_count_reg    = 'r11'
    ram_count_reg   = 'r12'
    temp_reg        = 'r13'
    offset_reg      = 'r14'
    addrHigh_reg    = 'r15'
    iter_reg        = 'mm0'
    shift_reg       = ['rdi','rsi','rdx']
    nr_shift_regs   = 3
    fma_io_regs     = 6 # number of registers each used for input and output
    ram_reg         = 'ymm15'

    shift_regs=shift_reg[0]
    shift_regs_clob='\"%'+shift_reg[0]+'\"'
    for i in range(1,nr_shift_regs):
        shift_regs = shift_regs+','+shift_reg[i]
        shift_regs_clob = shift_regs_clob+', \"%'+shift_reg[i]+'\"'

    for each in architectures:
        for item in each.isa:
            if item == isa:
                for threads in each.threads:
                    l1_size = each.l1_size // int(threads)
                    l2_size = each.l2_size // int(threads)
                    l3_size = each.l3_size // int(threads)
                    ram_size = each.ram_size // int(threads)
                    lines = each.lines // int(threads)
                    func_name = each.arch+'_'+each.model+'_'+isa+'_'+threads+'t'
                    sequence = util.generate_sequence(each.instr_groups,each.proportion)
                    file.write("/**\n")
                    file.write(" * assembler implementation of processor and memory stress test\n")
                    file.write(" * ISA: "+isa.upper()+"\n")
                    file.write(" * optimized for "+each.name+" - "+threads+" thread(s) per core\n")
                    file.write(" */\n")
                    file.write("int asm_work_"+func_name+"(threaddata_t* threaddata) __attribute__((noinline));\n")
                    file.write("int asm_work_"+func_name+"(threaddata_t* threaddata)\n")
                    file.write("{\n")
                    file.write("    if (*((unsigned long long*)threaddata->addrHigh) == 0) return EXIT_SUCCESS;\n")
                    file.write("        /* input: \n")
                    file.write("         *   - threaddata->addrMem    -> rax\n")
                    file.write("         *   - threaddata->addrHigh   -> rbx\n")
                    file.write("         *   - threaddata->iterations -> rcx\n")
                    file.write("         * output: \n")
                    file.write("         *   - rax -> threaddata->iterations\n")
                    file.write("         * register usage:\n")
                    file.write("         *   - {:<12} stores original pointer to buffer, used to periodically reset other pointers\n".format(pointer_reg+":"))
                    file.write("         *   - {:<12} pointer to L1 buffer\n".format(l1_addr+":"))
                    file.write("         *   - {:<12} pointer to L2 buffer\n".format(l2_addr+":"))
                    file.write("         *   - {:<12} pointer to L3 buffer\n".format(l3_addr+":"))
                    file.write("         *   - {:<12} pointer to RAM buffer\n".format(ram_addr+":"))
                    file.write("         *   - {:<12} counter for L2-pointer reset\n".format(l2_count_reg+":"))
                    file.write("         *   - {:<12} counter for L3-pointer reset\n".format(l3_count_reg+":"))
                    file.write("         *   - {:<12} counter for RAM-pointer reset\n".format(ram_count_reg+":"))
                    file.write("         *   - {:<12} register for temporary results\n".format(temp_reg+":"))
                    file.write("         *   - {:<12} stores cacheline width as increment for buffer addresses\n".format(offset_reg+":"))
                    file.write("         *   - {:<12} stores address of shared variable that controls load level\n".format(addrHigh_reg+":"))
                    file.write("         *   - {:<12} stores iteration counter\n".format(iter_reg+":"))
                    file.write("         *   - {:<12} registers for shift operations\n".format(shift_regs+":"))
                    file.write("         *   - {:<12} data registers for SIMD instructions\n".format("ymm*,ymm*:"))
                    # file.write("         *\n")
                    # file.write("         * access sequence: \n")
                    # file.write("         * "+sequence+" \n")
                    file.write("         */\n")
                    file.write("        __asm__ __volatile__(\n")
                    file.write("        \"mov %%rax, %%"+pointer_reg+";\" // store start address of buffer in "+pointer_reg+"\n")
                    file.write("        \"mov %%rbx, %%"+addrHigh_reg+";\" // store address of shared variable that controls load level in "+addrHigh_reg+"\n")
                    file.write("        \"movq %%rcx, %%"+iter_reg+";\" // store iteration counter in "+iter_reg+"\n")
                    file.write("        \"mov $64, %%"+offset_reg+";\" // increment after each cache/memory access\n")
                    file.write("        //Initialize registers for shift operations\n")
                    for i in range(0,nr_shift_regs):
                        file.write("        \"mov $0xAAAAAAAAAAAAAAAA, %%"+shift_reg[i]+";\"\n")
                    file.write("        //Initialize AVX-Registers for FMA Operations\n")
                    file.write("        \"vmovapd (%%rax), %%ymm0;\"\n")
                    fma_input_start  = 1
                    fma_input_end    = fma_input_start + fma_io_regs - 1
                    fma_output_start = fma_input_end + 1
                    fma_output_end   = fma_output_start + fma_io_regs - 1
                    for i in range(fma_input_start, fma_output_end):
                        file.write("        \"vmovapd "+str(i*32)+"(%%rax), %%ymm"+str(i)+";\"\n")
                    file.write("        \"mov $1, %%"+temp_reg+";\"\n")
                    file.write("        \"movd %%xmm14, %%"+temp_reg+";\"\n")
                    file.write("        \"movd %%xmm13, %%"+shift_reg[0]+";\"\n")
                    file.write("        \"vbroadcastss %%xmm13, %%xmm13;\"\n")
                    file.write("        \"mov %%"+pointer_reg+", %%"+l1_addr+";\" // address for L1-buffer\n")
                    file.write("        \"mov %%"+pointer_reg+", %%"+l2_addr+";\"\n")
                    file.write("        \"add $"+str(l1_size)+", %%"+l2_addr+";\" // address for L2-buffer\n")
                    file.write("        \"mov %%"+pointer_reg+", %%"+l3_addr+";\"\n")
                    file.write("        \"add $"+str(l2_size)+", %%"+l3_addr+";\" // address for L3-buffer\n")
                    file.write("        \"mov %%"+pointer_reg+", %%"+ram_addr+";\"\n")
                    file.write("        \"add $"+str(l3_size)+", %%"+ram_addr+";\" // address for RAM-buffer\n")
                    file.write("        \"movabs $"+str(util.l2_loop_count(each,threads,sequence))+", %%"+l2_count_reg+";\" // reset-counter for L2-buffer with "+str(util.repeat(sequence,lines)*util.l2_seq_count(sequence))+" cache lines accessed per loop ("+str(round((util.l2_accesses(each,threads,sequence)*each.cl_size)/1024,2))+" KB)\n")
                    file.write("        \"movabs $"+str(util.l3_loop_count(each,threads,sequence))+", %%"+l3_count_reg+";\" // reset-counter for L3-buffer with "+str(util.repeat(sequence,lines)*util.l3_seq_count(sequence))+" cache lines accessed per loop ("+str(round((util.l3_accesses(each,threads,sequence)*each.cl_size)/1024,2))+" KB)\n")
                    file.write("        \"movabs $"+str(util.ram_loop_count(each,threads,sequence))+", %%"+ram_count_reg+";\" // reset-counter for RAM-buffer with "+str(util.repeat(sequence,lines)*util.ram_seq_count(sequence))+" cache lines accessed per loop ("+str(round((util.ram_accesses(each,threads,sequence)*each.cl_size)/1024,2))+" KB)\n")
                    file.write("\n")
                    if version.enable_mac == 1:
                        file.write("        #ifdef __MACH__ /* Mac OS compatibility */\n")
                        file.write("        \".align 6;\"      /* alignment as power of 2 */\n")
                        file.write("        #else\n")
                    file.write("        \".align 64;\"     /* alignment in bytes */\n")
                    if version.enable_mac == 1:
                        file.write("        #endif\n")
                    file.write("        \"_work_loop_"+func_name+":\"\n")
                    file.write("        /****************************************************************************************************\n")
                    file.write("         decode 0                                 decode 1                                 decode 2             decode 3 */\n")

                    fma_pos = 0
                    shift_pos = 0
                    left = 0
                    l1_offset = 0
                    count = 0
                    for i in range(0,util.repeat(sequence,lines)):
                        for item in sequence: 
                            # d0: 256 bit fma
                            # d1: 128 bit store
                            # d2: 128 shr/shl
                            # d3: add or move l[1-3] addr or shr/shl
                            if item == 'REG': 
                                d0_inst   = 'vfmadd231pd %%ymm'+str(fma_output_start+fma_pos)+', %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'xor %%'+str(shift_reg[(shift_pos+nr_shift_regs-1)%nr_shift_regs])+', %%'+str(temp_reg)+';'
                                if left == 1:
                                    d3_inst = 'shr $1, %%'+str(shift_reg[shift_pos])+';'
                                else:
                                    d3_inst = 'shl $1, %%'+str(shift_reg[shift_pos])+';'
                                comment   = '// REG ops only'
                            elif item == 'L1_L':
                                d0_inst   = 'vfmadd231pd 32(%%'+l1_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'xor %%'+str(shift_reg[(shift_pos+nr_shift_regs-1)%nr_shift_regs])+', %%'+str(temp_reg)+';'
                                l1_offset = l1_offset + each.cl_size
                                if l1_offset < l1_size*each.l1_cover:
                                    d3_inst  = 'add %%'+offset_reg+', %%'+l1_addr+';'
                                else:
                                    l1_offset = 0
                                    d3_inst  = 'mov %%'+pointer_reg+', %%'+l1_addr+';'
                                comment   = '// L1 load'
                            elif item == 'L1_S':
                                d0_inst   = 'vfmadd231pd %%ymm'+str(fma_output_start+fma_pos)+', %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%xmm'+str(fma_input_start+fma_pos)+', 32(%%'+l1_addr+');'
                                l1_offset = l1_offset + each.cl_size
                                if l1_offset < l1_size*each.l1_cover:
                                    d3_inst  = 'add %%'+offset_reg+', %%'+l1_addr+';'
                                else:
                                    l1_offset = 0
                                    d3_inst  = 'mov %%'+pointer_reg+', %%'+l1_addr+';'
                                comment   = '// L1 store'
                            elif item == 'L1_LS':
                                d0_inst   = 'vfmadd231pd 32(%%'+l1_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%ymm'+str(fma_input_start+fma_pos)+', 64(%%'+l1_addr+');'
                                l1_offset = l1_offset + each.cl_size
                                if l1_offset < l1_size*each.l1_cover:
                                    d3_inst  = 'add %%'+offset_reg+', %%'+l1_addr+';'
                                else:
                                    l1_offset = 0
                                    d3_inst  = 'mov %%'+pointer_reg+', %%'+l1_addr+';'
                                comment   = '// L1 load, L1 store'
                            elif item == 'L2_L':
                                d0_inst   = 'vfmadd231pd 64(%%'+l2_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'xor %%'+str(shift_reg[(shift_pos+nr_shift_regs-1)%nr_shift_regs])+', %%'+str(temp_reg)+';'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+l2_addr+';'
                                comment   = '// L2 load'
                            elif item == 'L2_S':
                                d0_inst   = 'vfmadd231pd %%ymm'+str(fma_output_start+fma_pos)+', %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%xmm'+str(fma_input_start+fma_pos)+', 64(%%'+l2_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+l2_addr+';'
                                comment   = '// L2 store'
                            elif item == 'L2_LS':
                                d0_inst   = 'vfmadd231pd 64(%%'+l2_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%xmm'+str(fma_input_start+fma_pos)+', 96(%%'+l2_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+l2_addr+';'
                                comment   = '// L2 load, L2 store'
                            elif item == 'L3_L':        
                                d0_inst   = 'vfmadd231pd 64(%%'+l3_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'xor %%'+str(shift_reg[(shift_pos+nr_shift_regs-1)%nr_shift_regs])+', %%'+str(temp_reg)+';'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+l3_addr+';'
                                comment   = '// L3 load'
                            elif item == 'L3_S':
                                d0_inst   = 'vfmadd231pd %%ymm'+str(fma_output_start+fma_pos)+', %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%xmm'+str(fma_input_start+fma_pos)+', 96(%%'+l3_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+l3_addr+';'
                                comment   = '// L3 store'
                            elif item == 'L3_LS':        
                                d0_inst   = 'vfmadd231pd 64(%%'+l3_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%xmm'+str(fma_input_start+fma_pos)+', 96(%%'+l3_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+l3_addr+';'
                                comment   = '// L3 load, L3 store'
                            elif item == 'L3_P':
                                d0_inst   = 'vfmadd231pd 32(%%'+l1_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'prefetcht2 (%%'+l3_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+l3_addr+';'
                                comment   = '// L3 prefetch'
                            elif item == 'RAM_L':
                                d0_inst   = 'vfmadd231pd 32(%%'+ram_addr+'), %%ymm0, %%'+str(ram_reg)+';'
                                d1_inst   = 'xor %%'+str(shift_reg[(shift_pos+nr_shift_regs-1)%nr_shift_regs])+', %%'+str(temp_reg)+';'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+ram_addr+';'
                                comment   = '// RAM load'
                            elif item == 'RAM_S':
                                d0_inst   = 'vfmadd231pd %%ymm'+str(fma_output_start+fma_pos)+', %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%xmm'+str(fma_input_start+fma_pos)+', 64(%%'+ram_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+ram_addr+';'
                                comment   = '// RAM store'
                            elif item == 'RAM_LS':
                                d0_inst   = 'vfmadd231pd 32(%%'+ram_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'vmovapd %%xmm'+str(fma_input_start+fma_pos)+', 64(%%'+ram_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+ram_addr+';'
                                comment   = '// RAM load, RAM store'
                            elif item == 'RAM_P':
                                d0_inst   = 'vfmadd231pd 32(%%'+l1_addr+'), %%ymm0, %%ymm'+str(fma_input_start+fma_pos)+';'
                                d1_inst   = 'prefetcht2 (%%'+ram_addr+');'
                                d3_inst   = 'add %%'+str(offset_reg)+', %%'+ram_addr+';'
                                comment   = '// RAM prefetch'
                            else:
                                print("Error: instruction group \""+item+"\" undefined in "+isa+"_functions_c.work_functions")
                                sys.exit(2)
                            if count % 2 == 0:
                                d2_inst = 'vpsrlq %%xmm13, %%xmm13, %%xmm14;';
                            else:
                                d2_inst = 'vpsllq %%xmm13, %%xmm13, %%xmm14;';
                            # write instruction group to file
                            file.write("        \"{:<40} {:<40} {:<20} {:<20}\" {:<}\n".format(d0_inst, d1_inst, d2_inst, d3_inst, comment))
                            # prepare register numbers for next iteration
                            fma_pos   = (fma_pos+1)%fma_io_regs
                            shift_pos = shift_pos +1
                            count += 1
                            if shift_pos == nr_shift_regs:
                                shift_pos = 0
                                if left == 1:
                                    left = 0
                                else:
                                    left = 1

                    file.write("        \"movq %%"+iter_reg+", %%"+temp_reg+";\" // restore iteration counter\n")
                    if util.ram_seq_count(sequence) > 0:
                        file.write("        //reset RAM counter\n")
                        file.write("        \"sub $1, %%"+ram_count_reg+";\"\n")
                        file.write("        \"jnz _work_no_ram_reset_"+func_name+";\"\n")
                        file.write("        \"movabs $"+str(util.ram_loop_count(each,threads,sequence))+", %%"+ram_count_reg+";\"\n")
                        file.write("        \"mov %%"+pointer_reg+", %%"+ram_addr+";\"\n")
                        file.write("        \"add $"+str(l3_size)+", %%"+ram_addr+";\"\n")
                        file.write("        \"_work_no_ram_reset_"+func_name+":\"\n")
                    file.write("        \"inc %%"+temp_reg+";\" // increment iteration counter\n")
                    if util.l2_seq_count(sequence) > 0:
                        file.write("        //reset L2-Cache counter\n")
                        file.write("        \"sub $1, %%"+l2_count_reg+";\"\n")
                        file.write("        \"jnz _work_no_L2_reset_"+func_name+";\"\n")
                        file.write("        \"movabs $"+str(util.l2_loop_count(each,threads,sequence))+", %%"+l2_count_reg+";\"\n")
                        file.write("        \"mov %%"+pointer_reg+", %%"+l2_addr+";\"\n")
                        file.write("        \"add $"+str(l1_size)+", %%"+l2_addr+";\"\n")
                        file.write("        \"_work_no_L2_reset_"+func_name+":\"\n")
                    file.write("        \"movq %%"+temp_reg+", %%"+iter_reg+";\" // store iteration counter\n")
                    if util.l3_seq_count(sequence) > 0:
                        file.write("        //reset L3-Cache counter\n")
                        file.write("        \"sub $1, %%"+l3_count_reg+";\"\n")
                        file.write("        \"jnz _work_no_L3_reset_"+func_name+";\"\n")
                        file.write("        \"movabs $"+str(util.l3_loop_count(each,threads,sequence))+", %%"+l3_count_reg+";\"\n")
                        file.write("        \"mov %%"+pointer_reg+", %%"+l3_addr+";\"\n")
                        file.write("        \"add $"+str(l2_size)+", %%"+l3_addr+";\"\n")
                        file.write("        \"_work_no_L3_reset_"+func_name+":\"\n")
                    file.write("        \"mov %%"+pointer_reg+", %%"+l1_addr+";\"\n")
                    #file.write("        \"mfence;\"\n")
                    file.write("        \"testq $1, (%%"+addrHigh_reg+");\"\n")
                    file.write("        \"jnz _work_loop_"+func_name+";\"\n")
                    file.write("        \"movq %%"+iter_reg+", %%rax;\" // restore iteration counter\n")
                    file.write("        : \"=a\" (threaddata->iterations)\n")
                    file.write("        : \"a\"(threaddata->addrMem), \"b\"(threaddata->addrHigh), \"c\" (threaddata->iterations)\n")
                    file.write("        : \"%"+l3_addr+"\", \"%"+ram_addr+"\", \"%"+l2_count_reg+"\", \"%"+l3_count_reg+"\", \"%"+ram_count_reg+"\", \"%"+temp_reg+"\", \"%"+offset_reg+"\", \"%"+addrHigh_reg+"\", "+shift_regs_clob+", \"%mm0\", \"%mm1\", \"%mm2\", \"%mm3\", \"%mm4\", \"%mm5\", \"%mm6\", \"%mm7\", \"%ymm0\", \"%ymm1\", \"%ymm2\", \"%ymm3\", \"%ymm4\", \"%ymm5\", \"%ymm6\", \"%ymm7\", \"%ymm8\", \"%ymm9\", \"%ymm10\", \"%ymm11\", \"%ymm12\", \"%ymm13\", \"%ymm14\", \"%ymm15\"\n")
                    file.write("        );\n")
                    file.write("    return EXIT_SUCCESS;\n")
                    file.write("}\n")

