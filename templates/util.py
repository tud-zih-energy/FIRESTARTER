## generate a sequence with the requested proportion of the selected instruction groups
## tries to distribute the instructions evenly
def generate_sequence(instr_groups,proportion):
    sequence = []
    for i in range(0,int(proportion[0])):
        sequence.append(instr_groups[0])
    for j in range (1,len(instr_groups)):
        if int(proportion[j]) > 0:
            for i in range(0,int(proportion[j])):
                sequence.insert(1+i* (len(sequence)+int(proportion[j])-i) // int(proportion[j]),instr_groups[j])
    return sequence

## number of accesses in the sequence that access a certain level of the memory hierarchy
def l1_seq_count(sequence):
    count = 0
    for item in sequence:
        if item[:2] == 'L1':
            count+=1
    return count

def l2_seq_count(sequence):
    count = 0
    for item in sequence:
        if item[:2] == 'L2':
            count+=1
    return count

def l3_seq_count(sequence):
    count = 0
    for item in sequence:
        if item[:2] == 'L3':
            count+=1
    return count

def ram_seq_count(sequence):
    count = 0
    for item in sequence:
        if item[:3] == 'RAM':
            count+=1
    return count

## number of repetitions required to reach specified number of lines (instruction groups)
def repeat(sequence,lines):
    return lines // len(sequence)

## initial values for reset counters (number of loop iterations until end of buffer is reached)
def l1_loop_count(arch,threads,sequence):
    count = 0
    if l1_seq_count(sequence) > 0:
        count = int (arch.l1_cover * arch.l1_size // int(threads) // int(arch.cl_size) // (repeat(sequence,arch.lines // int(threads)) * l1_seq_count(sequence)))
    return count

def l2_loop_count(arch,threads,sequence):
    count = 0
    if l2_seq_count(sequence) > 0:
        count = int (arch.l2_cover * arch.l2_size // int(threads) // int(arch.cl_size) // (repeat(sequence,arch.lines // int(threads)) * l2_seq_count(sequence)))
    return count

def l3_loop_count(arch,threads,sequence):
    count = 0
    if l3_seq_count(sequence) > 0:
        count = int (arch.l3_cover * arch.l3_size // int(threads) // int(arch.cl_size) // (repeat(sequence,arch.lines // int(threads)) * l3_seq_count(sequence)))
    return count

def ram_loop_count(arch,threads,sequence):
    count = 0
    if ram_seq_count(sequence) > 0:
        count = int (arch.ram_cover * arch.ram_size // int(threads) // int(arch.cl_size) // (repeat(sequence,arch.lines // int(threads)) * ram_seq_count(sequence)))
    return count

## number of accesses until reset of pointer
def l1_accesses(arch,threads,sequence):
    return int(l1_loop_count(arch,threads,sequence)*(repeat(sequence, arch.lines // int(threads)) * l1_seq_count(sequence)))

def l2_accesses(arch,threads,sequence):
    return int(l2_loop_count(arch,threads,sequence)*(repeat(sequence, arch.lines // int(threads)) * l2_seq_count(sequence)))

def l3_accesses(arch,threads,sequence):
    return int(l3_loop_count(arch,threads,sequence)*(repeat(sequence, arch.lines // int(threads)) * l3_seq_count(sequence)))

def ram_accesses(arch,threads,sequence):
    return int(ram_loop_count(arch,threads,sequence)*(repeat(sequence, arch.lines // int(threads)) * ram_seq_count(sequence)))


def termination_condition(file, addr_high_reg, func_name):
    file.write("        \"sub $1, %%" + addr_high_reg + ";\"\n")
    # file.write("        \"testq $1, (%%" + addr_high_reg + ");\"\n")
    file.write("        \"jnz _work_loop_"+func_name+";\"\n")
