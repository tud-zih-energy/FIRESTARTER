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

#include <firestarter/Environment/X86/Payload/AVXPayload.hpp>
#include <firestarter/Logging/Log.hpp>

#include <iterator>
#include <utility>

using namespace firestarter::environment::x86::payload;
using namespace asmjit;
using namespace asmjit::x86;

int AVXPayload::compilePayload(
    std::vector<std::pair<std::string, unsigned>> const &proportion,
    unsigned instructionCacheSize,
    std::list<unsigned> const &dataCacheBufferSize, unsigned ramBufferSize,
    unsigned thread, unsigned numberOfLines, bool dumpRegisters,
    bool errorDetection) {
  // Compute the sequence of instruction groups and the number of its repetions
  // to reach the desired size
  auto sequence = this->generateSequence(proportion);
  auto repetitions =
      this->getNumberOfSequenceRepetitions(sequence, numberOfLines / thread);

  // compute count of flops and memory access for performance report
  unsigned flops = 0;
  unsigned bytes = 0;

  for (const auto &item : sequence) {
    auto it = this->instructionFlops.find(item);

    if (it == this->instructionFlops.end()) {
      workerLog::error() << "Instruction group " << item << " undefined in "
                         << name() << ".";
      return EXIT_FAILURE;
    }

    flops += it->second;

    it = this->instructionMemory.find(item);

    if (it != this->instructionMemory.end()) {
      bytes += it->second;
    }
  }

  this->_flops = repetitions * flops;
  this->_bytes = repetitions * bytes;
  this->_instructions = repetitions * sequence.size() * 2 + 4;

  // calculate the buffer sizes
  auto l1i_cache_size = instructionCacheSize / thread;
  auto dataCacheBufferSizeIterator = dataCacheBufferSize.begin();
  auto l1_size = *dataCacheBufferSizeIterator / thread;
  std::advance(dataCacheBufferSizeIterator, 1);
  auto l2_size = *dataCacheBufferSizeIterator / thread;
  std::advance(dataCacheBufferSizeIterator, 1);
  auto l3_size = *dataCacheBufferSizeIterator / thread;
  auto ram_size = ramBufferSize / thread;

  // calculate the reset counters for the buffers
  auto l2_loop_count =
      getL2LoopCount(sequence, numberOfLines, l2_size * thread, thread);
  auto l3_loop_count =
      getL3LoopCount(sequence, numberOfLines, l3_size * thread, thread);
  auto ram_loop_count =
      getRAMLoopCount(sequence, numberOfLines, ram_size * thread, thread);

  CodeHolder code;
  code.init(this->rt.environment());

  if (nullptr != this->loadFunction) {
    this->rt.release(&this->loadFunction);
  }

  Builder cb(&code);
  cb.addValidationOptions(
      BaseEmitter::ValidationOptions::kValidationOptionAssembler |
      BaseEmitter::ValidationOptions::kValidationOptionIntermediate);

  auto pointer_reg = rax;
  auto l1_addr = rbx;
  auto l2_addr = rcx;
  auto l3_addr = rdx;
  auto ram_addr = rdi;
  auto l2_count_reg = r8;
  auto l3_count_reg = r9;
  auto ram_count_reg = r10;
  auto temp_reg = r11;
  auto temp_reg2 = rbp;
  auto offset_reg = r12;
  auto addrHigh_reg = r13;
  auto iter_reg = r14;
  auto shift_regs = 6;
  auto add_regs = 10;
  auto trans_regs = 6;

  FuncDetail func;
  func.init(FuncSignatureT<unsigned long long, unsigned long long *,
                           volatile unsigned long long *, unsigned long long>(
                CallConv::kIdHost),
            this->rt.environment());

  FuncFrame frame;
  frame.init(func);

  // make xmm registers dirty
  for (int i = 0; i < 16; i++) {
    frame.addDirtyRegs(Ymm(i));
  }
  // make mmx registers dirty
  for (int i = 0; i < 8; i++) {
    frame.addDirtyRegs(Mm(i));
  }
  // make all other used registers dirty except RAX
  frame.addDirtyRegs(l1_addr, l2_addr, l3_addr, ram_addr, l2_count_reg,
                     l3_count_reg, ram_count_reg, temp_reg, temp_reg2,
                     offset_reg, addrHigh_reg, iter_reg);

  FuncArgsAssignment args(&func);
  args.assignAll(pointer_reg, addrHigh_reg, iter_reg);
  args.updateFuncFrame(frame);
  frame.finalize();

  cb.emitProlog(frame);
  cb.emitArgsAssignment(frame, args);

  // stop right away if low load is selected
  auto FunctionExit = cb.newLabel();

  cb.mov(temp_reg, ptr_64(addrHigh_reg));
  cb.test(temp_reg, temp_reg);
  cb.jz(FunctionExit);

  cb.mov(offset_reg,
         Imm(64)); // increment after each cache/memory access

  // Initialize AVX-Registers for Addition
  auto add_start = 0;
  auto add_end = add_regs - 1;
  auto trans_start = add_regs;
  auto trans_end = add_regs + trans_regs - 1;
  if (add_regs > 0) {
    for (int i = add_start; i <= add_end; i++) {
      cb.vmovapd(Ymm(i), ymmword_ptr(pointer_reg, 32 * i));
    }
  }

  // Initialize MMX-Registers for shift operations
  auto shift_start = 0;
  auto shift_end = shift_regs - 1;
  if (shift_regs > 1) {
    cb.mov(temp_reg, Imm(0x5555555555555555));
    cb.movq(Mm(shift_start), temp_reg);
    for (int i = shift_start + 1; i <= shift_end; i++) {
      cb.movq(Mm(i), Mm(shift_start));
    }
  }

  // Initialize AVX-Registers for Transfer-Operations
  if (trans_regs > 0) {
    if (trans_start % 2 == 0) {
      cb.mov(temp_reg, Imm(0x0F0F0F0F0F0F0F0F));
    } else {
      cb.mov(temp_reg, Imm(0xF0F0F0F0F0F0F0F0));
    }
    cb.pinsrq(Xmm(trans_start), temp_reg, Imm(0));
    cb.pinsrq(Xmm(trans_start), temp_reg, Imm(1));
    cb.vinsertf128(Ymm(trans_start), Ymm(trans_start), Xmm(trans_start),
                   Imm(1));
    for (int i = trans_start + 1; i <= trans_end; i++) {
      if (i % 2 == 0) {
        cb.shr(temp_reg, Imm(4));
      } else {
        cb.shl(temp_reg, Imm(4));
      }
      cb.pinsrq(Xmm(i), temp_reg, Imm(0));
      cb.pinsrq(Xmm(i), temp_reg, Imm(1));
      cb.vinsertf128(Ymm(i), Ymm(i), Xmm(i), Imm(1));
    }
  }

  cb.mov(l1_addr, pointer_reg); // address for L1-buffer
  cb.mov(l2_addr, pointer_reg);
  cb.add(l2_addr, Imm(l1_size)); // address for L2-buffer
  cb.mov(l3_addr, pointer_reg);
  cb.add(l3_addr, Imm(l2_size)); // address for L3-buffer
  cb.mov(ram_addr, pointer_reg);
  cb.add(ram_addr, Imm(l3_size)); // address for RAM-buffer
  cb.mov(l2_count_reg, Imm(l2_loop_count));
  workerLog::trace() << "reset counter for L2-buffer with "
                     << l2_loop_count
                     << " cache line accesses per loop ("
		     << l2_size/1024
                     << ") KiB";
  cb.mov(l3_count_reg, Imm(l3_loop_count));
  workerLog::trace() << "reset counter for L3-buffer with "
                     << l3_loop_count
                     << " cache line accesses per loop ("
		     << l3_size/1024
                     << ") KiB";
  cb.mov(ram_count_reg, Imm(ram_loop_count));
  workerLog::trace() << "reset counter for RAM-buffer with "
                     << ram_loop_count
                     << " cache line accesses per loop ("
		     << ram_size/1024
                     << ") KiB";

  cb.align(kAlignCode, 64);

  auto Loop = cb.newLabel();
  cb.bind(Loop);

  auto left = false;
  auto shift_dst = shift_start;
  auto add_dest = add_start + 1;
  auto mov_dst = trans_start;
  auto mov_src = mov_dst + 1;
  unsigned l1_offset = 0;

#define L1_INCREMENT()                                                         \
  l1_offset += 64;                                                             \
  if (l1_offset < l1_size * 0.5) {                                             \
    cb.add(l1_addr, offset_reg);                                               \
  } else {                                                                     \
    l1_offset = 0;                                                             \
    cb.mov(l1_addr, pointer_reg);                                              \
  }

#define L2_INCREMENT() cb.add(l2_addr, offset_reg);

#define L3_INCREMENT() cb.add(l3_addr, offset_reg)

#define RAM_INCREMENT() cb.add(ram_addr, offset_reg)

  for (unsigned count = 0; count < repetitions; count++) {
    for (const auto &item : sequence) {
      if (item == "REG") {
        cb.vaddpd(
            Ymm(add_dest), Ymm(add_dest),
            Ymm(add_start + (add_dest - add_start + add_regs + 1) % add_regs));
        cb.vmovdqa(Ymm(mov_dst), Ymm(mov_src));
      } else if (item == "L1_L") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l1_addr, 32));
        L1_INCREMENT();
      } else if (item == "L1_S") {
        cb.vaddpd(
            Ymm(add_dest), Ymm(add_dest),
            Ymm(add_start + (add_dest - add_start + add_regs - 1) % add_regs));
        cb.vmovapd(xmmword_ptr(l1_addr, 32), Xmm(add_dest));
        L1_INCREMENT();
        this->_instructions++;
      } else if (item == "L1_LS") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l1_addr, 32));
        cb.vmovapd(xmmword_ptr(l1_addr, 64), Xmm(add_dest));
        L1_INCREMENT();
        this->_instructions++;
      } else if (item == "L2_L") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l2_addr, 64));
        L2_INCREMENT();
      } else if (item == "L2_S") {
        cb.vaddpd(
            Ymm(add_dest), Ymm(add_dest),
            Ymm(add_start + (add_dest - add_start + add_regs - 1) % add_regs));
        cb.vmovapd(xmmword_ptr(l2_addr, 64), Xmm(add_dest));
        L2_INCREMENT();
        this->_instructions++;
      } else if (item == "L2_LS") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l2_addr, 64));
        cb.vmovapd(xmmword_ptr(l2_addr, 96), Xmm(add_dest));
        L2_INCREMENT();
        this->_instructions++;
      } else if (item == "L3_L") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l3_addr, 64));
        L3_INCREMENT();
      } else if (item == "L3_S") {
        cb.vaddpd(
            Ymm(add_dest), Ymm(add_dest),
            Ymm(add_start + (add_dest - add_start + add_regs - 1) % add_regs));
        cb.vmovapd(xmmword_ptr(l3_addr, 96), Xmm(add_dest));
        L3_INCREMENT();
        this->_instructions++;
      } else if (item == "L3_LS") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l3_addr, 64));
        cb.vmovapd(xmmword_ptr(l3_addr, 96), Xmm(add_dest));
        L3_INCREMENT();
        this->_instructions++;
      } else if (item == "L3_P") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l1_addr, 32));
        cb.prefetcht0(ptr(l3_addr));
        L3_INCREMENT();
        this->_instructions++;
      } else if (item == "RAM_L") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(ram_addr, 64));
        RAM_INCREMENT();
      } else if (item == "RAM_S") {
        cb.vaddpd(
            Ymm(add_dest), Ymm(add_dest),
            Ymm(add_start + (add_dest - add_start + add_regs - 1) % add_regs));
        cb.vmovapd(xmmword_ptr(ram_addr, 64), Xmm(add_dest));
        RAM_INCREMENT();
        this->_instructions++;
      } else if (item == "RAM_LS") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l3_addr, 64));
        cb.vmovapd(xmmword_ptr(ram_addr, 64), Xmm(add_dest));
        RAM_INCREMENT();
        this->_instructions++;
      } else if (item == "RAM_P") {
        cb.vaddpd(Ymm(add_dest), Ymm(add_dest), ymmword_ptr(l1_addr, 32));
        cb.prefetcht2(ptr(ram_addr));
        RAM_INCREMENT();
        this->_instructions++;
      } else {
        workerLog::error() << "Instruction group " << item << " not found in "
                           << this->name() << ".";
        return EXIT_FAILURE;
      }

      if (shift_regs > 1) {
        this->_instructions++;
        if (left) {
          cb.psrlw(Mm(shift_start + (shift_dst - shift_start + 3) % shift_regs),
                   Mm(shift_dst));
        } else {
          cb.psllw(Mm(shift_start + (shift_dst - shift_start + 3) % shift_regs),
                   Mm(shift_dst));
        }
      }

      add_dest++;
      if (add_dest > add_end) {
        // DO NOT REMOVE the + 1. It serves for the good of ymm0. If it was to
        // be overriden, the values in the other registers would rise up to inf.
        add_dest = add_start + 1;
      }
      mov_dst++;
      if (mov_dst > trans_end) {
        mov_dst = trans_start;
      }
      mov_src++;
      if (mov_src > trans_end) {
        mov_src = trans_start;
      }
      if (shift_regs > 1) {
        shift_dst++;
        if (shift_dst > shift_end) {
          shift_dst = shift_start;
          left = !left;
        }
      }
    }
  }

  if (this->getRAMSequenceCount(sequence) > 0) {
    // reset RAM counter
    auto NoRamReset = cb.newLabel();

    cb.sub(ram_count_reg, Imm(1));
    cb.jnz(NoRamReset);
    cb.mov(ram_count_reg, Imm(ram_loop_count));
    cb.mov(ram_addr, pointer_reg);
    cb.add(ram_addr, Imm(l3_size));
    cb.bind(NoRamReset);
    // adds always two instruction
    this->_instructions += 2;
  }
  if (this->getL2SequenceCount(sequence) > 0) {
    // reset L2-Cache counter
    auto NoL2Reset = cb.newLabel();

    cb.sub(l2_count_reg, Imm(1));
    cb.jnz(NoL2Reset);
    cb.mov(l2_count_reg, Imm(l2_loop_count));
    cb.mov(l2_addr, pointer_reg);
    cb.add(l2_addr, Imm(l1_size));
    cb.bind(NoL2Reset);
    // adds always two instruction
    this->_instructions += 2;
  }
  if (this->getL3SequenceCount(sequence) > 0) {
    // reset L3-Cache counter
    auto NoL3Reset = cb.newLabel();

    cb.sub(l3_count_reg, Imm(1));
    cb.jnz(NoL3Reset);
    cb.mov(l3_count_reg, Imm(l3_loop_count));
    cb.mov(l3_addr, pointer_reg);
    cb.add(l3_addr, Imm(l2_size));
    cb.bind(NoL3Reset);
    // adds always two instruction
    this->_instructions += 2;
  }
  cb.inc(iter_reg); // increment iteration counter
  cb.mov(l1_addr, pointer_reg);

  if (dumpRegisters) {
    auto SkipRegistersDump = cb.newLabel();

    cb.test(ptr_64(pointer_reg, -8), Imm(firestarter::DumpVariable::Wait));
    cb.jnz(SkipRegistersDump);

    // dump all the ymm register
    for (int i = 0; i < (int)this->registerCount(); i++) {
      cb.vmovapd(
          ymmword_ptr(pointer_reg, -64 - this->registerSize() * 8 * (i + 1)),
          Ymm(i));
    }

    // set read flag
    cb.mov(ptr_64(pointer_reg, -8), Imm(firestarter::DumpVariable::Wait));

    cb.bind(SkipRegistersDump);
  }

  if (errorDetection) {
    this->emitErrorDetectionCode<decltype(iter_reg), Ymm>(
        cb, iter_reg, addrHigh_reg, pointer_reg, temp_reg, temp_reg2);
  }

  cb.test(ptr_64(addrHigh_reg), Imm(LOAD_HIGH));
  cb.jnz(Loop);

  cb.bind(FunctionExit);

  cb.mov(rax, iter_reg); // restore iteration counter

  cb.emitEpilog(frame);

  cb.finalize();

  // String sb;
  // cb.dump(sb);

  Error err = this->rt.add(&this->loadFunction, &code);
  if (err) {
    workerLog::error() << "Asmjit adding Assembler to JitRuntime failed in "
                       << __FILE__ << " at " << __LINE__;
    return EXIT_FAILURE;
  }

  // skip if we could not determine cache size
  if (l1i_cache_size != 0) {
    auto loopSize = code.labelOffset(FunctionExit) - code.labelOffset(Loop);
    auto instructionCachePercentage = 100 * loopSize / l1i_cache_size;

    if (loopSize > l1i_cache_size) {
      workerLog::warn() << "Work-loop is bigger than the L1i-Cache.";
    }

    workerLog::trace() << "Using " << loopSize << " of " << l1i_cache_size
                       << " Bytes (" << instructionCachePercentage
                       << "%) from the L1i-Cache for the work-loop.";
    workerLog::trace() << "Sequence size: " << sequence.size();
    workerLog::trace() << "Repetition count: " << repetitions;
  }

  return EXIT_SUCCESS;
}

std::list<std::string> AVXPayload::getAvailableInstructions() const {
  std::list<std::string> instructions;

  transform(this->instructionFlops.begin(), this->instructionFlops.end(),
            back_inserter(instructions),
            [](const auto &item) { return item.first; });

  return instructions;
}

void AVXPayload::init(unsigned long long *memoryAddr,
                      unsigned long long bufferSize) {
  X86Payload::init(memoryAddr, bufferSize, 1.654738925401e-10,
                   1.654738925401e-15);
}
