/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2020-2023 TU Dresden, Center for Information Services and High
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

#include <firestarter/Environment/X86/Payload/FMA4Payload.hpp>

namespace firestarter::environment::x86::payload {

auto FMA4Payload::compilePayload(std::vector<std::pair<std::string, unsigned>> const& Proportion,
                                 unsigned InstructionCacheSize, std::list<unsigned> const& DataCacheBufferSize,
                                 unsigned RamBufferSize, unsigned Thread, unsigned NumberOfLines, bool DumpRegisters,
                                 bool ErrorDetection) -> int {
  using namespace asmjit;
  using namespace asmjit::x86;

  // Compute the sequence of instruction groups and the number of its repetions
  // to reach the desired size
  auto Sequence = generateSequence(Proportion);
  auto Repetitions = getNumberOfSequenceRepetitions(Sequence, NumberOfLines / Thread);

  // compute count of flops and memory access for performance report
  Flops = 0;
  Bytes = 0;

  for (const auto& Item : Sequence) {
    auto It = InstructionFlops.find(Item);

    if (It == InstructionFlops.end()) {
      workerLog::error() << "Instruction group " << Item << " undefined in " << name() << ".";
      return EXIT_FAILURE;
    }

    Flops += It->second;

    It = InstructionMemory.find(Item);

    if (It != InstructionMemory.end()) {
      Bytes += It->second;
    }
  }

  Flops *= Repetitions;
  Bytes *= Repetitions;
  Instructions = Repetitions * Sequence.size() * 4 + 6;

  // calculate the buffer sizes
  const auto L1iCacheSize = InstructionCacheSize / Thread;
  auto DataCacheBufferSizeIterator = DataCacheBufferSize.begin();
  const auto L1Size = *DataCacheBufferSizeIterator / Thread;
  std::advance(DataCacheBufferSizeIterator, 1);
  const auto L2Size = *DataCacheBufferSizeIterator / Thread;
  std::advance(DataCacheBufferSizeIterator, 1);
  const auto L3Size = *DataCacheBufferSizeIterator / Thread;
  const auto RamSize = RamBufferSize / Thread;

  // calculate the reset counters for the buffers
  const auto L2LoopCount = getL2LoopCount(Sequence, NumberOfLines, L2Size * Thread, Thread);
  const auto L3LoopCount = getL3LoopCount(Sequence, NumberOfLines, L3Size * Thread, Thread);
  const auto RamLoopCount = getRAMLoopCount(Sequence, NumberOfLines, RamSize * Thread, Thread);

  CodeHolder Code;
  Code.init(Rt.environment());

  if (nullptr != LoadFunction) {
    Rt.release(&LoadFunction);
  }

  Builder Cb(&Code);
  Cb.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateAssembler |
                          asmjit::DiagnosticOptions::kValidateIntermediate);

  const auto PointerReg = rax;
  const auto L1Addr = rbx;
  const auto L2Addr = rcx;
  const auto L3Addr = r8;
  const auto RamAddr = r9;
  const auto L2CountReg = r10;
  const auto L3CountReg = r11;
  const auto RamCountReg = r12;
  const auto TempReg = r13;
  const auto TempReg2 = rbp;
  const auto OffsetReg = r14;
  const auto AddrHighReg = r15;
  const auto IterReg = mm0;
  const auto ShiftReg = std::vector<Gp>({rdi, rsi, rdx});
  const auto ShiftReg32 = std::vector<Gp>({edi, esi, edx});
  const auto NbShiftRegs = 3;
  const auto MulRegs = 2;
  const auto AddRegs = 9;
  const auto AltDestRegs = 3;
  const auto RamReg = xmm15;

  FuncDetail Func;
  Func.init(FuncSignatureT<uint64_t, uint64_t*, volatile LoadThreadWorkType*, uint64_t>(CallConvId::kCDecl),
            Rt.environment());

  FuncFrame Frame;
  Frame.init(Func);

  // make (x|y)mm registers dirty
  for (int I = 0; I < 16; I++) {
    Frame.addDirtyRegs(Ymm(I));
  }
  for (int I = 0; I < 8; I++) {
    Frame.addDirtyRegs(Mm(I));
  }
  // make all other used registers dirty except RAX
  Frame.addDirtyRegs(L1Addr, L2Addr, L3Addr, RamAddr, L2CountReg, L3CountReg, RamCountReg, TempReg, TempReg2, OffsetReg,
                     AddrHighReg, IterReg, RamAddr);
  for (const auto& Reg : ShiftReg) {
    Frame.addDirtyRegs(Reg);
  }

  FuncArgsAssignment Args(&Func);
  // FIXME: asmjit assigment to mm0 does not seem to be supported
  Args.assignAll(PointerReg, AddrHighReg, TempReg);
  Args.updateFuncFrame(Frame);
  Frame.finalize();

  Cb.emitProlog(Frame);
  Cb.emitArgsAssignment(Frame, Args);

  // FIXME: movq from temp_reg to iter_reg
  Cb.movq(IterReg, TempReg);

  // stop right away if low load is selected
  auto FunctionExit = Cb.newLabel();

  Cb.mov(TempReg, ptr_64(AddrHighReg));
  Cb.test(TempReg, TempReg);
  Cb.jz(FunctionExit);

  Cb.mov(OffsetReg,
         Imm(64)); // increment after each cache/memory access
  // Initialize registers for shift operations
  for (auto const& Reg : ShiftReg32) {
    Cb.mov(Reg, Imm(0xAAAAAAAA));
  }
  // Initialize AVX-Registers for FMA4 Operations
  Cb.vmovapd(ymm0, ymmword_ptr(PointerReg));
  Cb.vmovapd(ymm1, ymmword_ptr(PointerReg));
  auto AddStart = MulRegs;
  auto AddEnd = MulRegs + AddRegs - 1;
  auto TransStart = AddRegs + MulRegs;
  auto TransEnd = AddRegs + MulRegs + AltDestRegs - 1;
  for (int I = AddStart; I <= TransEnd; I++) {
    Cb.vmovapd(Ymm(I), ymmword_ptr(PointerReg, 256 + I * 32));
  }
  Cb.mov(L1Addr, PointerReg); // address for L1-buffer
  Cb.mov(L2Addr, PointerReg);
  Cb.add(L2Addr, Imm(L1Size)); // address for L2-buffer
  Cb.mov(L3Addr, PointerReg);
  Cb.add(L3Addr, Imm(L2Size)); // address for L3-buffer
  Cb.mov(RamAddr, PointerReg);
  Cb.add(RamAddr, Imm(L3Size)); // address for RAM-buffer
  Cb.mov(L2CountReg, Imm(L2LoopCount));
  workerLog::trace() << "reset counter for L2-buffer with " << L2LoopCount << " cache line accesses per loop ("
                     << L2Size / 1024 << ") KiB";
  Cb.mov(L3CountReg, Imm(L3LoopCount));
  workerLog::trace() << "reset counter for L3-buffer with " << L3LoopCount << " cache line accesses per loop ("
                     << L3Size / 1024 << ") KiB";
  Cb.mov(RamCountReg, Imm(RamLoopCount));
  workerLog::trace() << "reset counter for RAM-buffer with " << RamLoopCount << " cache line accesses per loop ("
                     << RamSize / 1024 << ") KiB";

  Cb.align(AlignMode::kCode, 64);

  auto Loop = Cb.newLabel();
  Cb.bind(Loop);

  auto ShiftPos = 0;
  bool Left = false;
  auto AddDest = AddStart + 1;
  auto MovDest = TransStart;
  auto MovSrc = MovDest + 1;
  unsigned L1Offset = 0;

  const auto L1Increment = [&Cb, &L1Offset, &L1Size, &L1Addr, &OffsetReg, &PointerReg]() {
    L1Offset += 64;
    if (L1Offset < L1Size * 0.5) {
      Cb.add(L1Addr, OffsetReg);
    } else {
      L1Offset = 0;
      Cb.mov(L1Addr, PointerReg);
    }
  };
  const auto L2Increment = [&Cb, &L2Addr, &OffsetReg]() { Cb.add(L2Addr, OffsetReg); };
  const auto L3Increment = [&Cb, &L3Addr, &OffsetReg]() { Cb.add(L3Addr, OffsetReg); };
  const auto RamIncrement = [&Cb, &RamAddr, &OffsetReg]() { Cb.add(RamAddr, OffsetReg); };

  for (unsigned Count = 0; Count < Repetitions; Count++) {
    for (const auto& Item : Sequence) {
      if (Item == "REG") {
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        Cb.vfmaddpd(Xmm(MovDest), Xmm(MovDest), xmm1, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 2) % AddRegs)));
        Cb.xor_(ShiftReg[(ShiftPos + NbShiftRegs - 1) % NbShiftRegs], TempReg);
        MovDest++;
      } else if (Item == "L1_L") {
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        Cb.vfmaddpd(Ymm(AddDest), Ymm(AddDest), ymm1, ymmword_ptr(L1Addr, 32));
        L1Increment();
      } else if (Item == "L1_S") {
        Cb.vmovapd(xmmword_ptr(L1Addr, 32), Xmm(AddDest));
        Cb.vfmaddpd(Ymm(AddDest), Ymm(AddDest), ymm0, Ymm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        L1Increment();
      } else if (Item == "L1_LS") {
        Cb.vmovapd(xmmword_ptr(L1Addr, 64), Xmm(AddDest));
        Cb.vfmaddpd(Ymm(AddDest), Ymm(AddDest), ymm0, ymmword_ptr(L1Addr, 32));
        L1Increment();
      } else if (Item == "L2_L") {
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm1, xmmword_ptr(L2Addr, 64));
        L2Increment();
      } else if (Item == "L2_S") {
        Cb.vmovapd(xmmword_ptr(L2Addr, 64), Xmm(AddDest));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        L2Increment();
      } else if (Item == "L2_LS") {
        Cb.vmovapd(xmmword_ptr(L2Addr, 96), Xmm(AddDest));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, xmmword_ptr(L2Addr, 64));
        L2Increment();
      } else if (Item == "L3_L") {
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm1, xmmword_ptr(L3Addr, 64));
        L3Increment();
      } else if (Item == "L3_S") {
        Cb.vmovapd(xmmword_ptr(L3Addr, 96), Xmm(AddDest));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        L3Increment();
      } else if (Item == "L3_LS") {
        Cb.vmovapd(xmmword_ptr(L3Addr, 96), Xmm(AddDest));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, xmmword_ptr(L3Addr, 64));
        L3Increment();
      } else if (Item == "L3_P") {
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, xmmword_ptr(L1Addr, 32));
        Cb.prefetcht2(ptr(L3Addr));
        L3Increment();
      } else if (Item == "RAM_L") {
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        Cb.vfmaddpd(RamReg, RamReg, xmm1, xmmword_ptr(RamAddr, 64));
        RamIncrement();
      } else if (Item == "RAM_S") {
        Cb.vmovapd(xmmword_ptr(RamAddr, 64), Xmm(AddDest));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        RamIncrement();
      } else if (Item == "RAM_LS") {
        Cb.vmovapd(xmmword_ptr(RamAddr, 64), Xmm(AddDest));
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, xmmword_ptr(RamAddr, 32));
        RamIncrement();
      } else if (Item == "RAM_P") {
        Cb.vfmaddpd(Xmm(AddDest), Xmm(AddDest), xmm0, xmmword_ptr(L1Addr, 32));
        Cb.prefetcht2(ptr(RamAddr));
        RamIncrement();
      } else {
        workerLog::error() << "Instruction group " << Item << " not found in " << name() << ".";
        return EXIT_FAILURE;
      }

      if (Left) {
        Cb.shr(ShiftReg32[ShiftPos], Imm(1));
      } else {
        Cb.shl(ShiftReg32[ShiftPos], Imm(1));
      }
      AddDest++;
      if (AddDest > AddEnd) {
        AddDest = AddStart;
      }
      if (MovDest > TransEnd) {
        MovDest = TransStart;
      }
      MovSrc++;
      if (MovSrc > TransEnd) {
        MovSrc = TransStart;
      }
      ShiftPos++;
      if (ShiftPos == NbShiftRegs) {
        ShiftPos = 0;
        Left = !Left;
      }
    }
  }

  Cb.movq(TempReg, IterReg); // restore iteration counter
  if (getRAMSequenceCount(Sequence) > 0) {
    // reset RAM counter
    auto NoRamReset = Cb.newLabel();

    Cb.sub(RamCountReg, Imm(1));
    Cb.jnz(NoRamReset);
    Cb.mov(RamCountReg, Imm(RamLoopCount));
    Cb.mov(RamAddr, PointerReg);
    Cb.add(RamAddr, Imm(L3Size));
    Cb.bind(NoRamReset);
    // adds always two instruction
    Instructions += 2;
  }
  Cb.inc(TempReg); // increment iteration counter
  if (getL2SequenceCount(Sequence) > 0) {
    // reset L2-Cache counter
    auto NoL2Reset = Cb.newLabel();

    Cb.sub(L2CountReg, Imm(1));
    Cb.jnz(NoL2Reset);
    Cb.mov(L2CountReg, Imm(L2LoopCount));
    Cb.mov(L2Addr, PointerReg);
    Cb.add(L2Addr, Imm(L1Size));
    Cb.bind(NoL2Reset);
    // adds always two instruction
    Instructions += 2;
  }
  Cb.movq(IterReg, TempReg); // store iteration counter
  if (getL3SequenceCount(Sequence) > 0) {
    // reset L3-Cache counter
    auto NoL3Reset = Cb.newLabel();

    Cb.sub(L3CountReg, Imm(1));
    Cb.jnz(NoL3Reset);
    Cb.mov(L3CountReg, Imm(L3LoopCount));
    Cb.mov(L3Addr, PointerReg);
    Cb.add(L3Addr, Imm(L2Size));
    Cb.bind(NoL3Reset);
    // adds always two instruction
    Instructions += 2;
  }
  Cb.mov(L1Addr, PointerReg);

  if (DumpRegisters) {
    auto SkipRegistersDump = Cb.newLabel();

    Cb.test(ptr_64(PointerReg, -8), Imm(firestarter::DumpVariable::Wait));
    Cb.jnz(SkipRegistersDump);

    // dump all the ymm register
    for (unsigned I = 0; I < registerCount(); I++) {
      Cb.vmovapd(ymmword_ptr(PointerReg, -64 - (registerSize() * 8 * (I + 1))), Ymm(I));
    }

    // set read flag
    Cb.mov(ptr_64(PointerReg, -8), Imm(firestarter::DumpVariable::Wait));

    Cb.bind(SkipRegistersDump);
  }

  if (ErrorDetection) {
    emitErrorDetectionCode<decltype(IterReg), Ymm>(Cb, IterReg, AddrHighReg, PointerReg, TempReg, TempReg2);
  }

  Cb.test(ptr_64(AddrHighReg), Imm(LoadThreadWorkType::LoadHigh));
  Cb.jnz(Loop);

  Cb.bind(FunctionExit);

  Cb.movq(rax, IterReg);

  Cb.emitEpilog(Frame);

  Cb.finalize();

  // String sb;
  // cb.dump(sb);

  Error Err = Rt.add(&LoadFunction, &Code);
  if (Err) {
    workerLog::error() << "Asmjit adding Assembler to JitRuntime failed in " << __FILE__ << " at " << __LINE__;
    return EXIT_FAILURE;
  }

  // skip if we could not determine cache size
  if (L1iCacheSize != 0) {
    auto LoopSize = Code.labelOffset(FunctionExit) - Code.labelOffset(Loop);
    auto InstructionCachePercentage = 100 * LoopSize / L1iCacheSize;

    if (LoopSize > L1iCacheSize) {
      workerLog::warn() << "Work-loop is bigger than the L1i-Cache.";
    }

    workerLog::trace() << "Using " << LoopSize << " of " << L1iCacheSize << " Bytes (" << InstructionCachePercentage
                       << "%) from the L1i-Cache for the work-loop.";
    workerLog::trace() << "Sequence size: " << Sequence.size();
    workerLog::trace() << "Repetition count: " << Repetitions;
  }

  return EXIT_SUCCESS;
}

auto FMA4Payload::getAvailableInstructions() const -> std::list<std::string> {
  std::list<std::string> Instructions;

  transform(InstructionFlops.begin(), InstructionFlops.end(), back_inserter(Instructions),
            [](const auto& Item) { return Item.first; });

  return Instructions;
}

void FMA4Payload::init(uint64_t* MemoryAddr, uint64_t BufferSize) {
  X86Payload::init(MemoryAddr, BufferSize, 0.27948995982e-4, 0.27948995982e-4);
}

} // namespace firestarter::environment::x86::payload