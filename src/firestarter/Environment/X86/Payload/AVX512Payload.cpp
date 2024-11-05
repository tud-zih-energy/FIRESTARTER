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

#include <firestarter/Environment/X86/Payload/AVX512Payload.hpp>
#include <firestarter/Environment/X86/Payload/CompiledX86Payload.hpp>

namespace firestarter::environment::x86::payload {

auto AVX512Payload::compilePayload(const environment::payload::PayloadSettings& Settings, bool DumpRegisters,
                                   bool ErrorDetection) const -> environment::payload::CompiledPayload::UniquePtr {
  using Imm = asmjit::Imm;
  using Zmm = asmjit::x86::Zmm;
  // NOLINTBEGIN(readability-identifier-naming)
  constexpr asmjit::x86::Mem (*zmmword_ptr)(const asmjit::x86::Gp&, int32_t) = asmjit::x86::zmmword_ptr;
  constexpr auto zmm0 = asmjit::x86::zmm0;
  constexpr auto zmm1 = asmjit::x86::zmm1;
  constexpr auto zmm2 = asmjit::x86::zmm2;
  // NOLINTEND(readability-identifier-naming)

  // Compute the sequence of instruction groups and the number of its repetions
  // to reach the desired size
  auto Sequence = Settings.sequence();
  auto Repetitions =
      environment::payload::PayloadSettings::getNumberOfSequenceRepetitions(Sequence, Settings.linesPerThread());

  // compute count of flops and memory access for performance report
  environment::payload::PayloadStats Stats;

  for (const auto& Item : Sequence) {
    auto It = InstructionFlops.find(Item);

    if (It == InstructionFlops.end()) {
      workerLog::error() << "Instruction group " << Item << " undefined in " << name() << ".";
    }

    Stats.Flops += It->second;

    It = InstructionMemory.find(Item);

    if (It != InstructionMemory.end()) {
      Stats.Bytes += It->second;
    }
  }

  Stats.Flops *= Repetitions;
  Stats.Bytes *= Repetitions;
  Stats.Instructions = Repetitions * Sequence.size() * 4 + 6;

  // calculate the buffer sizes
  const auto L1iCacheSize = Settings.instructionCacheSizePerThread();
  const auto DataCacheBufferSizes = Settings.dataCacheBufferSizePerThread();
  auto DataCacheBufferSizeIterator = DataCacheBufferSizes.begin();
  const auto L1Size = *DataCacheBufferSizeIterator;
  std::advance(DataCacheBufferSizeIterator, 1);
  const auto L2Size = *DataCacheBufferSizeIterator;
  std::advance(DataCacheBufferSizeIterator, 1);
  const auto L3Size = *DataCacheBufferSizeIterator;
  const auto RamSize = Settings.ramBufferSizePerThread();

  // calculate the reset counters for the buffers
  const auto L2LoopCount =
      environment::payload::PayloadSettings::getL2LoopCount(Sequence, Settings.linesPerThread(), L2Size);
  const auto L3LoopCount =
      environment::payload::PayloadSettings::getL3LoopCount(Sequence, Settings.linesPerThread(), L3Size);
  const auto RamLoopCount =
      environment::payload::PayloadSettings::getRAMLoopCount(Sequence, Settings.linesPerThread(), RamSize);

  asmjit::CodeHolder Code;
  Code.init(asmjit::Environment::host());

  asmjit::x86::Builder Cb(&Code);
  Cb.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateAssembler |
                          asmjit::DiagnosticOptions::kValidateIntermediate);

  const auto PointerReg = asmjit::x86::rax;
  const auto L1Addr = asmjit::x86::rbx;
  const auto L2Addr = asmjit::x86::rcx;
  const auto L3Addr = asmjit::x86::r8;
  const auto RamAddr = asmjit::x86::r9;
  const auto L2CountReg = asmjit::x86::r10;
  const auto L3CountReg = asmjit::x86::r11;
  const auto RamCountReg = asmjit::x86::r12;
  const auto TempReg = asmjit::x86::r13;
  const auto TempReg2 = asmjit::x86::rbp;
  const auto OffsetReg = asmjit::x86::r14;
  const auto AddrHighReg = asmjit::x86::r15;
  const auto IterReg = asmjit::x86::mm0;
  const auto ShiftReg = std::vector<asmjit::x86::Gp>({asmjit::x86::rdi, asmjit::x86::rsi, asmjit::x86::rdx});
  const auto ShiftReg32 = std::vector<asmjit::x86::Gp>({asmjit::x86::edi, asmjit::x86::esi, asmjit::x86::edx});
  const auto NrShiftRegs = 3;
  const auto MulRegs = 3;
  const auto AddRegs = 22;
  const auto AltDstRegs = 5;
  const auto RamReg = asmjit::x86::zmm30;

  asmjit::FuncDetail Func;
  Func.init(asmjit::FuncSignature::build<uint64_t, double*, volatile LoadThreadWorkType*, uint64_t>(
                asmjit::CallConvId::kCDecl),
            Code.environment());

  asmjit::FuncFrame Frame;
  Frame.init(Func);

  // make zmm registers dirty
  for (auto I = 0U; I < 32U; I++) {
    Frame.addDirtyRegs(Zmm(I));
  }
  for (auto I = 0U; I < 8U; I++) {
    Frame.addDirtyRegs(asmjit::x86::Mm(I));
  }
  // make all other used registers dirty except RAX
  Frame.addDirtyRegs(L1Addr, L2Addr, L3Addr, RamAddr, L2CountReg, L3CountReg, RamCountReg, TempReg, TempReg2, OffsetReg,
                     AddrHighReg, IterReg, RamAddr);
  for (const auto& Reg : ShiftReg) {
    Frame.addDirtyRegs(Reg);
  }

  asmjit::FuncArgsAssignment Args(&Func);
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
  // Initialize AVX512-Registers for FMA Operations
  Cb.vmovapd(zmm0, zmmword_ptr(PointerReg, 0));
  Cb.vmovapd(zmm1, zmmword_ptr(PointerReg, 64));
  Cb.vmovapd(zmm2, zmmword_ptr(PointerReg, 128));
  auto AddStart = MulRegs;
  auto AddEnd = MulRegs + AddRegs - 1;
  auto TransStart = AddRegs + MulRegs;
  auto TransEnd = AddRegs + MulRegs + AltDstRegs - 1;
  for (auto I = AddStart; I <= TransEnd; I++) {
    Cb.vmovapd(Zmm(I), zmmword_ptr(PointerReg, 256 + (I * 64)));
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

  Cb.align(asmjit::AlignMode::kCode, 64);

  auto Loop = Cb.newLabel();
  Cb.bind(Loop);

  auto ShiftPos = 0;
  bool Left = false;
  auto AddDest = AddStart + 1;
  auto MovDst = TransStart;
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

  for (auto Count = 0U; Count < Repetitions; Count++) {
    for (const auto& Item : Sequence) {
      if (Item == "REG") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        Cb.vfmadd231pd(Zmm(MovDst), zmm2, zmm1);
        Cb.xor_(ShiftReg[(ShiftPos + NrShiftRegs - 1) % NrShiftRegs], TempReg);
        MovDst++;
      } else if (Item == "L1_L") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        Cb.vfmadd231pd(Zmm(AddDest), zmm1, zmmword_ptr(L1Addr, 64));
        L1Increment();
      } else if (Item == "L1_BROADCAST") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        Cb.vbroadcastsd(Zmm(AddDest), ptr_64(L1Addr, 64));
        L1Increment();
      } else if (Item == "L1_S") {
        Cb.vmovapd(zmmword_ptr(L1Addr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        L1Increment();
      } else if (Item == "L1_LS") {
        Cb.vmovapd(zmmword_ptr(L1Addr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmmword_ptr(L1Addr, 128));
        L1Increment();
      } else if (Item == "L2_L") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        Cb.vfmadd231pd(Zmm(AddDest), zmm1, zmmword_ptr(L2Addr, 64));
        L2Increment();
      } else if (Item == "L2_S") {
        Cb.vmovapd(zmmword_ptr(L2Addr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        L2Increment();
      } else if (Item == "L2_LS") {
        Cb.vmovapd(zmmword_ptr(L2Addr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmmword_ptr(L2Addr, 128));
        L2Increment();
      } else if (Item == "L3_L") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        Cb.vfmadd231pd(Zmm(AddDest), zmm1, zmmword_ptr(L3Addr, 64));
        L3Increment();
      } else if (Item == "L3_S") {
        Cb.vmovapd(zmmword_ptr(L3Addr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        L3Increment();
      } else if (Item == "L3_LS") {
        Cb.vmovapd(zmmword_ptr(L3Addr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmmword_ptr(L3Addr, 128));
        L3Increment();
      } else if (Item == "L3_P") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmmword_ptr(L1Addr, 64));
        Cb.prefetcht2(ptr(L3Addr));
        L3Increment();
      } else if (Item == "RAM_L") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        Cb.vfmadd231pd(RamReg, zmm1, zmmword_ptr(RamAddr, 64));
        RamIncrement();
      } else if (Item == "RAM_S") {
        Cb.vmovapd(zmmword_ptr(RamAddr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmm2);
        RamIncrement();
      } else if (Item == "RAM_LS") {
        Cb.vmovapd(zmmword_ptr(RamAddr, 64), Zmm(AddDest));
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmmword_ptr(RamAddr, 128));
        RamIncrement();
      } else if (Item == "RAM_P") {
        Cb.vfmadd231pd(Zmm(AddDest), zmm0, zmmword_ptr(L1Addr, 64));
        Cb.prefetcht2(ptr(RamAddr));
        RamIncrement();
      } else {
        workerLog::error() << "Instruction group " << Item << " not found in " << name() << ".";
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
      if (MovDst > TransEnd) {
        MovDst = TransStart;
      }
      ShiftPos++;
      if (ShiftPos == NrShiftRegs) {
        ShiftPos = 0;
        Left = !Left;
      }
    }
  }

  Cb.movq(TempReg, IterReg); // restore iteration counter
  if (environment::payload::PayloadSettings::getRAMSequenceCount(Sequence) > 0) {
    // reset RAM counter
    auto NoRamReset = Cb.newLabel();

    Cb.sub(RamCountReg, Imm(1));
    Cb.jnz(NoRamReset);
    Cb.mov(RamCountReg, Imm(RamLoopCount));
    Cb.mov(RamAddr, PointerReg);
    Cb.add(RamAddr, Imm(L3Size));
    Cb.bind(NoRamReset);
    // adds always two instruction
    Stats.Instructions += 2;
  }
  Cb.inc(TempReg); // increment iteration counter
  if (environment::payload::PayloadSettings::getL2SequenceCount(Sequence) > 0) {
    // reset L2-Cache counter
    auto NoL2Reset = Cb.newLabel();

    Cb.sub(L2CountReg, Imm(1));
    Cb.jnz(NoL2Reset);
    Cb.mov(L2CountReg, Imm(L2LoopCount));
    Cb.mov(L2Addr, PointerReg);
    Cb.add(L2Addr, Imm(L1Size));
    Cb.bind(NoL2Reset);
    // adds always two instruction
    Stats.Instructions += 2;
  }
  Cb.movq(IterReg, TempReg); // store iteration counter
  if (environment::payload::PayloadSettings::getL3SequenceCount(Sequence) > 0) {
    // reset L3-Cache counter
    auto NoL3Reset = Cb.newLabel();

    Cb.sub(L3CountReg, Imm(1));
    Cb.jnz(NoL3Reset);
    Cb.mov(L3CountReg, Imm(L3LoopCount));
    Cb.mov(L3Addr, PointerReg);
    Cb.add(L3Addr, Imm(L2Size));
    Cb.bind(NoL3Reset);
    // adds always two instruction
    Stats.Instructions += 2;
  }
  Cb.mov(L1Addr, PointerReg);

  if (DumpRegisters) {
    emitDumpRegisterCode<Zmm>(Cb, PointerReg, zmmword_ptr);
  }

  if (ErrorDetection) {
    emitErrorDetectionCode<decltype(IterReg), Zmm>(Cb, IterReg, AddrHighReg, PointerReg, TempReg, TempReg2);
  }

  Cb.test(ptr_64(AddrHighReg), Imm(LoadThreadWorkType::LoadHigh));
  Cb.jnz(Loop);

  Cb.bind(FunctionExit);

  Cb.movq(asmjit::x86::rax, IterReg);

  Cb.emitEpilog(Frame);

  Cb.finalize();

  auto CompiledPayloadPtr = CompiledX86Payload::create<AVX512Payload>(Stats, Code);

  // skip if we could not determine cache size
  if (L1iCacheSize) {
    auto LoopSize = Code.labelOffset(FunctionExit) - Code.labelOffset(Loop);
    auto InstructionCachePercentage = 100 * LoopSize / *L1iCacheSize;

    if (LoopSize > *L1iCacheSize) {
      workerLog::warn() << "Work-loop is bigger than the L1i-Cache.";
    }

    workerLog::trace() << "Using " << LoopSize << " of " << *L1iCacheSize << " Bytes (" << InstructionCachePercentage
                       << "%) from the L1i-Cache for the work-loop.";
    workerLog::trace() << "Sequence size: " << Sequence.size();
    workerLog::trace() << "Repetition count: " << Repetitions;
  }

  return CompiledPayloadPtr;
}

auto AVX512Payload::getAvailableInstructions() const -> std::list<std::string> {
  std::list<std::string> Instructions;

  transform(InstructionFlops.begin(), InstructionFlops.end(), back_inserter(Instructions),
            [](const auto& Item) { return Item.first; });

  return Instructions;
}

void AVX512Payload::init(double* MemoryAddr, uint64_t BufferSize) const {
  X86Payload::initMemory(MemoryAddr, BufferSize, 0.27948995982e-4, 0.27948995982e-4);
}

} // namespace firestarter::environment::x86::payload