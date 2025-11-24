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

#include "firestarter/X86/Payload/SSE2Payload.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/Payload/CompiledPayload.hpp"
#include "firestarter/Payload/PayloadSettings.hpp"
#include "firestarter/Payload/PayloadStats.hpp"
#include "firestarter/X86/Payload/CompiledX86Payload.hpp"
#include "firestarter/X86/Payload/X86Payload.hpp"

#include <asmjit/x86.h>
#include <cstdint>
#include <iterator>
#include <vector>

namespace firestarter::x86::payload {

auto SSE2Payload::compilePayload(const firestarter::payload::PayloadSettings& Settings, bool DumpRegisters,
                                 bool ErrorDetection,
                                 bool PrintAssembler) const -> firestarter::payload::CompiledPayload::UniquePtr {
  using Imm = asmjit::Imm;
  using Mm = asmjit::x86::Mm;
  using Xmm = asmjit::x86::Xmm;
  // NOLINTNEXTLINE(readability-identifier-naming)
  constexpr asmjit::x86::Mem (*xmmword_ptr)(const asmjit::x86::Gp&, int32_t) = asmjit::x86::xmmword_ptr;

  // Compute the sequence of instruction groups and the number of its repetions
  // to reach the desired size
  auto Sequence = Settings.sequence();
  auto Repetitions =
      firestarter::payload::PayloadSettings::getNumberOfSequenceRepetitions(Sequence, Settings.linesPerThread());

  // compute count of flops and memory access for performance report
  firestarter::payload::PayloadStats Stats;

  for (const auto& Item : Sequence) {
    auto It = instructionFlops().find(Item);

    if (It == instructionFlops().end()) {
      workerLog::error() << "Instruction group " << Item << " undefined in " << name() << ".";
    }

    Stats.Flops += It->second;

    It = instructionMemory().find(Item);

    if (It != instructionMemory().end()) {
      Stats.Bytes += It->second;
    }
  }

  Stats.Flops *= Repetitions;
  Stats.Bytes *= Repetitions;
  Stats.Instructions = Repetitions * Sequence.size() * 2 + 4;

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
      firestarter::payload::PayloadSettings::getL2LoopCount(Sequence, Settings.linesPerThread(), L2Size);
  const auto L3LoopCount =
      firestarter::payload::PayloadSettings::getL3LoopCount(Sequence, Settings.linesPerThread(), L3Size);
  const auto RamLoopCount =
      firestarter::payload::PayloadSettings::getRAMLoopCount(Sequence, Settings.linesPerThread(), RamSize);

  asmjit::CodeHolder Code;
  Code.init(asmjit::Environment::host());

  asmjit::x86::Builder Cb(&Code);
  Cb.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateAssembler);
  Cb.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateIntermediate);

  const auto PointerReg = asmjit::x86::rax;
  const auto L1Addr = asmjit::x86::rbx;
  const auto L2Addr = asmjit::x86::rcx;
  const auto L3Addr = asmjit::x86::rdx;
  const auto RamAddr = asmjit::x86::rdi;
  const auto L2CountReg = asmjit::x86::r8;
  const auto L3CountReg = asmjit::x86::r9;
  const auto RamCountReg = asmjit::x86::r10;
  const auto TempReg = asmjit::x86::r11;
  const auto TempReg2 = asmjit::x86::rbp;
  const auto OffsetReg = asmjit::x86::r12;
  const auto AddrHighReg = asmjit::x86::r13;
  const auto IterReg = asmjit::x86::r14;
  constexpr const auto MovRegs = 0;
  const auto AddRegs = 14;
  const auto TransRegs = 2;

  asmjit::FuncDetail Func;
  Func.init(asmjit::FuncSignature::build<uint64_t, double*, volatile LoadThreadWorkType*, uint64_t>(
                asmjit::CallConvId::kCDecl),
            Code.environment());

  asmjit::FuncFrame Frame;
  Frame.init(Func);

  // make xmm registers dirty
  for (auto I = 0U; I < 16U; I++) {
    Frame.addDirtyRegs(Xmm(I));
  }
  // make mmx registers dirty
  for (auto I = 0U; I < 8U; I++) {
    Frame.addDirtyRegs(Mm(I));
  }
  // make all other used registers dirty except RAX
  Frame.addDirtyRegs(L1Addr, L2Addr, L3Addr, RamAddr, L2CountReg, L3CountReg, RamCountReg, TempReg, TempReg2, OffsetReg,
                     AddrHighReg, IterReg);

  asmjit::FuncArgsAssignment Args(&Func);
  Args.assignAll(PointerReg, AddrHighReg, IterReg);
  Args.updateFuncFrame(Frame);
  Frame.finalize();

  Cb.emitProlog(Frame);
  Cb.emitArgsAssignment(Frame, Args);

  // stop right away if low load is selected
  auto FunctionExit = Cb.newLabel();

  Cb.mov(TempReg, ptr_64(AddrHighReg));
  Cb.test(TempReg, TempReg);
  Cb.jz(FunctionExit);

  Cb.mov(OffsetReg,
         Imm(64)); // increment after each cache/memory access

  // Initialize SSE-Registers for Addition
  const auto AddStart = 0;
  const auto AddEnd = AddRegs - 1;
  const auto TransStart = AddRegs;
  const auto TransEnd = AddRegs + TransRegs - 1;
  if (AddRegs > 0) {
    for (auto I = AddStart; I <= AddEnd; I++) {
      Cb.movapd(Xmm(I), xmmword_ptr(PointerReg, 32 * I));
    }
  }

  // Initialize MMX-Registers for shift operations
  const auto MovStart = 0;
  const auto MovEnd = MovRegs - 1;
  if (MovRegs > 0) {
    Cb.mov(TempReg, Imm(0x5555555555555555));
    Cb.movq(Mm(MovStart), TempReg);
    for (auto I = MovStart + 1; I <= MovEnd; I++) {
      Cb.movq(Mm(I), Mm(MovStart));
    }
  }

  // Initialize SSE-Registers for Transfer-Operations
  if (TransRegs > 0) {
    if (TransStart % 2 == 0) {
      Cb.mov(TempReg, Imm(0x0F0F0F0F0F0F0F0F));
    } else {
      Cb.mov(TempReg, Imm(0xF0F0F0F0F0F0F0F0));
    }
    Cb.pinsrq(Xmm(TransStart), TempReg, Imm(0));
    Cb.pinsrq(Xmm(TransStart), TempReg, Imm(1));
    for (auto I = TransStart + 1; I <= TransEnd; I++) {
      if (I % 2 == 0) {
        Cb.shr(TempReg, Imm(4));
      } else {
        Cb.shl(TempReg, Imm(4));
      }
      Cb.pinsrq(Xmm(I), TempReg, Imm(0));
      Cb.pinsrq(Xmm(I), TempReg, Imm(1));
    }
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

  auto MovqDest = MovStart;
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

  for (auto Count = 0U; Count < Repetitions; Count++) {
    for (const auto& Item : Sequence) {
      Cb.bind(Cb.newAnonymousLabel(Item.c_str()));
      if (Item == "REG") {
        Cb.addpd(Xmm(AddDest), Xmm(AddStart + ((AddDest - AddStart + AddRegs + 1) % AddRegs)));
        Cb.movdqa(Xmm(MovDest), Xmm(MovSrc));
      } else if (Item == "L1_L") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L1Addr, 32));
        L1Increment();
      } else if (Item == "L1_S") {
        Cb.addpd(Xmm(AddDest), Xmm(AddStart + ((AddDest - AddStart + AddRegs - 1) % AddRegs)));
        Cb.movapd(xmmword_ptr(L1Addr, 32), Xmm(AddDest));
        L1Increment();
        Stats.Instructions++;
      } else if (Item == "L1_LS") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L1Addr, 32));
        Cb.movapd(xmmword_ptr(L1Addr, 64), Xmm(AddDest));
        L1Increment();
        Stats.Instructions++;
      } else if (Item == "L2_L") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L2Addr, 64));
        L2Increment();
      } else if (Item == "L2_S") {
        Cb.addpd(Xmm(AddDest), Xmm(AddStart + ((AddDest - AddStart + AddRegs - 1) % AddRegs)));
        Cb.movapd(xmmword_ptr(L2Addr, 64), Xmm(AddDest));
        L2Increment();
        Stats.Instructions++;
      } else if (Item == "L2_LS") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L2Addr, 64));
        Cb.movapd(xmmword_ptr(L2Addr, 96), Xmm(AddDest));
        L2Increment();
        Stats.Instructions++;
      } else if (Item == "L3_L") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L3Addr, 64));
        L3Increment();
      } else if (Item == "L3_S") {
        Cb.addpd(Xmm(AddDest), Xmm(AddStart + ((AddDest - AddStart + AddRegs - 1) % AddRegs)));
        Cb.movapd(xmmword_ptr(L3Addr, 96), Xmm(AddDest));
        L3Increment();
        Stats.Instructions++;
      } else if (Item == "L3_LS") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L3Addr, 64));
        Cb.movapd(xmmword_ptr(L3Addr, 96), Xmm(AddDest));
        L3Increment();
        Stats.Instructions++;
      } else if (Item == "L3_P") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L1Addr, 32));
        Cb.prefetcht0(ptr(L3Addr));
        L3Increment();
        Stats.Instructions++;
      } else if (Item == "RAM_L") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(RamAddr, 64));
        RamIncrement();
      } else if (Item == "RAM_S") {
        Cb.addpd(Xmm(AddDest), Xmm(AddStart + ((AddDest - AddStart + AddRegs - 1) % AddRegs)));
        Cb.movapd(xmmword_ptr(RamAddr, 64), Xmm(AddDest));
        RamIncrement();
        Stats.Instructions++;
      } else if (Item == "RAM_LS") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L3Addr, 64));
        Cb.movapd(xmmword_ptr(RamAddr, 64), Xmm(AddDest));
        RamIncrement();
        Stats.Instructions++;
      } else if (Item == "RAM_P") {
        Cb.addpd(Xmm(AddDest), xmmword_ptr(L1Addr, 32));
        Cb.prefetcht2(ptr(RamAddr));
        RamIncrement();
        Stats.Instructions++;
      } else {
        workerLog::error() << "Instruction group " << Item << " not found in " << name() << ".";
      }

      if constexpr (MovRegs > 0) {
        Stats.Instructions++;
        Cb.movq(Mm(MovStart + ((MovqDest - MovStart + MovRegs - 1) % MovRegs)), Mm(MovqDest));
      }

      AddDest++;
      if (AddDest > AddEnd) {
        // DO NOT REMOVE the + 1. It serves for the good of ymm0. If it was to
        // be overriden, the values in the other registers would rise up to inf.
        AddDest = AddStart + 1;
      }
      MovDest++;
      if (MovDest > TransEnd) {
        MovDest = TransStart;
      }
      MovSrc++;
      if (MovSrc > TransEnd) {
        MovSrc = TransStart;
      }
      if (MovRegs > 0) {
        MovqDest++;
        if (MovqDest > MovEnd) {
          MovqDest = MovStart;
        }
      }
    }
  }

  if (firestarter::payload::PayloadSettings::getRAMSequenceCount(Sequence) > 0) {
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
  if (firestarter::payload::PayloadSettings::getL2SequenceCount(Sequence) > 0) {
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
  if (firestarter::payload::PayloadSettings::getL3SequenceCount(Sequence) > 0) {
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
  Cb.inc(IterReg); // increment iteration counter
  Cb.mov(L1Addr, PointerReg);

  if (DumpRegisters) {
    emitDumpRegisterCode<Xmm>(Cb, PointerReg, xmmword_ptr);
  }

  if (ErrorDetection) {
    emitErrorDetectionCode<decltype(IterReg), Xmm>(Cb, IterReg, AddrHighReg, PointerReg, TempReg, TempReg2);
  }

  Cb.test(ptr_64(AddrHighReg), Imm(LoadThreadWorkType::LoadHigh));
  Cb.jnz(Loop);

  Cb.bind(FunctionExit);

  Cb.mov(asmjit::x86::rax, IterReg); // restore iteration counter

  Cb.emitEpilog(Frame);

  Cb.finalize();

  if (PrintAssembler) {
    printAssembler(Cb);
  }

  auto CompiledPayloadPtr = CompiledX86Payload::create<SSE2Payload>(Stats, Code);

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

void SSE2Payload::init(double* MemoryAddr, uint64_t BufferSize) const {
  X86Payload::initMemory(MemoryAddr, BufferSize, 1.654738925401e-10, 1.654738925401e-15);
}

} // namespace firestarter::x86::payload