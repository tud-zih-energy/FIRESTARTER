/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
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

#include "firestarter/AArch64/Payload/AArch64DefaultPayload.hpp"
#include "firestarter/AArch64/Payload/CompiledAArch64Payload.hpp"
#include "firestarter/Constants.hpp"
#include "firestarter/Logging/Log.hpp"
#include "firestarter/Payload/CompiledPayload.hpp"
#include "firestarter/Payload/PayloadControlFlowDescription.hpp"
#include "firestarter/Payload/PayloadSettings.hpp"
#include "firestarter/Payload/PayloadStats.hpp"

#include <asmjit/a64.h>
#include <cstdint>
#include <iterator>
#include <vector>

namespace firestarter::aarch64::payload {

auto AArch64DefaultPayload::compilePayload(const firestarter::payload::PayloadSettings& Settings, bool DumpRegisters,
                                           bool ErrorDetection, bool PrintAssembler,
                                           firestarter::payload::HighLoadControlFlowDescription ControlFlow) const
    -> firestarter::payload::CompiledPayload::UniquePtr {
  using Imm = asmjit::Imm;
  using Gp = asmjit::a64::Gp;
  using VecV = asmjit::a64::VecV;
  using VecD = asmjit::a64::VecD;

  // Compute the sequence of instruction groups and the number of its repetitions
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

  asmjit::a64::Builder Cb(&Code);
  Cb.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateAssembler);
  Cb.addDiagnosticOptions(asmjit::DiagnosticOptions::kValidateIntermediate);

  const auto PointerReg = asmjit::a64::x0;
  const auto L1Addr = asmjit::a64::x1;
  const auto L2Addr = asmjit::a64::x2;
  const auto L3Addr = asmjit::a64::x3;
  const auto RamAddr = asmjit::a64::x4;
  const auto L2CountReg = asmjit::a64::x5;
  const auto L3CountReg = asmjit::a64::x6;
  const auto RamCountReg = asmjit::a64::x7;
  const auto TempReg = asmjit::a64::x8;
  const auto TempReg2 = asmjit::a64::x9;
  const auto OffsetReg = asmjit::a64::x10;
  const auto AddrHighReg = asmjit::a64::x11;
  const auto IterReg = asmjit::a64::x12;
  const auto RemainingIterationsReg = asmjit::a64::x13;
  const auto AddRegs = 14;
  const auto TransRegs = 2;

  asmjit::FuncDetail Func;
  Func.init(asmjit::FuncSignature::build<uint64_t, double*, volatile LoadThreadWorkType*, uint64_t>(
                asmjit::CallConvId::kCDecl),
            Code.environment());

  asmjit::FuncFrame Frame;
  Frame.init(Func);

  // make NEON registers dirty
  for (auto I = 0; I < 32; I++) {
    Frame.addDirtyRegs(VecV(I));
  }
  // make all other used registers dirty except x0
  Frame.addDirtyRegs(L1Addr, L2Addr, L3Addr, RamAddr, L2CountReg, L3CountReg, RamCountReg, TempReg, TempReg2, OffsetReg,
                     AddrHighReg, IterReg, RemainingIterationsReg);

  asmjit::FuncArgsAssignment Args(&Func);
  Args.assignAll(PointerReg, AddrHighReg, RemainingIterationsReg);
  Args.updateFuncFrame(Frame);
  Frame.finalize();

  Cb.emitProlog(Frame);
  Cb.emitArgsAssignment(Frame, Args);

  // stop right away if low load is selected
  auto FunctionExit = Cb.newLabel();

  // Set initial iteration count to zero
  Cb.mov(IterReg, Imm(0));

  if (ControlFlow == firestarter::payload::HighLoadControlFlowDescription::kMaxIterationCount) {
    // return if remaining iteration count is zero
    Cb.tst(RemainingIterationsReg, RemainingIterationsReg);
    Cb.b_eq(FunctionExit);
  }

  // stop right away if low load is selected
  Cb.ldr(TempReg, asmjit::a64::ptr(AddrHighReg));
  Cb.cbz(TempReg, FunctionExit);

  Cb.mov(OffsetReg, Imm(64)); // increment after each cache/memory access

  // Initialize NEON registers for Addition
  const auto AddStart = 0;
  const auto AddEnd = AddRegs - 1;
  const auto TransStart = AddRegs;
  const auto TransEnd = AddRegs + TransRegs - 1;
  if (AddRegs > 0) {
    for (auto I = AddStart; I <= AddEnd; I++) {
      Cb.ldr(VecV(I).d2(), asmjit::a64::ptr(PointerReg, 32 * I));
    }
  }

  // Initialize NEON registers for Transfer-Operations
  if (TransRegs > 0) {
    if (TransStart % 2 == 0) {
      Cb.mov(TempReg, Imm(0x0F0F0F0F0F0F0F0F));
    } else {
      Cb.mov(TempReg, Imm(0xF0F0F0F0F0F0F0F0));
    }

    Cb.dup(VecV(TransStart).d2(), TempReg);

    for (auto I = TransStart + 1; I <= TransEnd; I++) {
      if (I % 2 == 0) {
        Cb.lsr(TempReg, TempReg, Imm(4));
      } else {
        Cb.lsl(TempReg, TempReg, Imm(4));
      }
      Cb.dup(VecV(I).d2(), TempReg);
    }
  }

  Cb.mov(L1Addr, PointerReg); // address for L1-buffer
  Cb.mov(L2Addr, PointerReg);
  Cb.add(L2Addr, L2Addr, Imm(L1Size)); // address for L2-buffer
  Cb.mov(L3Addr, PointerReg);
  Cb.add(L3Addr, L3Addr, Imm(L2Size)); // address for L3-buffer
  Cb.mov(RamAddr, PointerReg);
  Cb.add(RamAddr, RamAddr, Imm(L3Size)); // address for RAM-buffer
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

  auto AddDest = AddStart + 1;
  auto MovDst = TransStart;
  auto MovSrc = MovDst + 1;
  unsigned L1Offset = 0;

#define L1_INCREMENT()                                                                                                 \
  L1Offset += 64;                                                                                                      \
  if (L1Offset < L1Size * 0.5) {                                                                                       \
    Cb.add(L1Addr, L1Addr, OffsetReg);                                                                                 \
  } else {                                                                                                             \
    L1Offset = 0;                                                                                                      \
    Cb.mov(L1Addr, PointerReg);                                                                                        \
  }

#define L2_INCREMENT() Cb.add(L2Addr, L2Addr, OffsetReg)

#define L3_INCREMENT() Cb.add(L3Addr, L3Addr, OffsetReg)

#define RAM_INCREMENT() Cb.add(RamAddr, RamAddr, OffsetReg)

  for (unsigned Count = 0; Count < Repetitions; Count++) {
    for (const auto& Item : Sequence) {
      if (Item == "REG") {
        Cb.add(VecD(AddDest).d2(), VecD(AddDest).d2(),
               VecD(AddStart + (AddDest - AddStart + AddRegs + 1) % AddRegs).d2());
        AddDest++;
        if (AddDest > AddEnd) {
          AddDest = AddStart + 1;
        }
      } else if (Item == "L1_L") {
        Cb.ldr(VecV(MovDst).d2(), asmjit::a64::ptr(L1Addr, 32));
        MovDst++;
        L1_INCREMENT();
      } else if (Item == "L1_S") {
        Cb.str(VecV(MovDst).d2(), asmjit::a64::ptr(L1Addr, 32));
        MovDst++;
        L1_INCREMENT();
      } else if (Item == "L2_L") {
        Cb.ldr(VecV(MovDst).d2(), asmjit::a64::ptr(L2Addr, 64));
        MovDst++;
        L2_INCREMENT();
      } else if (Item == "L2_S") {
        Cb.str(VecV(MovDst).d2(), asmjit::a64::ptr(L2Addr, 64));
        MovDst++;
        L2_INCREMENT();
      } else if (Item == "L3_L") {
        Cb.ldr(VecV(MovDst).d2(), asmjit::a64::ptr(L3Addr, 64));
        MovDst++;
        L3_INCREMENT();
      } else if (Item == "L3_S") {
        Cb.str(VecV(MovDst).d2(), asmjit::a64::ptr(L3Addr, 64));
        MovDst++;
        L3_INCREMENT();
      } else if (Item == "RAM_L") {
        Cb.ldr(VecV(MovDst).d2(), asmjit::a64::ptr(RamAddr, 64));
        MovDst++;
        RAM_INCREMENT();
      } else if (Item == "RAM_S") {
        Cb.str(VecV(MovDst).d2(), asmjit::a64::ptr(RamAddr, 64));
        MovDst++;
        RAM_INCREMENT();
      } else {
        workerLog::error() << "Instruction group " << Item << " not found in " << this->name() << ".";
      }
      if (MovDst > TransEnd) {
        MovDst = TransStart;
      }
    }
  }

#undef L1_INCREMENT
#undef L2_INCREMENT
#undef L3_INCREMENT
#undef RAM_INCREMENT

  if (firestarter::payload::PayloadSettings::getRAMSequenceCount(Sequence) > 0) {
    // reset RAM counter
    auto NoRamReset = Cb.newLabel();

    Cb.sub(RamCountReg, RamCountReg, Imm(1));
    Cb.cbnz(RamCountReg, NoRamReset);
    Cb.mov(RamCountReg, Imm(RamLoopCount));
    Cb.mov(RamAddr, PointerReg);
    Cb.add(RamAddr, RamAddr, Imm(L3Size));
    Cb.bind(NoRamReset);
    Stats.Instructions += 2;
  }
  if (firestarter::payload::PayloadSettings::getL2SequenceCount(Sequence) > 0) {
    // reset L2-Cache counter
    auto NoL2Reset = Cb.newLabel();

    Cb.sub(L2CountReg, L2CountReg, Imm(1));
    Cb.cbnz(L2CountReg, NoL2Reset);
    Cb.mov(L2CountReg, Imm(L2LoopCount));
    Cb.mov(L2Addr, PointerReg);
    Cb.add(L2Addr, L2Addr, Imm(L1Size));
    Cb.bind(NoL2Reset);
    Stats.Instructions += 2;
  }
  if (firestarter::payload::PayloadSettings::getL3SequenceCount(Sequence) > 0) {
    // reset L3-Cache counter
    auto NoL3Reset = Cb.newLabel();

    Cb.sub(L3CountReg, L3CountReg, Imm(1));
    Cb.cbnz(L3CountReg, NoL3Reset);
    Cb.mov(L3CountReg, Imm(L3LoopCount));
    Cb.mov(L3Addr, PointerReg);
    Cb.add(L3Addr, L3Addr, Imm(L2Size));
    Cb.bind(NoL3Reset);
    Stats.Instructions += 2;
  }

  // increment iteration counter
  Cb.add(IterReg, IterReg, Imm(1));
  if (ControlFlow == firestarter::payload::HighLoadControlFlowDescription::kMaxIterationCount) {
    // decrement remaining iterations
    Cb.sub(RemainingIterationsReg, RemainingIterationsReg, Imm(1));
    // Return if the remaining instructions reached zero
    Cb.tst(RemainingIterationsReg, RemainingIterationsReg);
    Cb.b_eq(FunctionExit);
  }

  Cb.mov(L1Addr, PointerReg);

  if (DumpRegisters) {
    workerLog::error() << "Dump Registers n/a for AArch64";
  }

  if (ErrorDetection) {
    workerLog::error() << "Error Detection n/a for AArch64";
  }

  Cb.ldr(TempReg, asmjit::a64::ptr(AddrHighReg));
  Cb.cmp(TempReg, Imm(LoadThreadWorkType::LoadHigh));
  Cb.b_eq(Loop);

  Cb.bind(FunctionExit);

  Cb.mov(asmjit::a64::x0, IterReg); // restore iteration counter

  Cb.emitEpilog(Frame);

  Cb.finalize();

  if (PrintAssembler) {
    printAssembler(Cb);
  }

  auto CompiledPayloadPtr = CompiledAArch64Payload::create<AArch64DefaultPayload>(Stats, Code);

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

auto AArch64DefaultPayload::getAvailableInstructions() const -> std::list<std::string> {
  return AArch64Payload::getAvailableInstructions();
}

void AArch64DefaultPayload::init(double* MemoryAddr, uint64_t BufferSize) const {
  AArch64Payload::initMemory(MemoryAddr, BufferSize, 1.654738925401e-10, 1.654738925401e-15);
}

} // namespace firestarter::aarch64::payload
