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

#pragma once

#include "firestarter/Constants.hpp"          // IWYU pragma: keep
#include "firestarter/DumpRegisterStruct.hpp" // IWYU pragma: keep
#include "firestarter/LoadWorkerMemory.hpp"
#include "firestarter/Logging/Log.hpp" // IWYU pragma: keep
#include "firestarter/Payload/Payload.hpp"
#include "firestarter/X86/X86ProcessorInformation.hpp"

#include <asmjit/x86.h>
#include <cassert>
#include <cstdint>
#include <map> // IWYU pragma: keep
#include <type_traits>
#include <utility>

constexpr const auto InitBlocksize = 1024;

/// This abstract class models a payload that can be compiled with settings and executed for X86 CPUs.
namespace firestarter::x86::payload {

class X86Payload : public firestarter::payload::Payload {
private:
  /// This list contains the features (cpu extenstions) that are requied to execute the payload.
  std::list<asmjit::CpuFeatures::X86::Id> FeatureRequests;

  /// The mapping from instructions to the number of flops per instruction. This map is required to have an entry for
  /// every instruction.
  std::map<std::string, unsigned> InstructionFlops;

  /// The mapping from instructions to the size of main memory accesses for this instuction. This map is not required to
  /// contain all instructions.
  std::map<std::string, unsigned> InstructionMemory;

public:
  /// Abstract constructor for a payload on X86 CPUs.
  /// \arg FeatureRequests This list with features (cpu extenstions) that are requied to execute the payload.
  /// \arg Name The name of this payload. It is usally named by the CPU extension this payload uses e.g., SSE2 or FMA.
  /// \arg RegisterSize The size of the SIMD registers in units of doubles (8B).
  /// \arg RegisterCount The number of SIMD registers used by the payload.
  /// \arg InstructionFlops The mapping from instructions to the number of flops per instruction. This map is required
  /// to have an entry for every instruction.
  /// \arg InstructionMemory The mapping from instructions to the size of main memory accesses for this instuction. This
  /// map is not required to contain all instructions.
  X86Payload(std::initializer_list<asmjit::CpuFeatures::X86::Id> FeatureRequests, std::string Name,
             unsigned RegisterSize, unsigned RegisterCount, std::map<std::string, unsigned>&& InstructionFlops,
             std::map<std::string, unsigned>&& InstructionMemory) noexcept
      : Payload(std::move(Name), RegisterSize, RegisterCount)
      , FeatureRequests(FeatureRequests)
      , InstructionFlops(std::move(InstructionFlops))
      , InstructionMemory(std::move(InstructionMemory)) {}

private:
  /// Check if this payload is available on the current system. This is equivalent to checking if the supplied Topology
  /// contains all features that are in FeatureRequests.
  /// \arg Topology The CPUTopology that is used to check agains if this payload is supported.
  /// \returns true if the payload is supported on the given CPUTopology.
  [[nodiscard]] auto isAvailable(const ProcessorInformation& Topology) const -> bool final {
    const auto* FinalTopology = dynamic_cast<const X86ProcessorInformation*>(&Topology);
    assert(FinalTopology && "isAvailable not called with const X86CPUTopology*");

    bool Available = true;

    for (auto const& Feature : FeatureRequests) {
      Available &= FinalTopology->featuresAsmjit().has(Feature);
    }

    return Available;
  };

protected:
  /// Print the generated assembler Code of asmjit
  /// \arg Builder The builder that contains the assembler code.
  static void printAssembler(asmjit::BaseBuilder& Builder) {
    asmjit::String Sb;
    asmjit::FormatOptions FormatOptions{};

    asmjit::Formatter::formatNodeList(Sb, FormatOptions, &Builder);
    log::info() << Sb.data();
  }

  /// Emit the code to dump the xmm, ymm or zmm registers into memory for the dump registers feature.
  /// \tparam Vec the type of the vector register used.
  /// \arg Cb The asmjit code builder that is used to emit the assembler code.
  /// \arg PointerReg the register containing the pointer into memory in LoadWorkerMemory that is used in the high-load
  /// routine.
  /// \arg VecPtr The function that is used to create a ptr to the vector register
  template <class Vec>
  void emitDumpRegisterCode(asmjit::x86::Builder& Cb, const asmjit::x86::Gpq& PointerReg,
                            asmjit::x86::Mem (*VecPtr)(const asmjit::x86::Gp&, int32_t)) const {
    constexpr const auto DumpRegisterStructRegisterValuesTopOffset =
        -static_cast<int32_t>(LoadWorkerMemory::getMemoryOffset()) +
        static_cast<int32_t>(offsetof(LoadWorkerMemory, ExtraVars.Drs.Padding));
    constexpr const auto DumpRegisterStructDumpVariableOffset =
        -static_cast<int32_t>(LoadWorkerMemory::getMemoryOffset()) +
        static_cast<int32_t>(offsetof(LoadWorkerMemory, ExtraVars.Drs.DumpVar));

    auto SkipRegistersDump = Cb.newLabel();

    Cb.test(ptr_64(PointerReg, DumpRegisterStructDumpVariableOffset), asmjit::Imm(firestarter::DumpVariable::Wait));
    Cb.jnz(SkipRegistersDump);

    // dump all the vector registers register
    for (unsigned I = 0; I < registerCount(); I++) {
      Cb.vmovapd(VecPtr(PointerReg,
                        DumpRegisterStructRegisterValuesTopOffset - static_cast<int32_t>(registerSize() * 8 * (I + 1))),
                 Vec(I));
    }

    // set read flag
    Cb.mov(ptr_64(PointerReg, DumpRegisterStructDumpVariableOffset), asmjit::Imm(firestarter::DumpVariable::Wait));

    Cb.bind(SkipRegistersDump);
  }

  /// Emit the code to detect errors between this and two other threads that execute the same payload concurrently. We
  /// backup the registers in Mm2...Mm7. We will check every 0x3fff iterations. If the check did not succeed we write
  /// the LoadThreadWorkType::LoadStop flag in the AddrHighReg and therefore abort as soon as we pass the check in the
  /// high-load routine.
  /// \tparam MaybeConstIterRegT The type of the iteration register. If this is Mm, we assume that Mm0 is used by the
  /// payload and the other Mm1...Mm7 are free to use. If they are free we will use them to backup rax, rbx, rcx, rdx,
  /// r8 and r9. Otherwise we push them on the stack.
  /// \tparam MaybeConstVectorRegT This is the type of the vector register. It can be either Xmm, Ymm or Zmm. In case of
  /// Xmm we backup xmm0 on the stack, in case of Ymm we backup ymm0 im Mm4...Mm7 and in case of Zmm we use zmm31 for
  /// the backup. This register may not be used in the payload.
  /// \arg Cb The asmjit code builder that is used to emit the assembler code.
  /// \arg IterReg The register that holds the iteration counter of the high-load loop.
  /// \arg AddrHighReg The register contains the pointer to the memory address where the LoadThreadWorkType is saved.
  /// \arg PointerReg The register contains the pointer into memory in LoadWorkerMemory that is used in the high-load
  /// routine.
  /// \arg TempReg The first register that can be used to store temporary values.
  /// \arg TempReg2 The second register that can be used to store temporary values.
  template <class MaybeConstIterRegT, class MaybeConstVectorRegT>
  void emitErrorDetectionCode(asmjit::x86::Builder& Cb, MaybeConstIterRegT& IterReg,
                              const asmjit::x86::Gpq& AddrHighReg, const asmjit::x86::Gpq& PointerReg,
                              const asmjit::x86::Gpq& TempReg, const asmjit::x86::Gpq& TempReg2) const {
    using IterRegT = std::remove_const_t<MaybeConstIterRegT>;
    using VectorRegT = std::remove_const_t<MaybeConstVectorRegT>;

    // we don't want anything to break... so we use asserts for everything that
    // could break it
    static_assert(std::is_base_of_v<asmjit::x86::Vec, VectorRegT>, "VectorReg must be of asmjit::asmjit::x86::Vec");
    static_assert(std::is_same_v<asmjit::x86::Xmm, VectorRegT> || std::is_same_v<asmjit::x86::Ymm, VectorRegT> ||
                      std::is_same_v<asmjit::x86::Zmm, VectorRegT>,
                  "VectorReg ist not of any supported type");
    static_assert(std::is_same_v<asmjit::x86::Mm, IterRegT> || std::is_same_v<asmjit::x86::Gpq, IterRegT>,
                  "IterReg is not of any supported type");

    if constexpr (std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      assert(IterReg == asmjit::x86::mm0 && "iter_reg must be mm0");
    }

    assert(IterReg != TempReg && "iter_reg must be != temp_reg");
    assert(TempReg != TempReg2 && "temp_reg must be != temp_reg2");
    assert(TempReg != AddrHighReg && "temp_reg must be != addrHigh_reg");
    assert(TempReg != PointerReg && "temp_reg must be != pointer_reg");

    assert(IterReg != asmjit::x86::r8 && "iter_reg must be != r8");
    assert(IterReg != asmjit::x86::r9 && "iter_reg must be != r9");
    assert(IterReg != asmjit::x86::rax && "iter_reg must be != rax");
    assert(IterReg != asmjit::x86::rbx && "iter_reg must be != rbx");
    assert(IterReg != asmjit::x86::rcx && "iter_reg must be != rcx");
    assert(IterReg != asmjit::x86::rdx && "iter_reg must be != rdx");

    assert(TempReg != asmjit::x86::r8 && "temp_reg must be != r8");
    assert(TempReg != asmjit::x86::r9 && "temp_reg must be != r9");
    assert(TempReg != asmjit::x86::rax && "temp_reg must be != rax");
    assert(TempReg != asmjit::x86::rbx && "temp_reg must be != rbx");
    assert(TempReg != asmjit::x86::rcx && "temp_reg must be != rcx");
    assert(TempReg != asmjit::x86::rdx && "temp_reg must be != rdx");

    assert(TempReg2 != asmjit::x86::r8 && "temp_reg2 must be != r8");
    assert(TempReg2 != asmjit::x86::r9 && "temp_reg2 must be != r9");
    assert(TempReg2 != asmjit::x86::rax && "temp_reg2 must be != rax");
    assert(TempReg2 != asmjit::x86::rbx && "temp_reg2 must be != rbx");
    assert(TempReg2 != asmjit::x86::rcx && "temp_reg2 must be != rcx");
    assert(TempReg2 != asmjit::x86::rdx && "temp_reg2 must be != rdx");

    assert(AddrHighReg != asmjit::x86::r8 && "addrHigh_reg must be != r8");
    assert(AddrHighReg != asmjit::x86::r9 && "addrHigh_reg must be != r9");
    assert(AddrHighReg != asmjit::x86::rax && "addrHigh_reg must be != rax");
    assert(AddrHighReg != asmjit::x86::rbx && "addrHigh_reg must be != rbx");
    assert(AddrHighReg != asmjit::x86::rcx && "addrHigh_reg must be != rcx");
    assert(AddrHighReg != asmjit::x86::rdx && "addrHigh_reg must be != rdx");

    auto SkipErrorDetection = Cb.newLabel();

    if constexpr (std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      Cb.movq(TempReg, IterReg);
    } else {
      Cb.mov(TempReg, IterReg);
    }
    // round about 50-100 Hz
    // more or less, but this isn't really that relevant
    Cb.and_(TempReg, asmjit::Imm(0x3fff));
    Cb.test(TempReg, TempReg);
    Cb.jnz(SkipErrorDetection);

    Cb.mov(TempReg, asmjit::Imm(0xffffffff));

    auto RegisterCount = registerCount();

    // Create a backup of VectorReg(0)
    if constexpr (std::is_same_v<asmjit::x86::Xmm, VectorRegT>) {
      Cb.movq(TempReg2, asmjit::x86::xmm0);
      Cb.push(TempReg2);
      Cb.crc32(TempReg, TempReg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(TempReg2, asmjit::x86::xmm0);
      Cb.push(TempReg2);
      Cb.crc32(TempReg, TempReg2);

    } else if constexpr (std::is_same_v<asmjit::x86::Ymm, VectorRegT> && std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      Cb.movq(TempReg2, asmjit::x86::xmm0);
      Cb.movq(asmjit::x86::Mm(7), TempReg2);
      Cb.crc32(TempReg, TempReg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(TempReg2, asmjit::x86::xmm0);
      Cb.movq(asmjit::x86::Mm(6), TempReg2);
      Cb.crc32(TempReg, TempReg2);

      Cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

      Cb.movq(TempReg2, asmjit::x86::xmm0);
      Cb.movq(asmjit::x86::Mm(5), TempReg2);
      Cb.crc32(TempReg, TempReg2);
      Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.movq(TempReg2, asmjit::x86::xmm0);
      Cb.movq(asmjit::x86::Mm(4), TempReg2);
      Cb.crc32(TempReg, TempReg2);
    } else if constexpr (std::is_same_v<asmjit::x86::Zmm, VectorRegT> && std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      // We use vector registers zmm31 for our backup
      Cb.vmovapd(asmjit::x86::zmm31, asmjit::x86::zmm0);
      RegisterCount--;
    }

    // Calculate the hash of the remaining VectorReg
    // use VectorReg(0) as a temporary place to unpack values
    for (unsigned I = 1; I < RegisterCount; I++) {
      if constexpr (std::is_same_v<asmjit::x86::Xmm, VectorRegT>) {
        Cb.vmovapd(asmjit::x86::xmm0, asmjit::x86::Xmm(I));

        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
        Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
      } else if constexpr (std::is_same_v<asmjit::x86::Ymm, VectorRegT>) {
        Cb.vmovapd(asmjit::x86::ymm0, asmjit::x86::Ymm(I));

        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
        Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);

        Cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
        Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
      } else if constexpr (std::is_same_v<asmjit::x86::Zmm, VectorRegT>) {
        Cb.vmovapd(asmjit::x86::ymm0, asmjit::x86::Ymm(I));

        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
        Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);

        Cb.vextractf128(asmjit::x86::xmm0, asmjit::x86::ymm0, asmjit::Imm(1));

        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
        Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);

        Cb.vextractf32x4(asmjit::x86::xmm0, asmjit::x86::Zmm(I), asmjit::Imm(2));

        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
        Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);

        Cb.vextractf32x4(asmjit::x86::xmm0, asmjit::x86::Zmm(I), asmjit::Imm(3));

        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
        Cb.movhlps(asmjit::x86::xmm0, asmjit::x86::xmm0);
        Cb.movq(TempReg2, asmjit::x86::xmm0);
        Cb.crc32(TempReg, TempReg2);
      }
    }

    // Restore VectorReg(0) from backup
    if constexpr (std::is_same_v<asmjit::x86::Xmm, VectorRegT>) {
      Cb.pop(TempReg2);
      Cb.movq(asmjit::x86::xmm0, TempReg2);
      Cb.movlhps(asmjit::x86::xmm0, asmjit::x86::xmm0);
      Cb.pop(TempReg2);
      Cb.pinsrw(asmjit::x86::xmm0, TempReg2.r32(), asmjit::Imm(0));
      Cb.shr(TempReg2, asmjit::Imm(32));
      Cb.movd(TempReg2.r32(), asmjit::x86::Mm(7));
      Cb.pinsrw(asmjit::x86::xmm0, TempReg2.r32(), asmjit::Imm(1));
    } else if constexpr (std::is_same_v<asmjit::x86::Ymm, VectorRegT> && std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      Cb.movq(TempReg2, asmjit::x86::Mm(5));
      Cb.movq(asmjit::x86::xmm0, TempReg2);
      Cb.movq(TempReg2, asmjit::x86::Mm(4));
      Cb.pinsrq(asmjit::x86::xmm0, TempReg2, asmjit::Imm(1));

      Cb.vinsertf128(asmjit::x86::ymm0, asmjit::x86::ymm0, asmjit::x86::xmm0, asmjit::Imm(1));

      Cb.movq(TempReg2, asmjit::x86::Mm(7));
      Cb.movq(asmjit::x86::xmm0, TempReg2);
      Cb.movq(TempReg2, asmjit::x86::Mm(6));
      Cb.pinsrq(asmjit::x86::xmm0, TempReg2, asmjit::Imm(1));
    } else if constexpr (std::is_same_v<asmjit::x86::Zmm, VectorRegT> && std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      // We use vector registers zmm31 for our backup
      Cb.vmovapd(asmjit::x86::zmm0, asmjit::x86::zmm31);
    }

    // before starting the communication, backup r8, r9, rax, rbx, rcx and rdx
    if constexpr (std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      Cb.movq(asmjit::x86::Mm(7), asmjit::x86::rax);
      Cb.movq(asmjit::x86::Mm(6), asmjit::x86::rbx);
      Cb.movq(asmjit::x86::Mm(5), asmjit::x86::rcx);
      Cb.movq(asmjit::x86::Mm(4), asmjit::x86::rdx);
      Cb.movq(asmjit::x86::Mm(3), asmjit::x86::r8);
      Cb.movq(asmjit::x86::Mm(2), asmjit::x86::r9);
    } else {
      Cb.push(asmjit::x86::rax);
      Cb.push(asmjit::x86::rbx);
      Cb.push(asmjit::x86::rcx);
      Cb.push(asmjit::x86::rdx);
      Cb.push(asmjit::x86::r8);
      Cb.push(asmjit::x86::r9);
    }

    // do the actual communication
    // temp_reg contains our hash

    // save the pointer_reg. it might be any of r8, r9, rax, rbx, rcx or rdx
    Cb.mov(TempReg2, PointerReg);

    // Don't touch me!
    // This sychronization and communication works even if the threads run at
    // different (changing) speed, with just one "lock cmpxchg16b" Brought to you
    // by a few hours of headache for two people.
    auto Communication = [&](const int32_t ErrorDetetectionStructOffset) {
      const auto CommunicationOffset =
          ErrorDetetectionStructOffset + static_cast<int32_t>(offsetof(ErrorDetectionStruct::OneSide, Communication));
      const auto Local0Offset =
          ErrorDetetectionStructOffset + static_cast<int32_t>(offsetof(ErrorDetectionStruct::OneSide, Locals[0]));
      const auto Local1Offset =
          ErrorDetetectionStructOffset + static_cast<int32_t>(offsetof(ErrorDetectionStruct::OneSide, Locals[1]));
      const auto Local2Offset =
          ErrorDetetectionStructOffset + static_cast<int32_t>(offsetof(ErrorDetectionStruct::OneSide, Locals[2]));
      const auto Local3Offset =
          ErrorDetetectionStructOffset + static_cast<int32_t>(offsetof(ErrorDetectionStruct::OneSide, Locals[3]));
      const auto ErrorOffset =
          ErrorDetetectionStructOffset + static_cast<int32_t>(offsetof(ErrorDetectionStruct::OneSide, Error));

      // communication
      Cb.mov(asmjit::x86::r8, asmjit::x86::ptr_64(TempReg2, CommunicationOffset));

      // temp data
      Cb.mov(asmjit::x86::r9, TempReg2);

      Cb.mov(asmjit::x86::rdx, asmjit::x86::ptr_64(asmjit::x86::r9, Local0Offset));
      Cb.mov(asmjit::x86::rax, asmjit::x86::ptr_64(asmjit::x86::r9, Local1Offset));

      auto L0 = Cb.newLabel();
      Cb.bind(L0);

      // Atomically ompare the data in the communicaton with the local data.
      Cb.lock();
      Cb.cmpxchg16b(asmjit::x86::ptr(asmjit::x86::r8));

      auto L1 = Cb.newLabel();
      Cb.jnz(L1);

      // Communication had the same data as saved in locals 0 and 1. rcx, rbx saved in communication.
      // Save written data rcx, rbx in locals 0 and 1.
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local0Offset), asmjit::x86::rcx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local1Offset), asmjit::x86::rbx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset), asmjit::Imm(0));
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset), asmjit::Imm(0));

      Cb.mov(asmjit::x86::rax, asmjit::Imm(2));

      auto L6 = Cb.newLabel();
      Cb.jmp(L6);

      Cb.bind(L1);

      // Communication had differnt data as saved in locals 0 and 1. rdx, rax contains the data in communication.
      // Compare the iteration counter of this and the other thread
      Cb.cmp(asmjit::x86::rcx, asmjit::x86::rdx);

      auto L2 = Cb.newLabel();
      Cb.jle(L2);

      // The current iteration counter is bigger than the counter of the other thread.
      // Save the current counter and hash into our local storage.
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local0Offset), asmjit::x86::rcx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local1Offset), asmjit::x86::rbx);

      // Repeat the lock cmpxchg16b routine until the other thread catches up.
      Cb.jmp(L0);

      Cb.bind(L2);

      // The current iteration counter is smaller equal than the iteration counter of the other thread.

      auto L3 = Cb.newLabel();

      // Check if the read value from the other thread is saved locally.
      Cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset), asmjit::Imm(0));
      Cb.jne(L3);
      Cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset), asmjit::Imm(0));
      Cb.jne(L3);

      // Save the last read value from the other thread into the local storage.
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset), asmjit::x86::rdx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset), asmjit::x86::rax);

      Cb.bind(L3);

      // Check if the id of the two threads are equal
      Cb.cmp(asmjit::x86::rcx, asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset));
      Cb.mov(asmjit::x86::rax, asmjit::Imm(4));
      // If the iteration counter of this thread is smaller, skip this check. The other thread will wait for this one.
      Cb.jne(L6);

      // Compare the hashes and write teh result
      Cb.cmp(asmjit::x86::rbx, asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset));
      auto L4 = Cb.newLabel();
      Cb.jne(L4);

      // Hash check succeeded.
      Cb.mov(asmjit::x86::rax, asmjit::Imm(0));

      auto L5 = Cb.newLabel();
      Cb.jmp(L5);

      Cb.bind(L4);

      // Hash check failed
      Cb.mov(asmjit::x86::rax, asmjit::Imm(1));

      Cb.bind(L5);

      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset), asmjit::Imm(0));
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset), asmjit::Imm(0));

      Cb.bind(L6);

      // if check failed
      Cb.cmp(asmjit::x86::rax, asmjit::Imm(1));
      auto L7 = Cb.newLabel();
      Cb.jne(L7);

      // write the error flag
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, ErrorOffset), asmjit::Imm(1));

      // stop the execution after some time
      Cb.mov(asmjit::x86::ptr_64(AddrHighReg), asmjit::Imm(LoadThreadWorkType::LoadStop));
      Cb.mfence();

      Cb.bind(L7);

      auto L9 = Cb.newLabel();
      Cb.jmp(L9);
    };

    constexpr const auto ErrorDetectionStructCommunicationLeftOffset =
        -static_cast<int32_t>(LoadWorkerMemory::getMemoryOffset()) +
        static_cast<int32_t>(offsetof(LoadWorkerMemory, ExtraVars.Eds.Left.Communication));
    constexpr const auto ErrorDetectionStructCommunicationRightOffset =
        -static_cast<int32_t>(LoadWorkerMemory::getMemoryOffset()) +
        static_cast<int32_t>(offsetof(LoadWorkerMemory, ExtraVars.Eds.Right.Communication));

    // left communication
    // move hash
    Cb.mov(asmjit::x86::rbx, TempReg);
    // move iterations counter
    if constexpr (std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      Cb.movq(asmjit::x86::rcx, IterReg);
    } else {
      Cb.mov(asmjit::x86::rcx, IterReg);
    }

    Communication(ErrorDetectionStructCommunicationLeftOffset);

    // right communication
    // move hash
    Cb.mov(asmjit::x86::rbx, TempReg);
    // move iterations counter
    if constexpr (std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      Cb.movq(asmjit::x86::rcx, IterReg);
    } else {
      Cb.mov(asmjit::x86::rcx, IterReg);
    }

    Communication(ErrorDetectionStructCommunicationRightOffset);

    // restore r8, r9, rax, rbx, rcx and rdx
    if constexpr (std::is_same_v<asmjit::x86::Mm, IterRegT>) {
      Cb.movq(asmjit::x86::rax, asmjit::x86::Mm(7));
      Cb.movq(asmjit::x86::rbx, asmjit::x86::Mm(6));
      Cb.movq(asmjit::x86::rcx, asmjit::x86::Mm(5));
      Cb.movq(asmjit::x86::rdx, asmjit::x86::Mm(4));
      Cb.movq(asmjit::x86::r8, asmjit::x86::Mm(3));
      Cb.movq(asmjit::x86::r9, asmjit::x86::Mm(2));
    } else {
      Cb.pop(asmjit::x86::r9);
      Cb.pop(asmjit::x86::r8);
      Cb.pop(asmjit::x86::rdx);
      Cb.pop(asmjit::x86::rcx);
      Cb.pop(asmjit::x86::rbx);
      Cb.pop(asmjit::x86::rax);
    }

    Cb.bind(SkipErrorDetection);
  }

  static void initMemory(double* MemoryAddr, uint64_t BufferSize, double FirstValue, double LastValue);

  /// Function to produce a low load on the cpu.
  /// \arg LoadVar The variable that controls the load. If this variable changes from LoadThreadWorkType::LowLoad to
  /// something else this function will return.
  /// \arg Period The period of the low/high load switching. This function will sleep 1% of the Period and check if the
  /// LoadVar changed.
  void lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) const final;

public:
  /// Get the available instruction items that are supported by this payload.
  /// \returns The available instruction items that are supported by this payload.
  [[nodiscard]] auto getAvailableInstructions() const -> std::list<std::string> final;

  /// Get the mapping from instructions to the number of flops per instruction. This map is required to have an entry
  /// for every instruction.
  [[nodiscard]] auto instructionFlops() const -> const auto& { return InstructionFlops; }

  /// Get the mapping from instructions to the size of main memory accesses for this instuction. This map is not
  /// required to contain all instructions.
  [[nodiscard]] auto instructionMemory() const -> const auto& { return InstructionMemory; }
};

} // namespace firestarter::x86::payload
