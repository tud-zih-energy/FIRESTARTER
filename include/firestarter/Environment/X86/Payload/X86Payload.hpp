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

#include "../../../Constants.hpp"          // IWYU pragma: keep
#include "../../../DumpRegisterStruct.hpp" // IWYU pragma: keep
#include "../../../Logging/Log.hpp"        // IWYU pragma: keep
#include "../../Payload/Payload.hpp"
#include "../X86CPUTopology.hpp"
#include "firestarter/LoadWorkerMemory.hpp"
#include <asmjit/x86.h>
#include <cassert>
#include <cstdint>
#include <map> // IWYU pragma: keep
#include <type_traits>
#include <utility>

constexpr const auto InitBlocksize = 1024;

namespace firestarter::environment::x86::payload {

class X86Payload : public environment::payload::Payload {
private:
  // we can use this to check, if our platform support this payload
  std::list<asmjit::CpuFeatures::X86::Id> FeatureRequests;

protected:
  /// Emit the code to dump the xmm, ymm or zmm registers into memory for the dump registers feature.
  /// \arg Vec the type of the vector register used.
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

  // add MM regs to dirty regs
  // zmm31 is used for backup if VectorReg is of type asmjit::x86::Zmm
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

      Cb.lock();
      Cb.cmpxchg16b(asmjit::x86::ptr(asmjit::x86::r8));

      auto L1 = Cb.newLabel();
      Cb.jnz(L1);

      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local0Offset), asmjit::x86::rcx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local1Offset), asmjit::x86::rbx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset), asmjit::Imm(0));
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset), asmjit::Imm(0));

      Cb.mov(asmjit::x86::rax, asmjit::Imm(2));

      auto L6 = Cb.newLabel();
      Cb.jmp(L6);

      Cb.bind(L1);

      Cb.cmp(asmjit::x86::rcx, asmjit::x86::rdx);

      auto L2 = Cb.newLabel();
      Cb.jle(L2);

      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local0Offset), asmjit::x86::rcx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local1Offset), asmjit::x86::rbx);

      Cb.jmp(L0);

      Cb.bind(L2);

      auto L3 = Cb.newLabel();

      Cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset), asmjit::Imm(0));
      Cb.jne(L3);
      Cb.cmp(asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset), asmjit::Imm(0));
      Cb.jne(L3);

      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local2Offset), asmjit::x86::rdx);
      Cb.mov(asmjit::x86::ptr_64(asmjit::x86::r9, Local3Offset), asmjit::x86::rax);

      Cb.bind(L3);

      Cb.cmp(asmjit::x86::rcx, asmjit::x86::ptr_64(asmjit::x86::r9, 16));
      Cb.mov(asmjit::x86::rax, asmjit::Imm(4));
      Cb.jne(L6);

      Cb.cmp(asmjit::x86::rbx, asmjit::x86::ptr_64(asmjit::x86::r9, 24));
      auto L4 = Cb.newLabel();
      Cb.jne(L4);

      Cb.mov(asmjit::x86::rax, asmjit::Imm(0));

      auto L5 = Cb.newLabel();
      Cb.jmp(L5);

      Cb.bind(L4);

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

  // use cpuid and usleep as low load
  void lowLoadFunction(volatile LoadThreadWorkType& LoadVar, std::chrono::microseconds Period) const final;

public:
  X86Payload(std::initializer_list<asmjit::CpuFeatures::X86::Id> FeatureRequests, std::string Name,
             unsigned RegisterSize, unsigned RegisterCount) noexcept
      : Payload(std::move(Name), RegisterSize, RegisterCount)
      , FeatureRequests(FeatureRequests) {}

  [[nodiscard]] auto isAvailable(const X86CPUTopology& Topology) const -> bool { return isAvailable(&Topology); }

private:
  [[nodiscard]] auto isAvailable(const CPUTopology* Topology) const -> bool final {
    const auto* FinalTopology = dynamic_cast<const X86CPUTopology*>(Topology);
    assert(FinalTopology && "isAvailable not called with const X86CPUTopology*");

    bool Available = true;

    for (auto const& Feature : FeatureRequests) {
      Available &= FinalTopology->featuresAsmjit().has(Feature);
    }

    return Available;
  };
};

} // namespace firestarter::environment::x86::payload
