#include <firestarter/Environment/X86/Payload/FMAPayload.hpp>
#include <firestarter/Logging/Log.hpp>

#include <iterator>
#include <utility>

using namespace firestarter::environment::x86::payload;
using namespace asmjit;
using namespace asmjit::x86;

int FMAPayload::compilePayload(std::map<std::string, unsigned> proportion,
                               std::list<unsigned> dataCacheBufferSize,
                               unsigned ramBufferSize, unsigned thread,
                               unsigned numberOfLines) {
  CodeHolder code;
  code.init(this->rt.codeInfo());
  code.addEmitterOptions(BaseEmitter::kOptionStrictValidation);

  if (nullptr != this->loadFunction) {
    this->rt.release(&this->loadFunction);
  }

  Builder cb(&code);

  auto sequence = this->generateSequence(proportion);
  auto repetitions =
      this->getNumberOfSequenceRepetitions(sequence, numberOfLines / thread);

  auto dataCacheBufferSizeIterator = dataCacheBufferSize.begin();
  auto l1_size = *dataCacheBufferSizeIterator / thread;
  std::advance(dataCacheBufferSizeIterator, 1);
  auto l2_size = *dataCacheBufferSizeIterator / thread;
  std::advance(dataCacheBufferSizeIterator, 1);
  auto l3_size = *dataCacheBufferSizeIterator / thread;
  auto ram_size = ramBufferSize / thread;

  auto l2_loop_count =
      getL2LoopCount(sequence, numberOfLines, l2_size * thread);
  auto l3_loop_count =
      getL3LoopCount(sequence, numberOfLines, l3_size * thread);
  auto ram_loop_count =
      getRAMLoopCount(sequence, numberOfLines, ram_size * thread);

  // TODO: they are sometimes by 1 off. check if this has an significant
  // influece.
  log::debug() << "Loop counts: " << l2_loop_count << " " << l3_loop_count
               << " " << ram_loop_count;

  auto pointer_reg = rax;
  auto l1_addr = rbx;
  auto l2_addr = rcx;
  auto l3_addr = r8;
  auto ram_addr = r9;
  auto l2_count_reg = r10;
  auto l3_count_reg = r11;
  auto ram_count_reg = r12;
  auto temp_reg = r13;
  auto offset_reg = r14;
  auto addrHigh_reg = r15;
  auto iter_reg = mm0;
  auto shift_reg = std::vector<Gp>({rdi, rsi, rdx});
  auto shift_reg32 = std::vector<Gp>({edi, esi, edx});
  auto nr_shift_regs = 3;
  auto mul_regs = 2;
  auto add_regs = 9;
  auto alt_dst_regs = 3;
  auto ram_reg = ymm15;

  FuncDetail func;
  func.init(FuncSignatureT<unsigned long long, unsigned long long *,
                           volatile unsigned long long *, unsigned long long>(
      CallConv::kIdHost));

  FuncFrame frame;
  frame.init(func);

  // make (x|y)mm registers dirty
  for (int i = 0; i < 16; i++) {
    frame.addDirtyRegs(Ymm(i));
  }
  frame.addDirtyRegs(Mm(0));
  // make all other used registers dirty except RAX
  frame.addDirtyRegs(l1_addr, l2_addr, l3_addr, ram_addr, l2_count_reg,
                     l3_count_reg, ram_count_reg, temp_reg, offset_reg,
                     addrHigh_reg, iter_reg, ram_addr);
  for (const auto &reg : shift_reg) {
    frame.addDirtyRegs(reg);
  }

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
  // Initialize registers for shift operations
  for (auto const &reg : shift_reg32) {
    cb.mov(reg, Imm(0xAAAAAAAA));
  }
  // Initialize AVX-Registers for FMA Operations
  cb.vmovapd(ymm0, ymmword_ptr(pointer_reg));
  cb.vmovapd(ymm1, ymmword_ptr(pointer_reg));
  auto add_start = mul_regs;
  auto add_end = mul_regs + add_regs - 1;
  auto trans_start = add_regs + mul_regs;
  auto trans_end = add_regs + mul_regs + alt_dst_regs - 1;
  for (int i = add_start; i <= trans_end; i++) {
    cb.vmovapd(Ymm(i), ymmword_ptr(pointer_reg, 256 + i * 32));
  }
  cb.mov(l1_addr, pointer_reg); // address for L1-buffer
  cb.mov(l2_addr, pointer_reg);
  cb.add(l2_addr, Imm(l1_size)); // address for L2-buffer
  cb.mov(l3_addr, pointer_reg);
  cb.add(l3_addr, Imm(l2_size)); // address for L3-buffer
  cb.mov(ram_addr, pointer_reg);
  cb.add(ram_addr, Imm(l3_size)); // address for RAM-buffer
  cb.mov(l2_count_reg, Imm(l2_loop_count));
  log::debug() << "reset counter for L2-buffer with "
               << " cache line accesses per loop ("
               << ") KB";
  cb.mov(l3_count_reg, Imm(l3_loop_count));
  log::debug() << "reset counter for L3-buffer with "
               << " cache line accesses per loop ("
               << ") KB";
  cb.mov(ram_count_reg, Imm(ram_loop_count));
  log::debug() << "reset counter for RAM-buffer with "
               << " cache line accesses per loop ("
               << ") KB";

  cb.addNode(cb.newAlignNode(kAlignCode, 64));

  auto Loop = cb.newLabel();
  cb.bind(Loop);

  unsigned shift_pos = 0;
  bool left = false;
  unsigned add_dest = add_start + 1;
  unsigned mov_dst = trans_start;
  unsigned mov_src = mov_dst + 1;
  unsigned l1_offset = 0;

  for (unsigned count = 0; count < repetitions; count++) {
    for (const auto &item : sequence) {
      // TODO: add different instructions
      if (item == "REG") {
        cb.vfmadd231pd(
            Ymm(add_dest), ymm0,
            Ymm(add_start + (add_dest - add_start + add_regs + 1) % add_regs));
        cb.vfmadd231pd(
            Ymm(mov_dst), ymm1,
            Ymm(add_start + (add_dest - add_start + add_regs + 2) % add_regs));
        cb.xor_(shift_reg[(shift_pos + nr_shift_regs - 1) % nr_shift_regs],
                temp_reg);
        mov_dst++;
      } else if (item == "L1_LS") {
        cb.vmovapd(xmmword_ptr(l1_addr, 64), Xmm(add_dest));
        cb.vfmadd231pd(Ymm(add_dest), ymm0, ymmword_ptr(l1_addr, 32));
        l1_offset += 64;
        if (l1_offset < l1_size * 0.5) {
          cb.add(l1_addr, offset_reg);
        } else {
          l1_offset = 0;
          cb.mov(l1_addr, pointer_reg);
        }
      } else if (item == "L1_2LS_256") {
        cb.vfmadd231pd(Ymm(add_dest), ymm0, ymmword_ptr(l1_addr, 64));
        cb.vfmadd231pd(Ymm(mov_dst), ymm1, ymmword_ptr(l1_addr, 96));
        cb.vmovapd(ymmword_ptr(l1_addr, 32), Ymm(add_dest));
        l1_offset += 128;
        if (l1_offset < l1_size * 0.5) {
          cb.add(l1_addr, Imm(128));
        } else {
          l1_offset = 0;
          cb.mov(l1_addr, pointer_reg);
        }
      } else if (item == "L2_LS") {
        cb.vmovapd(xmmword_ptr(l2_addr, 96), Xmm(add_dest));
        cb.vfmadd231pd(Ymm(add_dest), ymm0, ymmword_ptr(l2_addr, 64));
        cb.add(l2_addr, offset_reg);
      } else if (item == "L2_LS_256") {
        cb.vmovapd(ymmword_ptr(l2_addr, 96), Ymm(add_dest));
        cb.vfmadd231pd(Ymm(add_dest), ymm0, ptr(l2_addr, 64));
        cb.add(l2_addr, offset_reg);
      } else if (item == "L3_LS") {
        cb.vmovapd(xmmword_ptr(l3_addr, 96), Xmm(add_dest));
        cb.vfmadd231pd(Ymm(add_dest), ymm0, ymmword_ptr(l3_addr, 64));
        cb.add(l3_addr, offset_reg);
      } else if (item == "L3_LS_256") {
        cb.vmovapd(ymmword_ptr(l2_addr, 96), Ymm(add_dest));
        cb.vfmadd231pd(Ymm(add_dest), ymm0, ptr(l3_addr, 64));
        cb.add(l3_addr, offset_reg);
      } else if (item == "RAM_L") {
        cb.vfmadd231pd(
            Ymm(add_dest), ymm0,
            Ymm(add_start + (add_dest - add_start + add_regs + 1) % add_regs));
        cb.vfmadd231pd(ram_reg, ymm1, ptr(ram_addr, 64));
        cb.add(ram_addr, offset_reg);
      }

      if (item != "L1_2LS_256" && item != "L2_2LS_256") {
        if (left) {
          cb.shr(shift_reg32[shift_pos], Imm(1));
        } else {
          cb.shl(shift_reg32[shift_pos], Imm(1));
        }
      }
      add_dest++;
      if (add_dest > add_end) {
        add_dest = add_start;
      }
      if (mov_dst > trans_end) {
        mov_dst = trans_start;
      }
      mov_src++;
      if (mov_src > trans_end) {
        mov_src = trans_start;
      }
      shift_pos++;
      if (shift_pos == nr_shift_regs) {
        shift_pos = 0;
        left = !left;
      }
    }
  }

  cb.movq(temp_reg, iter_reg); // restore iteration counter
  if (this->getRAMSequenceCount(sequence) > 0) {
    // reset RAM counter
    auto NoRamReset = cb.newLabel();

    cb.sub(ram_count_reg, Imm(1));
    cb.jnz(NoRamReset);
    cb.mov(ram_count_reg, Imm(ram_loop_count));
    cb.mov(ram_addr, pointer_reg);
    cb.add(ram_addr, Imm(l3_size));
    cb.bind(NoRamReset);
  }
  cb.inc(temp_reg); // increment iteration counter
  if (this->getL2SequenceCount(sequence) > 0) {
    // reset L2-Cache counter
    auto NoL2Reset = cb.newLabel();

    cb.sub(l2_count_reg, Imm(1));
    cb.jnz(NoL2Reset);
    cb.mov(l2_count_reg, Imm(l2_loop_count));
    cb.mov(l2_addr, pointer_reg);
    cb.add(l2_addr, Imm(l1_size));
    cb.bind(NoL2Reset);
  }
  cb.movq(iter_reg, temp_reg); // store iteration counter
  if (this->getL3SequenceCount(sequence) > 0) {
    // reset L3-Cache counter
    auto NoL3Reset = cb.newLabel();

    cb.sub(l3_count_reg, Imm(1));
    cb.jnz(NoL3Reset);
    cb.mov(l3_count_reg, Imm(l3_loop_count));
    cb.mov(l3_addr, pointer_reg);
    cb.add(l3_addr, Imm(l2_size));
    cb.bind(NoL3Reset);
  }
  cb.mov(l1_addr, pointer_reg);

  cb.test(ptr_64(addrHigh_reg), Imm(1));
  cb.jnz(Loop);

  cb.bind(FunctionExit);

  cb.movq(rax, iter_reg);

  cb.emitEpilog(frame);

  cb.finalize();

  // String sb;
  // cb.dump(sb);

  Error err = this->rt.add(&this->loadFunction, &code);
  if (err) {
    log::error() << "Error: Asmjit adding Assembler to JitRuntime failed in "
                 << __FILE__ << " at " << __LINE__;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

std::list<std::string> FMAPayload::getAvailableInstructions(void) {}

void FMAPayload::init(unsigned long long *memoryAddr,
                      unsigned long long bufferSize) {
  X86Payload::init(memoryAddr, bufferSize, 0.27948995982e-4, 0.27948995982e-4);
}
