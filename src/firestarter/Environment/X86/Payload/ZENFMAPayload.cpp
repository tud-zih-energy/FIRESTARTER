#include <firestarter/Environment/X86/Payload/ZENFMAPayload.hpp>
#include <firestarter/Logging/Log.hpp>

#include <iterator>
#include <utility>

using namespace firestarter::environment::x86::payload;
using namespace asmjit;
using namespace asmjit::x86;

int ZENFMAPayload::compilePayload(
    std::vector<std::pair<std::string, unsigned>> proportion,
    std::list<unsigned> dataCacheBufferSize, unsigned ramBufferSize,
    unsigned thread, unsigned numberOfLines) {
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
      log::error() << "Error: Instruction group " << item << " undefined in "
                   << name << ".";
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

  // calculate the buffer sizes
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
  auto nr_shift_regs = 3;
  auto fma_io_regs = 6;
  auto ram_reg = ymm15;

  FuncDetail func;
  func.init(FuncSignatureT<unsigned long long, unsigned long long *,
                           volatile unsigned long long *, unsigned long long>(
                CallConv::kIdHost),
            this->rt.environment());

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
  // FIXME: asmjit assigment to mm0 does not seem to be supported
  args.assignAll(pointer_reg, addrHigh_reg, temp_reg);
  args.updateFuncFrame(frame);
  frame.finalize();

  cb.emitProlog(frame);
  cb.emitArgsAssignment(frame, args);

  // FIXME: movq from temp_reg to iter_reg
  cb.movq(iter_reg, temp_reg);

  // stop right away if low load is selected
  auto FunctionExit = cb.newLabel();

  cb.mov(temp_reg, ptr_64(addrHigh_reg));
  cb.test(temp_reg, temp_reg);
  cb.jz(FunctionExit);

  cb.mov(offset_reg,
         Imm(64)); // increment after each cache/memory access
  // Initialize registers for shift operations
  for (auto const &reg : shift_reg) {
    cb.mov(reg, Imm(0xAAAAAAAAAAAAAAAA));
  }
  // Initialize AVX-Registers for FMA Operations
  cb.vmovapd(ymm0, ymmword_ptr(pointer_reg));
  auto fma_input_start = 1;
  auto fma_input_end = fma_input_start + fma_io_regs - 1;
  auto fma_output_start = fma_input_end + 1;
  auto fma_output_end = fma_output_start + fma_io_regs - 1;
  for (int i = fma_input_start; i < fma_output_end; i++) {
    cb.vmovapd(Ymm(i), ymmword_ptr(pointer_reg, 256 + i * 32));
  }

  // Initialize xmm14 for shift operation
  // cb.mov(temp_reg, Imm(1));
  // cb.movd(temp_reg, Xmm(14));
  cb.movd(shift_reg[0], Xmm(13));
  cb.vbroadcastss(Xmm(13), Xmm(13));
  cb.vmovapd(Xmm(14), Xmm(13));
  cb.vpsrlq(Xmm(14), Xmm(14), Imm(1));

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

  cb.align(kAlignCode, 64);

  auto Loop = cb.newLabel();
  cb.bind(Loop);

  unsigned shift_pos = 0;
  bool left = false;
  unsigned itemCount = 0;
  unsigned fma_pos = 0;
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
        cb.vfmadd231pd(Ymm(fma_input_start + fma_pos), ymm0,
                       Ymm(fma_output_start + fma_pos));
        cb.xor_(temp_reg,
                shift_reg[(shift_pos + nr_shift_regs - 1) % nr_shift_regs]);
        if (left) {
          cb.shr(shift_reg[shift_pos], Imm(1));
        } else {
          cb.shl(shift_reg[shift_pos], Imm(1));
        }
      } else if (item == "L1_LS") {
        cb.vfmadd231pd(Ymm(fma_input_start + fma_pos), ymm0,
                       ymmword_ptr(l1_addr, 32));
        cb.vmovapd(ymmword_ptr(l1_addr, 64), Ymm(fma_input_start + fma_pos));
        L1_INCREMENT();
      } else if (item == "L2_L") {
        cb.vfmadd231pd(Ymm(fma_input_start + fma_pos), ymm0,
                       ymmword_ptr(l2_addr, 64));
        cb.xor_(temp_reg,
                shift_reg[(shift_pos + nr_shift_regs - 1) % nr_shift_regs]);
        L2_INCREMENT();
      } else if (item == "L3_L") {
        cb.vfmadd231pd(Ymm(fma_input_start + fma_pos), ymm0,
                       ymmword_ptr(l3_addr, 64));
        cb.xor_(temp_reg,
                shift_reg[(shift_pos + nr_shift_regs - 1) % nr_shift_regs]);
        L3_INCREMENT();
      } else if (item == "RAM_L") {
        cb.vfmadd231pd(Ymm(ram_reg), ymm0, ymmword_ptr(ram_addr, 32));
        cb.xor_(temp_reg,
                shift_reg[(shift_pos + nr_shift_regs - 1) % nr_shift_regs]);
        RAM_INCREMENT();
      } else {
        log::error() << "Error: Instruction group " << item << " not found in "
                     << this->name << ".";
        return EXIT_FAILURE;
      }

      switch (itemCount % 4) {
      case 0:
        cb.vpsrlq(Xmm(13), Xmm(13), Imm(1));
        break;
      case 1:
        cb.vpsllq(Xmm(14), Xmm(14), Imm(1));
        break;
      case 2:
        cb.vpsllq(Xmm(13), Xmm(13), Imm(1));
        break;
      case 3:
        cb.vpsrlq(Xmm(14), Xmm(14), Imm(1));
        break;
      }

      itemCount++;
      fma_pos = (fma_pos + 1) % fma_io_regs;
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

std::list<std::string> ZENFMAPayload::getAvailableInstructions(void) {
  std::list<std::string> instructions;

  transform(this->instructionFlops.begin(), this->instructionFlops.end(),
            back_inserter(instructions),
            [](const auto &item) { return item.first; });

  return instructions;
}

void ZENFMAPayload::init(unsigned long long *memoryAddr,
                         unsigned long long bufferSize) {
  X86Payload::init(memoryAddr, bufferSize, 0.27948995982e-4, 0.27948995982e-4);
}