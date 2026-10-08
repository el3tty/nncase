// Lifted from IDA/Hex-Rays output; addresses refer to the original binary.
// PuPdp0ComputeInstruction1.cpp  @0x41de80
// Simulator52.cpp  @0x418050
// Simulator48.cpp  @0x417740
// Simulator49.cpp  @0x417960
// Simulator50.cpp  @0x417bb0
// Simulator51.cpp  @0x417df0
// Simulator47.cpp  @0x417510
// Simulator54.cpp  @0x4184c0
// Simulator53.cpp  @0x4182a0
#include "isa/insn_pupdp0.h"
#include "globals.h"
#include "isa/pu_common.h"
#include "engines/pdp0.h"

// ---- PuPdp0ComputeInstruction ----
void PuPdp0ComputeInstruction::get_next_pc()
{
    next_pc_ = pc_ + 2;
}

// PuPdp0ComputeInstruction2.cpp  @0x422b20
void PuPdp0ComputeInstruction::operation()
{
    PDP0* pdp0 = static_cast<PDP0*>(PDP0::GetPDP0());
    pdp0->regs_.psum_offset_ = compute_param_;                 // +65632

    // Queue a snapshot of the configuration registers (inlined std::deque::push_back of a shared_ptr).
    pdp0->compute_queue_.push_back(pdp0->GetPdp0Compute());

    // Run when the DmStoreOf queue (+65720) is non-empty.
    // verified against asm @0x422b20: the original tests the store_of_queue size (PDP0+0x100b8 deque), not compute_queue.
    if (!pdp0->store_of_queue_.empty())
        pdp0->Compute();
}

// PuPdp0ComputeInstruction3.cpp  @0x4255e0
PuPdp0ComputeInstruction::~PuPdp0ComputeInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40f0fa.
template <>
PuPdp0ComputeInstruction Simulator::InstParser<PuPdp0ComputeInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  PuPdp0ComputeInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.tcu_id_ = kinst_bits(raw, 7, 3);
  inst.raddr_s_ = kinst_bits(raw, 10, 5);
  inst.reserved_15_ = kinst_bits(raw, 15, 1);
  inst.compute_param_ = g_gp_reg[inst.raddr_s_];
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000003LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.name_ = **(const std::string **)&insn_name_;   // TODO(layout): Simulator::insn_name_ holds a std::string*
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- PuPdp0Conf_deqInstruction ----
template <>
PuPdp0Conf_deqInstruction Simulator::InstParser<PuPdp0Conf_deqInstruction, 32>(uint8_t** pc)
{
    PuPdp0Conf_deqInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.rbx_ = pu::bits(raw, 17, 5);
    inst.quant_type_ = pu::bits(raw, 22, 2);
    inst.reserved_24_ = pu::bits(raw, 24, 8);
    inst.rbx_val_ = g_gp_reg[inst.rbx_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0Conf_deqInstruction1.cpp  @0x41de50
void PuPdp0Conf_deqInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0Conf_deqInstruction2.cpp  @0x422a50
void PuPdp0Conf_deqInstruction::operation()
{
    // verified against asm @0x422a50: one qword store at PDP0+0x10030 = {rs1_val, unsigned_flag}
    PDP0 *pdp0 = static_cast<PDP0 *>(PDP0::GetPDP0());
    pdp0->regs_.in_zero_point_ = rbx_val_;
    pdp0->regs_.unsigned_flag_ = quant_type_;
}

// PuPdp0Conf_deqInstruction3.cpp  @0x4256d0
PuPdp0Conf_deqInstruction::~PuPdp0Conf_deqInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuPdp0FetchifConf1Instruction ----
template <>
PuPdp0FetchifConf1Instruction Simulator::InstParser<PuPdp0FetchifConf1Instruction, 32>(uint8_t** pc)
{
    PuPdp0FetchifConf1Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.stride_w_ = pu::bits(raw, 17, 5);
    inst.stride_h_ = pu::bits(raw, 22, 5);
    inst.reserved_27_ = pu::bits(raw, 27, 5);

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0FetchifConf1Instruction1.cpp  @0x41de10
void PuPdp0FetchifConf1Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0FetchifConf1Instruction2.cpp  @0x422980
void PuPdp0FetchifConf1Instruction::operation()
{
    uint32_t* regs = static_cast<uint32_t*>(PDP0::GetPDP0());  // TODO(layout): PDP0 register file, 32-bit word index
    regs[16385] = stride_w_;  // PDP0 + 65540
    regs[16386] = stride_h_;  // PDP0 + 65544
}

// PuPdp0FetchifConf1Instruction3.cpp  @0x425810
PuPdp0FetchifConf1Instruction::~PuPdp0FetchifConf1Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuPdp0FetchifConf2Instruction ----
template <>
PuPdp0FetchifConf2Instruction Simulator::InstParser<PuPdp0FetchifConf2Instruction, 32>(uint8_t** pc)
{
    PuPdp0FetchifConf2Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.rgic_ = pu::bits(raw, 17, 5);
    inst.rgic_last_ = pu::bits(raw, 22, 5);
    inst.reserved_27_ = pu::bits(raw, 27, 5);
    inst.rgic_val_ = g_gp_reg[inst.rgic_];
    inst.rgic_last_val_ = g_gp_reg[inst.rgic_last_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0FetchifConf2Instruction1.cpp  @0x41de20
void PuPdp0FetchifConf2Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0FetchifConf2Instruction2.cpp  @0x4229a0
void PuPdp0FetchifConf2Instruction::operation()
{
    // No configuration is written; only the PDP0 singleton is (lazily) created.
    (void)PDP0::GetPDP0();
}

// PuPdp0FetchifConf2Instruction3.cpp  @0x4257c0
PuPdp0FetchifConf2Instruction::~PuPdp0FetchifConf2Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuPdp0FetchifConf3Instruction ----
template <>
PuPdp0FetchifConf3Instruction Simulator::InstParser<PuPdp0FetchifConf3Instruction, 32>(uint8_t** pc)
{
    PuPdp0FetchifConf3Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.reserved_17_ = pu::bits(raw, 17, 10);
    inst.rshape_ = pu::bits(raw, 27, 3);
    inst.reserved_30_ = pu::bits(raw, 30, 2);
    inst.shape_ = g_shape_reg[inst.rshape_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0FetchifConf3Instruction1.cpp  @0x41de30
void PuPdp0FetchifConf3Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0FetchifConf3Instruction2.cpp  @0x4229b0
void PuPdp0FetchifConf3Instruction::operation()
{
    uint32_t* regs = static_cast<uint32_t*>(PDP0::GetPDP0());  // TODO(layout): PDP0 register file, 32-bit word index
    regs[16387] = pu::shape_word(shape_, 3);
    regs[16388] = pu::shape_word(shape_, 2);
    regs[16389] = pu::shape_word(shape_, 1);
    regs[16390] = pu::shape_word(shape_, 0);
}

// PuPdp0FetchifConf3Instruction3.cpp  @0x425770
PuPdp0FetchifConf3Instruction::~PuPdp0FetchifConf3Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuPdp0FetchifConf4Instruction ----
template <>
PuPdp0FetchifConf4Instruction Simulator::InstParser<PuPdp0FetchifConf4Instruction, 32>(uint8_t** pc)
{
    PuPdp0FetchifConf4Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.rpad_value_ = pu::bits(raw, 17, 5);
    inst.sspad_ = pu::bits(raw, 22, 3);
    inst.reserved_25_ = pu::bits(raw, 25, 7);
    inst.rpad_value_val_ = g_gp_reg[inst.rpad_value_];
    inst.shape_ = g_shape_reg[inst.sspad_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0FetchifConf4Instruction1.cpp  @0x41de40
void PuPdp0FetchifConf4Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0FetchifConf4Instruction2.cpp  @0x422a00
void PuPdp0FetchifConf4Instruction::operation()
{
    uint32_t* regs = static_cast<uint32_t*>(PDP0::GetPDP0());  // TODO(layout): PDP0 register file, 32-bit word index
    regs[16391] = rpad_value_val_;
    regs[16392] = pu::shape_word(shape_, 2);
    regs[16393] = pu::shape_word(shape_, 3);
    regs[16394] = pu::shape_word(shape_, 1);
    regs[16395] = pu::shape_word(shape_, 0);
}

// PuPdp0FetchifConf4Instruction3.cpp  @0x425720
PuPdp0FetchifConf4Instruction::~PuPdp0FetchifConf4Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuPdp0ModeConfInstruction ----
template <>
PuPdp0ModeConfInstruction Simulator::InstParser<PuPdp0ModeConfInstruction, 32>(uint8_t** pc)
{
    PuPdp0ModeConfInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.mode_ = pu::bits(raw, 17, 3);
    inst.reserved_20_ = pu::bits(raw, 20, 12);

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0ModeConfInstruction1.cpp  @0x41de00
void PuPdp0ModeConfInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0ModeConfInstruction2.cpp  @0x422960
void PuPdp0ModeConfInstruction::operation()
{
    void* pdp0 = PDP0::GetPDP0();  // TODO(layout): raw offset into PDP0
    pu::at<uint32_t>(pdp0, 0x10000) = mode_;  // PDP0 mode register
}

// PuPdp0ModeConfInstruction3.cpp  @0x425860
PuPdp0ModeConfInstruction::~PuPdp0ModeConfInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuPdp0OfConfInstruction ----
template <>
PuPdp0OfConfInstruction Simulator::InstParser<PuPdp0OfConfInstruction, 32>(uint8_t** pc)
{
    PuPdp0OfConfInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.rstride_d_ = pu::bits(raw, 17, 3);
    inst.rshape_d_ = pu::bits(raw, 20, 3);
    inst.reserved_23_ = pu::bits(raw, 23, 9);
    inst.shape0_ = g_shape_reg[inst.rstride_d_];
    inst.shape1_ = g_shape_reg[inst.rshape_d_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0OfConfInstruction1.cpp  @0x41de70
void PuPdp0OfConfInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0OfConfInstruction2.cpp  @0x422aa0
void PuPdp0OfConfInstruction::operation()
{
    uint32_t* regs = static_cast<uint32_t*>(PDP0::GetPDP0());  // TODO(layout): PDP0 register file, 32-bit word index
    regs[16400] = pu::shape_word(shape1_, 3);
    regs[16401] = pu::shape_word(shape1_, 2);
    regs[16402] = pu::shape_word(shape1_, 1);
    regs[16403] = pu::shape_word(shape1_, 0);
    regs[16404] = pu::shape_word(shape0_, 2);
    regs[16405] = pu::shape_word(shape0_, 1);
    regs[16406] = pu::shape_word(shape0_, 0);
    regs[16407] = 0;
}

// PuPdp0OfConfInstruction3.cpp  @0x425630
PuPdp0OfConfInstruction::~PuPdp0OfConfInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuPdp0WConfInstruction ----
template <>
PuPdp0WConfInstruction Simulator::InstParser<PuPdp0WConfInstruction, 32>(uint8_t** pc)
{
    PuPdp0WConfInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(*pc) - reinterpret_cast<uintptr_t>(g_DDR));  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.kernel_h_ = pu::bits(raw, 17, 5);
    inst.kernel_w_ = pu::bits(raw, 22, 5);
    inst.reserved_27_ = pu::bits(raw, 27, 5);

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    inst.name_ = *reinterpret_cast<const std::string*>(insn_name_);
    *pc += 4;
    return inst;
}

// PuPdp0WConfInstruction1.cpp  @0x41de60
void PuPdp0WConfInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuPdp0WConfInstruction2.cpp  @0x422a80
void PuPdp0WConfInstruction::operation()
{
    uint32_t* regs = static_cast<uint32_t*>(PDP0::GetPDP0());  // TODO(layout): PDP0 register file, 32-bit word index
    regs[16398] = kernel_w_;  // PDP0 + 65592
    regs[16399] = kernel_h_;  // PDP0 + 65596
}

// PuPdp0WConfInstruction3.cpp  @0x425680
PuPdp0WConfInstruction::~PuPdp0WConfInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}
