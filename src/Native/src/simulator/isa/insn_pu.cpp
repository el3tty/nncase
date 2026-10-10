// Lifted from IDA/Hex-Rays output; addresses refer to the original binary.
// PuComputeInstruction1.cpp  @0x41dde0
// Simulator45.cpp  @0x4170a0
// Simulator37.cpp  @0x415e50
// Simulator38.cpp  @0x416080
// Simulator39.cpp  @0x4162d0
// Simulator40.cpp  @0x416530
// Simulator41.cpp  @0x416780
// Simulator46.cpp  @0x4172c0
// Simulator43.cpp  @0x416bf0
// Simulator44.cpp  @0x416e50
// Simulator42.cpp  @0x4169d0
#include "isa/insn_pu.h"
#include "globals.h"
#include "isa/pu_common.h"
#include "engines/conv2d.h"
#include "engines/act0.h"
#include "engines/checkpoint.h"

// ---- PuComputeInstruction ----
void PuComputeInstruction::get_next_pc()
{
    next_pc_ = pc_ + 2;
}

// PuComputeInstruction2.cpp  @0x423280
void PuComputeInstruction::operation()
{
    if (flag_)
        debug_flag = 1;
    Conv2D* conv = Conv2D::GetConv2D();
    conv->cfg_.shift_mode_ = of_shift_mode_;                   // +152

    // Queue a snapshot of the current configuration (inlined std::deque::push_back of a shared_ptr).
    conv->compute_queue_.push_back(conv->GetPuCompute());

    // cfg.mode (+144) is PuComputeConf's compute_mode.  Anything but 1 runs the compute immediately.
    if (conv->cfg_.mode_ != 1) {
        conv->Compute();
        return;
    }
    // Mode 1: run if a DmStoreOf is already queued, or the oldest Act0Compute does not route to the DM
    // store (out_route == 0).
    // verified against asm @0x423280: act0_queue.front() is dereferenced without an emptiness check (cmpb $0,0x1c(front)); Compute() if store_queue non-empty or that byte is 0
    if (!conv->store_queue_.empty() || !conv->act0_queue_.front()->out_route_)
        conv->Compute();
}

// PuComputeInstruction3.cpp  @0x425900
PuComputeInstruction::~PuComputeInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40ec45.
template <>
PuComputeInstruction Simulator::InstParser<PuComputeInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  PuComputeInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.tcu_id_ = kinst_bits(raw, 7, 3);
  inst.of_shift_mode_ = kinst_bits(raw, 10, 2);
  inst.reserved_12_ = kinst_bits(raw, 12, 4);
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x400000003LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- PuComputeConfInstruction ----
template <>
PuComputeConfInstruction Simulator::InstParser<PuComputeConfInstruction, 32>(uint8_t** pc)
{
    PuComputeConfInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.load_psum_ = pu::bits(raw, 17, 1);
    inst.clr_psum_ = pu::bits(raw, 18, 1);
    inst.dest_target_ = pu::bits(raw, 19, 1);
    inst.release_if_ = pu::bits(raw, 20, 1);
    inst.mode_ = pu::bits(raw, 21, 1);
    inst.reserved_22_ = pu::bits(raw, 22, 8);

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuComputeConfInstruction1.cpp  @0x41ddd0
void PuComputeConfInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuComputeConfInstruction2.cpp  @0x41e0b0
void PuComputeConfInstruction::operation()
{
    // Latch the compute-stage flags into the Conv2D singleton.
    void* conv = Conv2D::GetConv2D();  // TODO(layout): raw offsets into Conv2D
    pu::at<uint8_t>(conv, 140) = load_psum_ != 0;
    pu::at<uint8_t>(conv, 141) = clr_psum_ != 0;
    pu::at<uint32_t>(conv, 144) = dest_target_;
    pu::at<uint32_t>(conv, 148) = mode_;
    pu::at<uint8_t>(conv, 142) = release_if_ != 0;
}

// PuComputeConfInstruction3.cpp  @0x425950
PuComputeConfInstruction::~PuComputeConfInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuFetchifConf1Instruction ----
template <>
PuFetchifConf1Instruction Simulator::InstParser<PuFetchifConf1Instruction, 32>(uint8_t** pc)
{
    PuFetchifConf1Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.stride_w_ = pu::bits(raw, 17, 5);
    inst.stride_h_ = pu::bits(raw, 22, 5);
    inst.rstride_s_ = pu::bits(raw, 27, 3);
    inst.reserved_30_ = pu::bits(raw, 30, 2);
    inst.rstride_s_val_ = g_shape_reg[inst.rstride_s_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuFetchifConf1Instruction1.cpp  @0x41dd50
void PuFetchifConf1Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuFetchifConf1Instruction2.cpp  @0x41e190
void PuFetchifConf1Instruction::operation()
{
    uint32_t* cfg = reinterpret_cast<uint32_t*>(Conv2D::GetConv2D());  // TODO(layout): Conv2D config words
    // IF-map fetch configuration: two immediates plus three rstride_s_val dimensions.
    cfg[0] = stride_w_;
    cfg[1] = stride_h_;
    cfg[2] = pu::shape_word(rstride_s_val_, 2);
    cfg[3] = pu::shape_word(rstride_s_val_, 1);
    cfg[4] = pu::shape_word(rstride_s_val_, 0);
    cfg[5] = 0;
}

// PuFetchifConf1Instruction3.cpp  @0x425bd0
PuFetchifConf1Instruction::~PuFetchifConf1Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuFetchifConf2Instruction ----
template <>
PuFetchifConf2Instruction Simulator::InstParser<PuFetchifConf2Instruction, 32>(uint8_t** pc)
{
    PuFetchifConf2Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

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
    *pc += 4;
    return inst;
}

// PuFetchifConf2Instruction1.cpp  @0x41dd60
void PuFetchifConf2Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuFetchifConf2Instruction2.cpp  @0x425b80
PuFetchifConf2Instruction::~PuFetchifConf2Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuFetchifConf3Instruction ----
template <>
PuFetchifConf3Instruction Simulator::InstParser<PuFetchifConf3Instruction, 32>(uint8_t** pc)
{
    PuFetchifConf3Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.raddr_s_ = pu::bits(raw, 17, 5);
    inst.rgroups_ = pu::bits(raw, 22, 5);
    inst.rshape_ = pu::bits(raw, 27, 3);
    inst.reserved_30_ = pu::bits(raw, 30, 2);
    inst.raddr_s_val_ = g_gp_reg[inst.raddr_s_];
    inst.rgroups_val_ = g_gp_reg[inst.rgroups_];
    inst.rshape_val_ = g_shape_reg[inst.rshape_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuFetchifConf3Instruction1.cpp  @0x41dd70
void PuFetchifConf3Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuFetchifConf3Instruction2.cpp  @0x41e0f0
void PuFetchifConf3Instruction::operation()
{
    uint32_t* cfg = reinterpret_cast<uint32_t*>(Conv2D::GetConv2D());  // TODO(layout): Conv2D config words
    cfg[6] = rgroups_val_;
    cfg[7] = raddr_s_val_;
    cfg[8] = pu::shape_word(rshape_val_, 3);
    cfg[9] = pu::shape_word(rshape_val_, 2);
    cfg[10] = pu::shape_word(rshape_val_, 1);
    cfg[11] = pu::shape_word(rshape_val_, 0);
}

// PuFetchifConf3Instruction3.cpp  @0x425b30
PuFetchifConf3Instruction::~PuFetchifConf3Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuFetchifConf4Instruction ----
template <>
PuFetchifConf4Instruction Simulator::InstParser<PuFetchifConf4Instruction, 32>(uint8_t** pc)
{
    PuFetchifConf4Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.rpad_value_ = pu::bits(raw, 17, 5);
    inst.reserved_22_ = pu::bits(raw, 22, 5);
    inst.sspad_ = pu::bits(raw, 27, 3);
    inst.reserved_30_ = pu::bits(raw, 30, 2);
    inst.rpad_value_val_ = g_gp_reg[inst.rpad_value_];
    inst.sspad_val_ = g_shape_reg[inst.sspad_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuFetchifConf4Instruction1.cpp  @0x41dd80
void PuFetchifConf4Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuFetchifConf4Instruction2.cpp  @0x41e220
void PuFetchifConf4Instruction::operation()
{
    void* conv = Conv2D::GetConv2D();  // TODO(layout): raw byte offsets into Conv2D
    pu::at<uint8_t>(conv, 48) = rpad_value_val_;  // low byte only
    pu::at<uint32_t>(conv, 52) = pu::shape_word(sspad_val_, 2);
    pu::at<uint32_t>(conv, 56) = pu::shape_word(sspad_val_, 3);
    pu::at<uint32_t>(conv, 64) = pu::shape_word(sspad_val_, 0);
    pu::at<uint32_t>(conv, 60) = pu::shape_word(sspad_val_, 1);
}

// PuFetchifConf4Instruction3.cpp  @0x425ae0
PuFetchifConf4Instruction::~PuFetchifConf4Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuFetchifConf_deqInstruction ----
template <>
PuFetchifConf_deqInstruction Simulator::InstParser<PuFetchifConf_deqInstruction, 32>(uint8_t** pc)
{
    PuFetchifConf_deqInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.ric_ = pu::bits(raw, 17, 5);
    inst.rbx_ = pu::bits(raw, 22, 5);
    inst.quant_type_ = pu::bits(raw, 27, 2);
    inst.reserved_29_ = pu::bits(raw, 29, 3);
    inst.ric_val_ = g_gp_reg[inst.ric_];
    inst.rbx_val_ = g_gp_reg[inst.rbx_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuFetchifConf_deqInstruction1.cpp  @0x41dd90
void PuFetchifConf_deqInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuFetchifConf_deqInstruction2.cpp  @0x41fa30
void PuFetchifConf_deqInstruction::operation()
{
    uint8_t* conv = reinterpret_cast<uint8_t*>(Conv2D::GetConv2D());  // TODO(layout): byte table at Conv2D+68
    if (ric_val_ == 31)
        std::memset(conv + 68, static_cast<uint8_t>(rbx_val_), 24);  // broadcast to all 24 slots (68..91)
    else
        conv[68 + ric_val_] = static_cast<uint8_t>(rbx_val_);
    pu::at<uint32_t>(conv, 92) = quant_type_;

    // CheckPoint trace of the re-packed instruction word.
    // verified against asm @0x41fa30: the binary really packs the register *values* ric_val/rbx_val (0x38/0x3c)
    // together with the field values; the shift/or chain matches.
    const uint64_t inner = static_cast<uint64_t>(rbx_val_) | (static_cast<uint64_t>(quant_type_) << 8);
    const uint64_t packed =
        (((8 * ((8 * (static_cast<uint64_t>(funct4_)
                      | (16 * (static_cast<uint64_t>(ric_val_) | (32 * inner)))))
                | pu_id_))
          | tcu_id_) << 7)
        | opcode_;
    pu::log_inst_word(packed);
}

// PuFetchifConf_deqInstruction3.cpp  @0x425a90
PuFetchifConf_deqInstruction::~PuFetchifConf_deqInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuForward_psumInstruction ----
template <>
PuForward_psumInstruction Simulator::InstParser<PuForward_psumInstruction, 32>(uint8_t** pc)
{
    PuForward_psumInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_ = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.raddr_ = pu::bits(raw, 13, 5);
    inst.rlen_ = pu::bits(raw, 18, 5);
    inst.reserved_21_ = pu::bits(raw, 21, 9) & 0x1FD;   // verified against asm @0x4172c0: raw[29:21] with bit 1 (raw[22]) not copied
    inst.raddr_val_ = g_gp_reg[inst.raddr_];
    inst.rlen_val_ = g_gp_reg[inst.rlen_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuForward_psumInstruction1.cpp  @0x41ddf0
void PuForward_psumInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuForward_psumInstruction2.cpp  @0x41e5d0
void PuForward_psumInstruction::operation()
{
    Conv2D* conv = reinterpret_cast<Conv2D*>(Conv2D::GetConv2D());
    // Act0::GetAct0() singleton lazy-init elided by the decompiler.
    // verified against asm @0x41e5d0: row count is the int at Act0+0x40004 (0x5CC764); src index = (int)((rs1>>2) + row*1024), copy size 4*rs2_val bytes
    const int num_rows = static_cast<int>(reinterpret_cast<const Act0*>(Act0_act0)->channels_);
    const uint64_t src_word_off = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(raddr_val_))) >> 2;
    for (int row = 0; row < num_rows; ++row) {
        // Each row is 4096 bytes (1024 words) in both PSUM_L1 and Act0_act0.
        const int src = static_cast<int>(src_word_off + static_cast<uint64_t>(row) * 1024);
        std::memcpy(Act0_act0 + static_cast<size_t>(row) * 1024, PSUM_L1 + src, 4 * rlen_val_);
    }
    conv->Activate();
}

// PuForward_psumInstruction3.cpp  @0x4258b0
PuForward_psumInstruction::~PuForward_psumInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuOfConf1Instruction ----
template <>
PuOfConf1Instruction Simulator::InstParser<PuOfConf1Instruction, 32>(uint8_t** pc)
{
    PuOfConf1Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.rgoc_ = pu::bits(raw, 17, 5);
    inst.rgoc_last_ = pu::bits(raw, 22, 5);
    inst.rstride_d_ = pu::bits(raw, 27, 3);
    inst.reserved_30_ = pu::bits(raw, 30, 2);
    inst.rgoc_val_ = g_gp_reg[inst.rgoc_];
    inst.rgoc_last_val_ = g_gp_reg[inst.rgoc_last_];
    inst.rstride_d_val_ = g_shape_reg[inst.rstride_d_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuOfConf1Instruction1.cpp  @0x41ddb0
void PuOfConf1Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuOfConf1Instruction2.cpp  @0x41e1e0
void PuOfConf1Instruction::operation()
{
    uint32_t* cfg = reinterpret_cast<uint32_t*>(Conv2D::GetConv2D());  // TODO(layout): Conv2D config words
    // Output-feature-map rstride_d_val: three dimensions, cfg[29] cleared.
    cfg[26] = pu::shape_word(rstride_d_val_, 2);
    cfg[27] = pu::shape_word(rstride_d_val_, 1);
    cfg[28] = pu::shape_word(rstride_d_val_, 0);
    cfg[29] = 0;
}

// PuOfConf1Instruction3.cpp  @0x4259f0
PuOfConf1Instruction::~PuOfConf1Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuOfConf2Instruction ----
template <>
PuOfConf2Instruction Simulator::InstParser<PuOfConf2Instruction, 32>(uint8_t** pc)
{
    PuOfConf2Instruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    inst.tcu_id_  = pu::bits(raw, 7, 3);
    inst.pu_id_ = pu::bits(raw, 10, 3);
    inst.funct4_ = pu::bits(raw, 13, 4);
    inst.raddr_d_ = pu::bits(raw, 17, 5);
    inst.reserved_22_ = pu::bits(raw, 22, 5);
    inst.rshape_d_ = pu::bits(raw, 27, 3);
    inst.reserved_30_ = pu::bits(raw, 30, 2);
    inst.raddr_d_val_ = g_gp_reg[inst.raddr_d_];
    inst.rshape_d_val_ = g_shape_reg[inst.rshape_d_];

    inst.pc_ = cur;
    inst.info_ = pu::kInstInfo;
    inst.pc_rel_ = cur - start_pc_;
    *pc += 4;
    return inst;
}

// PuOfConf2Instruction1.cpp  @0x41ddc0
void PuOfConf2Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuOfConf2Instruction2.cpp  @0x41e140
void PuOfConf2Instruction::operation()
{
    uint32_t* cfg = reinterpret_cast<uint32_t*>(Conv2D::GetConv2D());  // TODO(layout): Conv2D config words
    cfg[30] = pu::shape_word(rshape_d_val_, 3);
    cfg[31] = pu::shape_word(rshape_d_val_, 2);
    cfg[32] = pu::shape_word(rshape_d_val_, 1);
    cfg[33] = pu::shape_word(rshape_d_val_, 0);
    cfg[34] = raddr_d_val_;
}

// PuOfConf2Instruction3.cpp  @0x4259a0
PuOfConf2Instruction::~PuOfConf2Instruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}

// ---- PuWConfInstruction ----
template <>
PuWConfInstruction Simulator::InstParser<PuWConfInstruction, 32>(uint8_t** pc)
{
    PuWConfInstruction inst;  // KInstruction base (vptr, empty name) is constructed by the compiler
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(*pc);
    const uint32_t cur = KPU_PC(*pc);  // offset in DDR

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
    *pc += 4;
    return inst;
}

// PuWConfInstruction1.cpp  @0x41dda0
void PuWConfInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// PuWConfInstruction2.cpp  @0x41f8c0
void PuWConfInstruction::operation()
{
    void* conv = Conv2D::GetConv2D();  // TODO(layout): raw byte offsets into Conv2D
    pu::at<uint32_t>(conv, 96) = kernel_h_;   // weight configuration word 0
    pu::at<uint32_t>(conv, 100) = kernel_w_;  // weight configuration word 1

    // CheckPoint trace of the re-packed instruction word (fields back at bits 7, 10, 13, 17, 22).
    const uint64_t packed =
        (((8 * ((8 * (static_cast<uint64_t>(funct4_)
                      | (16 * (static_cast<uint64_t>(kernel_h_) | (32 * static_cast<uint64_t>(kernel_w_)))))) | pu_id_))
          | tcu_id_) << 7)
        | opcode_;
    pu::log_inst_word(packed);
}

// PuWConfInstruction3.cpp  @0x425a40
PuWConfInstruction::~PuWConfInstruction()
{
    // vptr reset to KInstruction vtable; std::string name released by the compiler
}
