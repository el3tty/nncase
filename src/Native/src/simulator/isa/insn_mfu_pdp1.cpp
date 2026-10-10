// Lifted from IDA/Hex-Rays output (K230 NPU C-model simulator).
// Member names describe the decoded instruction fields; byte offsets are given in the header.
#include <cstring>
#include "isa/insn_mfu_pdp1.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/pdp1.h"
#include "engines/simulator.h"

// ---- MfuPdp1ComputeInstruction ----
namespace {
// Returns `width` bits of `raw` starting at bit `lo`.
inline uint32_t field(uint64_t raw, unsigned lo, unsigned width)
{
    return static_cast<uint32_t>((raw >> lo) & ((1ull << width) - 1));
}
}  // namespace

// Simulator67.cpp  @0x41a490
template <>
MfuPdp1ComputeInstruction Simulator::InstParser<MfuPdp1ComputeInstruction, 32>(uint8_t ** pc)
{
    MfuPdp1ComputeInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.raddr_d_ = field(raw, 7, 5);
    inst.raddr_s_ = field(raw, 12, 5);
    inst.rshape_ = field(raw, 17, 3);
    inst.reserved_20_ = field(raw, 20, 12);  // verified against asm: wider immediate (unused by operation)
    inst.raddr_d_val_ = _G.gp_reg[inst.raddr_d_];
    inst.raddr_s_val_ = _G.gp_reg[inst.raddr_s_];
    inst.rshape_val_ = _G.shape_reg[inst.rshape_];
    // (IDA also computed MMU-translated raddr_s_val here, but the result was discarded.)
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuPdp1ComputeInstruction1.cpp  @0x41df50
void MfuPdp1ComputeInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuPdp1ComputeInstruction2.cpp  @0x424d10
void MfuPdp1ComputeInstruction::operation()
{
    // verified against asm @0x424d10: dword stores at PDP1+0xa8/+0xa4, halfword stores at +0xba/+0xb4/+0xb6/+0xb8
    PDP1 *pdp1 = PDP1::GetPDP1();
    Pdp1Config &pdp1_cfg = pdp1->cfg_;
    pdp1_cfg.dst_addr_  = raddr_d_val_;
    pdp1_cfg.src_addr_  = raddr_s_val_;
    pdp1_cfg.shape_w_   = static_cast<uint16_t>(rshape_val_);        // bits 0..15
    pdp1_cfg.dim1_count_ = static_cast<uint16_t>(rshape_val_ >> 48); // bits 48..63
    pdp1_cfg.dim2_count_ = static_cast<uint16_t>(rshape_val_ >> 32); // bits 32..47
    pdp1_cfg.shape_h_   = static_cast<uint16_t>(rshape_val_ >> 16);  // bits 16..31
    pdp1->PdpRedCompute();
}

// MfuPdp1ComputeInstruction3.cpp  @0x4251d0
MfuPdp1ComputeInstruction::~MfuPdp1ComputeInstruction()
{
}

// ---- MfuPdp1Conf1Instruction ----
// Simulator61.cpp  @0x419630
template <>
MfuPdp1Conf1Instruction Simulator::InstParser<MfuPdp1Conf1Instruction, 32>(uint8_t ** pc)
{
    MfuPdp1Conf1Instruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.stride_w_ = field(raw, 12, 5);
    inst.stride_h_ = field(raw, 17, 5);
    inst.rstride_s_ = field(raw, 22, 3);
    inst.funct2_ = field(raw, 25, 2);
    inst.rstride_d_ = field(raw, 27, 3);
    inst.reserved_30_ = field(raw, 30, 2);
    inst.rstride_s_val_ = _G.shape_reg[inst.rstride_s_];
    inst.rstride_d_val_ = _G.shape_reg[inst.rstride_d_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuPdp1Conf1Instruction1.cpp  @0x41def0
void MfuPdp1Conf1Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuPdp1Conf1Instruction2.cpp  @0x4249c0
void MfuPdp1Conf1Instruction::operation()
{
    // verified against asm @0x4249c0: u16 store at PDP1+0xc0 (pair {stride_h:stride_w}), u64 at +0xd0, u8 at +0xc7, u64 at +0xe0
    Pdp1Config &pdp1_cfg = PDP1::GetPDP1()->cfg_;
    const uint16_t strides = static_cast<uint16_t>(stride_w_ | (stride_h_ << 8));
    std::memcpy(&pdp1_cfg.stride_w_, &strides, sizeof strides);
    pdp1_cfg.src_pitch_ = static_cast<uint16_t>(rstride_s_val_);
    pdp1_cfg.src_plane_rows_ = static_cast<uint16_t>(rstride_s_val_ >> 16);
    pdp1_cfg.src_dim2_ = static_cast<uint16_t>(rstride_s_val_ >> 32);
    pdp1_cfg.pool_mode_ = static_cast<uint8_t>(funct2_);
    pdp1_cfg.dst_pitch_ = static_cast<uint16_t>(rstride_d_val_);
    pdp1_cfg.dst_plane_rows_ = static_cast<uint16_t>(rstride_d_val_ >> 16);
    pdp1_cfg.dst_dim2_ = static_cast<uint16_t>(rstride_d_val_ >> 32);
}

// MfuPdp1Conf1Instruction3.cpp  @0x4253b0
MfuPdp1Conf1Instruction::~MfuPdp1Conf1Instruction()
{
}

// ---- MfuPdp1Conf2Instruction ----
// Simulator62.cpp  @0x419880
template <>
MfuPdp1Conf2Instruction Simulator::InstParser<MfuPdp1Conf2Instruction, 32>(uint8_t ** pc)
{
    MfuPdp1Conf2Instruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rcount_w_ = field(raw, 12, 5);
    inst.rcount_h_ = field(raw, 17, 5);
    inst.rpe_h_ = field(raw, 22, 5);
    inst.rpe_last_h_ = field(raw, 27, 5);
    inst.rcount_w_val_ = _G.gp_reg[inst.rcount_w_];
    inst.rcount_h_val_ = _G.gp_reg[inst.rcount_h_];
    inst.rpe_h_val_ = _G.gp_reg[inst.rpe_h_];
    inst.rpe_last_h_val_ = _G.gp_reg[inst.rpe_last_h_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuPdp1Conf2Instruction1.cpp  @0x41df00
void MfuPdp1Conf2Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuPdp1Conf2Instruction2.cpp  @0x424520
void MfuPdp1Conf2Instruction::operation()
{
    // verified against asm @0x424520: halfword stores at PDP1+0xbc / +0xbe
    Pdp1Config &pdp1_cfg = PDP1::GetPDP1()->cfg_;
    pdp1_cfg.out_w_ = static_cast<uint16_t>(rcount_w_val_);
    pdp1_cfg.out_h_ = static_cast<uint16_t>(rcount_h_val_);
}

// MfuPdp1Conf2Instruction3.cpp  @0x425360
MfuPdp1Conf2Instruction::~MfuPdp1Conf2Instruction()
{
}

// ---- MfuPdp1Conf3Instruction ----
// Simulator63.cpp  @0x419af0
template <>
MfuPdp1Conf3Instruction Simulator::InstParser<MfuPdp1Conf3Instruction, 32>(uint8_t ** pc)
{
    MfuPdp1Conf3Instruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rpe_channels_ = field(raw, 12, 5);
    inst.rpe_last_channels_ = field(raw, 17, 5);
    inst.rpad_value_ = field(raw, 22, 5);
    inst.sspad_ = field(raw, 27, 3);
    inst.reserved_30_ = field(raw, 30, 2);
    inst.rpe_channels_val_ = _G.gp_reg[inst.rpe_channels_];
    inst.rpe_last_channels_val_ = _G.gp_reg[inst.rpe_last_channels_];
    inst.rpad_value_val_ = _G.gp_reg[inst.rpad_value_];
    inst.sspad_val_ = _G.shape_reg[inst.sspad_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuPdp1Conf3Instruction1.cpp  @0x41df10
void MfuPdp1Conf3Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuPdp1Conf3Instruction2.cpp  @0x424350
void MfuPdp1Conf3Instruction::operation()
{
    // verified against asm @0x424350: all stores are narrow (byte, byte, halfword, byte x4) relative to PDP1 (0xd8,0xd9,0xda,0xc5,0xc2,0xc3,0xc4)
    Pdp1Config &pdp1_cfg = PDP1::GetPDP1()->cfg_;
    pdp1_cfg.conf3_rs1_ = static_cast<uint8_t>(rpe_channels_val_);
    pdp1_cfg.conf3_rs2_ = static_cast<uint8_t>(rpe_last_channels_val_);
    pdp1_cfg.pad_value_ = static_cast<uint16_t>(rpad_value_val_);
    pdp1_cfg.pad_right_  = static_cast<uint8_t>(sspad_val_);
    pdp1_cfg.pad_top_    = static_cast<uint8_t>(sspad_val_ >> 48);
    pdp1_cfg.pad_bottom_ = static_cast<uint8_t>(sspad_val_ >> 32);
    pdp1_cfg.pad_left_   = static_cast<uint8_t>(sspad_val_ >> 16);
}

// MfuPdp1Conf3Instruction3.cpp  @0x425310
MfuPdp1Conf3Instruction::~MfuPdp1Conf3Instruction()
{
}

// ---- MfuPdp1Conf4Instruction ----
// Simulator64.cpp  @0x419d60
template <>
MfuPdp1Conf4Instruction Simulator::InstParser<MfuPdp1Conf4Instruction, 32>(uint8_t ** pc)
{
    MfuPdp1Conf4Instruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rwindow_w_ = field(raw, 12, 5);
    inst.rwindow_h_ = field(raw, 17, 5);
    inst.rscale_ = field(raw, 22, 5);
    inst.enable_h2c_ = field(raw, 27, 1);
    inst.enable_bw_ = field(raw, 28, 1);
    inst.reserved_29_ = field(raw, 29, 3);
    inst.rwindow_w_val_ = _G.gp_reg[inst.rwindow_w_];
    inst.rwindow_h_val_ = _G.gp_reg[inst.rwindow_h_];
    inst.rscale_val_ = _G.gp_reg[inst.rscale_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuPdp1Conf4Instruction1.cpp  @0x41df20
void MfuPdp1Conf4Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuPdp1Conf4Instruction2.cpp  @0x424b60
void MfuPdp1Conf4Instruction::operation()
{
    // verified against asm @0x424b60: byte stores at PDP1+0xad/+0xac, halfword at +0xae, bytes at +0xb0/+0xb1
    Pdp1Config &pdp1_cfg = PDP1::GetPDP1()->cfg_;
    pdp1_cfg.window_h_   = static_cast<uint8_t>(rwindow_h_val_);
    pdp1_cfg.window_w_   = static_cast<uint8_t>(rwindow_w_val_);
    pdp1_cfg.avg_scale_  = static_cast<uint16_t>(rscale_val_);
    pdp1_cfg.enable_h2c_ = static_cast<uint8_t>(enable_h2c_);
    pdp1_cfg.enable_bw_  = static_cast<uint8_t>(enable_bw_);
}

// MfuPdp1Conf4Instruction3.cpp  @0x4252c0
MfuPdp1Conf4Instruction::~MfuPdp1Conf4Instruction()
{
}

// ---- MfuPdp1Conf_deqInstruction ----
// Simulator65.cpp  @0x419fd0
template <>
MfuPdp1Conf_deqInstruction Simulator::InstParser<MfuPdp1Conf_deqInstruction, 32>(uint8_t ** pc)
{
    MfuPdp1Conf_deqInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rscale_ = field(raw, 12, 5);
    inst.rbias_ = field(raw, 17, 5);
    inst.cfg_ = static_cast<uint16_t>(field(raw, 22, 2) | (field(raw, 24, 5) << 8));
    inst.reserved_29_ = field(raw, 29, 3);
    inst.rscale_val_ = _G.gp_reg[inst.rscale_];
    inst.rbias_val_ = _G.gp_reg[inst.rbias_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuPdp1Conf_deqInstruction1.cpp  @0x41df30
void MfuPdp1Conf_deqInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuPdp1Conf_deqInstruction2.cpp  @0x424830
void MfuPdp1Conf_deqInstruction::operation()
{
    // verified against asm @0x424830: dword store at PDP1+0xf0 ({zero:scale}), halfword at +0xf4 (src_dtype | dequant_flag << 8)
    Pdp1Config &pdp1_cfg = PDP1::GetPDP1()->cfg_;
    pdp1_cfg.dequant_scale_ = static_cast<uint16_t>(rscale_val_);
    pdp1_cfg.dequant_zero_  = static_cast<uint16_t>(rbias_val_);
    const uint16_t dtype_cfg = static_cast<uint16_t>(cfg_);
    std::memcpy(&pdp1_cfg.src_dtype_, &dtype_cfg, sizeof dtype_cfg);
}

// MfuPdp1Conf_deqInstruction3.cpp  @0x425270
MfuPdp1Conf_deqInstruction::~MfuPdp1Conf_deqInstruction()
{
}

// ---- MfuPdp1Conf_quantInstruction ----
// Simulator66.cpp  @0x41a230
template <>
MfuPdp1Conf_quantInstruction Simulator::InstParser<MfuPdp1Conf_quantInstruction, 32>(uint8_t ** pc)
{
    MfuPdp1Conf_quantInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rscale_ = field(raw, 12, 5);
    inst.rbias_ = field(raw, 17, 5);
    inst.cfg_ = static_cast<uint16_t>(field(raw, 22, 2) | (field(raw, 24, 5) << 8));
    inst.reserved_29_ = field(raw, 29, 3);
    inst.rscale_val_ = _G.gp_reg[inst.rscale_];
    inst.rbias_val_ = _G.gp_reg[inst.rbias_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuPdp1Conf_quantInstruction1.cpp  @0x41df40
void MfuPdp1Conf_quantInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuPdp1Conf_quantInstruction2.cpp  @0x4246a0
void MfuPdp1Conf_quantInstruction::operation()
{
    // verified against asm @0x4246a0: three u16 stores at PDP1+0xf6/+0xf8/+0xfa
    Pdp1Config &pdp1_cfg = PDP1::GetPDP1()->cfg_;
    pdp1_cfg.quant_scale_ = static_cast<uint16_t>(rscale_val_);
    pdp1_cfg.quant_zero_  = static_cast<uint16_t>(rbias_val_);
    const uint16_t dtype_cfg = static_cast<uint16_t>(cfg_);
    std::memcpy(&pdp1_cfg.dst_dtype_, &dtype_cfg, sizeof dtype_cfg);   // dst_dtype | quant_flag << 8
}

// MfuPdp1Conf_quantInstruction3.cpp  @0x425220
MfuPdp1Conf_quantInstruction::~MfuPdp1Conf_quantInstruction()
{
}
