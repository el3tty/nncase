// Lifted from IDA/Hex-Rays output (K230 NPU C-model simulator).
// Member names describe the decoded instruction fields; byte offsets are given in the header.
#include "isa/insn_mfu_act1.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/meshnet.h"
#include "engines/simulator.h"

// ---- MfuAct1ComputeInstruction ----
namespace {
// Returns `width` bits of `raw` starting at bit `lo`.
inline uint32_t field(uint64_t raw, unsigned lo, unsigned width)
{
    return static_cast<uint32_t>((raw >> lo) & ((1ull << width) - 1));
}
// Virtual-to-physical style translation through the MMU table: 28-bit offset + 32 * entry[addr >> 28].
inline uint32_t mmu_translate(uint32_t addr)
{
    return (addr & 0xFFFFFFF) + 32 * MMU_MMUItem[2 * (addr >> 28)];
}
}  // namespace

// Simulator75.cpp  @0x41b7c0
template <>
MfuAct1ComputeInstruction Simulator::InstParser<MfuAct1ComputeInstruction, 32>(uint8_t ** pc)
{
    MfuAct1ComputeInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.raddr_d1_ = field(raw, 7, 5);
    inst.raddr_s1_ = field(raw, 12, 5);
    inst.raddr_s2_ = field(raw, 17, 5);
    inst.raddr_arg_ = field(raw, 22, 5);
    inst.reserved_27_ = field(raw, 27, 5);
    inst.raddr_d1_val_ = g_gp_reg[inst.raddr_d1_];
    inst.raddr_s1_val_ = g_gp_reg[inst.raddr_s1_];
    inst.raddr_s2_val_ = g_gp_reg[inst.raddr_s2_];
    inst.raddr_arg_val_ = g_gp_reg[inst.raddr_arg_];
    inst.raddr_s1_mmu_addr_ = mmu_translate(inst.raddr_s1_val_);
    inst.raddr_s2_mmu_addr_ = mmu_translate(inst.raddr_s2_val_);
    inst.raddr_d1_mmu_addr_ = mmu_translate(inst.raddr_d1_val_);
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1ComputeInstruction1.cpp  @0x41dfd0
void MfuAct1ComputeInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1ComputeInstruction2.cpp  @0x421cc0
void MfuAct1ComputeInstruction::operation()
{
    if (flag_)
        debug_flag = 1;
    // The MFU Act1 engine takes all its parameters from the Act1Conf* registers set earlier.
    reinterpret_cast<MeshNet *>(MeshNet::GetMeshNet())->MfuAct1();
}

// MfuAct1ComputeInstruction3.cpp  @0x424f50
MfuAct1ComputeInstruction::~MfuAct1ComputeInstruction()
{
}

// ---- MfuAct1Conf_deqInstruction ----
// Simulator72.cpp  @0x41b100
template <>
MfuAct1Conf_deqInstruction Simulator::InstParser<MfuAct1Conf_deqInstruction, 32>(uint8_t ** pc)
{
    MfuAct1Conf_deqInstruction inst;
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
    inst.quant_type_ = field(raw, 22, 2);
    inst.sid_ = field(raw, 24, 1);
    inst.rshift_bits_ = field(raw, 25, 5);
    inst.reserved_30_ = field(raw, 30, 2);
    inst.rscale_val_ = g_gp_reg[inst.rscale_];
    inst.rbias_val_ = g_gp_reg[inst.rbias_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1Conf_deqInstruction1.cpp  @0x41dfa0
void MfuAct1Conf_deqInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1Conf_deqInstruction2.cpp  @0x421000
void MfuAct1Conf_deqInstruction::operation()
{
    // verified against asm @0x421000: the "globals" 0x54BAB8..0x54BAC2 are MeshNet fields (singleton at 0x54AA60).
    // sid == 0: +0x1058 dword = (rbias << 16) | (uint16)rscale (a1_s1_scale / a1_s1_zero), +0x105c = type | shift << 8;
    // sid != 0: +0x105e = (uint16)rscale, +0x1060 = (uint16)rbias, +0x1062 = type | shift << 8 (a1_s2_*).
    MeshNet * const mn = MeshNet::GetMeshNet();
    if (sid_) {
        mn->a1_s2_scale_ = static_cast<uint16_t>(rscale_val_);
        mn->a1_s2_zero_ = static_cast<uint16_t>(rbias_val_);
        mn->a1_s2_type_ = quant_type_;
        mn->a1_s2_shift_ = rshift_bits_;
    } else {
        mn->a1_s1_scale_ = static_cast<uint16_t>(rscale_val_);
        mn->a1_s1_zero_ = static_cast<uint16_t>(rbias_val_);
        mn->a1_s1_type_ = quant_type_;
        mn->a1_s1_shift_ = rshift_bits_;
    }
}

// MfuAct1Conf_deqInstruction3.cpp  @0x425040
MfuAct1Conf_deqInstruction::~MfuAct1Conf_deqInstruction()
{
}

// ---- MfuAct1Conf_quantInstruction ----
// Simulator73.cpp  @0x41b360
template <>
MfuAct1Conf_quantInstruction Simulator::InstParser<MfuAct1Conf_quantInstruction, 32>(uint8_t ** pc)
{
    MfuAct1Conf_quantInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.quant_cfg_ = static_cast<uint16_t>(field(raw, 12, 2) | (field(raw, 14, 5) << 8));
    inst.reserved_19_ = field(raw, 19, 13);  // verified against asm: wider immediate (unused by operation)
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1Conf_quantInstruction1.cpp  @0x41dfb0
void MfuAct1Conf_quantInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1Conf_quantInstruction2.cpp  @0x421780
void MfuAct1Conf_quantInstruction::operation()
{
    // verified against asm @0x421780: 16-bit store to MeshNet+0x1064 (= 0x54BAC4): a1_dst_type | a1_fit_shift << 8.
    MeshNet * const mn = MeshNet::GetMeshNet();
    mn->a1_dst_type_ = static_cast<uint8_t>(quant_cfg_ & 0xFF);
    mn->a1_fit_shift_ = static_cast<uint8_t>(quant_cfg_ >> 8);
}

// MfuAct1Conf_quantInstruction3.cpp  @0x424ff0
MfuAct1Conf_quantInstruction::~MfuAct1Conf_quantInstruction()
{
}

// ---- MfuAct1ConfDestInstruction ----

// Simulator71.cpp  @0x41ae90
template <>
MfuAct1ConfDestInstruction Simulator::InstParser<MfuAct1ConfDestInstruction, 32>(uint8_t ** pc)
{
    MfuAct1ConfDestInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rlen_ = field(raw, 12, 5);
    inst.rshape_ = field(raw, 17, 3);
    inst.reserved_20_ = field(raw, 20, 12);  // verified against asm: wider immediate (unused by operation)
    inst.rlen_val_ = g_gp_reg[inst.rlen_];
    inst.rshape_val_ = g_shape_reg[inst.rshape_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1ConfDestInstruction1.cpp  @0x41df90
void MfuAct1ConfDestInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1ConfDestInstruction2.cpp  @0x4215c0
void MfuAct1ConfDestInstruction::operation()
{
    // Act1 destination config (MeshNet +4172 / +4176).
    MeshNet * const mn = MeshNet::GetMeshNet();
    mn->a1_dst_len_ = rlen_val_;
    mn->a1_dst_dims_ = rshape_val_;
}

// MfuAct1ConfDestInstruction3.cpp  @0x425090
MfuAct1ConfDestInstruction::~MfuAct1ConfDestInstruction()
{
}

// ---- MfuAct1ConfInstruction ----
// Simulator74.cpp  @0x41b590
template <>
MfuAct1ConfInstruction Simulator::InstParser<MfuAct1ConfInstruction, 32>(uint8_t ** pc)
{
    MfuAct1ConfInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.funct4_ = field(raw, 12, 4);
    inst.is_by_channel_ = field(raw, 16, 1);
    inst.is_16_segments_ = field(raw, 17, 1);
    inst.reserved_18_ = field(raw, 18, 14);  // verified against asm: wider immediate (unused by operation)
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1ConfInstruction1.cpp  @0x41dfc0
void MfuAct1ConfInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1ConfInstruction2.cpp  @0x421af0
void MfuAct1ConfInstruction::operation()
{
    // verified against asm @0x421af0: the "globals" 0x54BAC6..0x54BAC8 are MeshNet fields (singleton at 0x54AA60,
    // +0x1066 / +0x1067 / +0x1068 = a1_op_mul / a1_per_channel / a1_use_mfu_fit), each a single byte store.
    MeshNet * const mn = MeshNet::GetMeshNet();
    mn->a1_per_channel_ = is_by_channel_ != 0;
    mn->a1_use_mfu_fit_ = is_16_segments_ != 0;
    mn->a1_op_mul_ = funct4_;
}

// MfuAct1ConfInstruction3.cpp  @0x424fa0
MfuAct1ConfInstruction::~MfuAct1ConfInstruction()
{
}

// ---- MfuAct1ConfSrc1Instruction ----
// Simulator69.cpp  @0x41a9b0
template <>
MfuAct1ConfSrc1Instruction Simulator::InstParser<MfuAct1ConfSrc1Instruction, 32>(uint8_t ** pc)
{
    MfuAct1ConfSrc1Instruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rslice_ = field(raw, 12, 5);
    inst.rright_repeats_ = field(raw, 17, 5);
    inst.rslice_repeats_ = field(raw, 22, 5);
    inst.sid_ = field(raw, 27, 1);
    inst.slice_loc_ = field(raw, 28, 1);
    inst.reserved_29_ = field(raw, 29, 3);
    inst.rslice_val_ = g_gp_reg[inst.rslice_];
    inst.rright_repeats_val_ = g_gp_reg[inst.rright_repeats_];
    inst.rslice_repeats_val_ = g_gp_reg[inst.rslice_repeats_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1ConfSrc1Instruction1.cpp  @0x41df70
void MfuAct1ConfSrc1Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1ConfSrc1Instruction2.cpp  @0x4220c0
void MfuAct1ConfSrc1Instruction::operation()
{
    // Act1 source-1 config: sid selects the source (0: src1 at +4112..+4120/+4136, 1: src2 at +4124..+4132/+4137).
    // verified against asm @0x4220c0: one qword store {slice_len, rpt_a} + dword rpt_b + byte no_l1_check per source.
    MeshNet * const mn = MeshNet::GetMeshNet();
    if (sid_) {
        mn->a1_s2_slice_len_ = rslice_val_;
        mn->a1_s2_rpt_a_ = rright_repeats_val_;
        mn->a1_s2_rpt_b_ = rslice_repeats_val_;
        mn->a1_s2_no_l1_check_ = slice_loc_;
    } else {
        mn->a1_s1_slice_len_ = rslice_val_;
        mn->a1_s1_rpt_a_ = rright_repeats_val_;
        mn->a1_s1_rpt_b_ = rslice_repeats_val_;
        mn->a1_s1_no_l1_check_ = slice_loc_;
    }
}

// MfuAct1ConfSrc1Instruction3.cpp  @0x425130
MfuAct1ConfSrc1Instruction::~MfuAct1ConfSrc1Instruction()
{
}

// ---- MfuAct1ConfSrc2Instruction ----
// Simulator70.cpp  @0x41ac20
template <>
MfuAct1ConfSrc2Instruction Simulator::InstParser<MfuAct1ConfSrc2Instruction, 32>(uint8_t ** pc)
{
    MfuAct1ConfSrc2Instruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rleft_repeats_ = field(raw, 12, 5);
    inst.rshape_ = field(raw, 17, 3);
    inst.sid_ = field(raw, 20, 1);
    inst.source_type_ = field(raw, 21, 1);
    inst.reserved_22_ = field(raw, 22, 10);  // verified against asm: wider immediate (unused by operation)
    inst.rleft_repeats_val_ = g_gp_reg[inst.rleft_repeats_];
    inst.rshape_val_ = g_shape_reg[inst.rshape_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1ConfSrc2Instruction1.cpp  @0x41df80
void MfuAct1ConfSrc2Instruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1ConfSrc2Instruction2.cpp  @0x421eb0
void MfuAct1ConfSrc2Instruction::operation()
{
    // Act1 source-2 config: sid selects the source (0: src1 at +4140/+4144/+4152, 1: src2 at +4156/+4160/+4168).
    MeshNet * const mn = MeshNet::GetMeshNet();
    if (sid_) {
        mn->a1_s2_rpt_c_ = rleft_repeats_val_;
        mn->a1_src2_dims_ = rshape_val_;
        mn->a1_src2_psum_ = source_type_;
    } else {
        mn->a1_s1_rpt_c_ = rleft_repeats_val_;
        mn->a1_src1_dims_ = rshape_val_;
        mn->a1_src1_psum_ = source_type_;
    }
}

// MfuAct1ConfSrc2Instruction3.cpp  @0x4250e0
MfuAct1ConfSrc2Instruction::~MfuAct1ConfSrc2Instruction()
{
}

// ---- MfuAct1ConfStrideInstruction ----
// Simulator68.cpp  @0x41a740
template <>
MfuAct1ConfStrideInstruction Simulator::InstParser<MfuAct1ConfStrideInstruction, 32>(uint8_t ** pc)
{
    MfuAct1ConfStrideInstruction inst;
    const uint64_t *word = reinterpret_cast<const uint64_t *>(*pc);
    // IDA loads a 64-bit word; only the low 32 bits (the instruction) are decoded.
    const uint64_t raw = *word;
    const uint32_t pc_abs = KPU_PC(word);

    inst.taken_ = 0;
    inst.flag_ = 0;
    inst.opcode_ = raw & 0x7F;
    // verified against asm: register field is raw[11:7] (5 bits)
    inst.funct5_ = field(raw, 7, 5);
    inst.rstride_s1_ = field(raw, 12, 3);
    inst.rstride_s2_ = field(raw, 15, 3);
    inst.rstride_d1_ = field(raw, 18, 3);
    inst.reserved_21_ = field(raw, 21, 11);  // verified against asm: wider immediate (unused by operation)
    inst.rstride_s1_val_ = g_shape_reg[inst.rstride_s1_];
    inst.rstride_s2_val_ = g_shape_reg[inst.rstride_s2_];
    inst.rstride_d1_val_ = g_shape_reg[inst.rstride_d1_];
    inst.info_ = 0x400000005LL;  // two u32: instruction type 5 / kind 4
    inst.pc_ = pc_abs;
    inst.pc_rel_ = pc_abs - start_pc_;
    *pc += 4;
    return inst;
}

// MfuAct1ConfStrideInstruction1.cpp  @0x41df60
void MfuAct1ConfStrideInstruction::get_next_pc()
{
    next_pc_ = pc_ + 4;
}

// MfuAct1ConfStrideInstruction2.cpp  @0x421930
void MfuAct1ConfStrideInstruction::operation()
{
    // Act1 destination pitches (MeshNet +4104).
    MeshNet * const mn = MeshNet::GetMeshNet();
    mn->a1_dst_chw_ = rstride_d1_val_;
}

// MfuAct1ConfStrideInstruction3.cpp  @0x425180
MfuAct1ConfStrideInstruction::~MfuAct1ConfStrideInstruction()
{
}
