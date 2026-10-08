#pragma once
// Reconstructed from IDA/Hex-Rays output (inferred declarations).
// asm: mfu_act1_conf_deq rscale, rbias, quant_type, sid
// asm: mfu_act1_conf_quant quant_type
// asm: mfu_act1_conf_dest rlen, rshape   (operation() treats the rlen register value as a destination address; low-medium confidence)
// asm: mfu_act1_conf funct4, is_by_channel, is_16_segments
// asm: mfu_act1_conf_src1 rslice, rright_repeats, rslice_repeats, sid, slice_loc   (operation() uses only the rslice_repeats register value, as the source address; low-medium confidence)
// asm: mfu_act1_conf_src2 rleft_repeats, rshape, sid, source_type   (operation() treats the rleft_repeats register value as a source address; low-medium confidence)
// asm: mfu_act1_conf_stride rstride_s1, rstride_s2, rstride_d1
#include <typeinfo>
#include <xmmintrin.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <deque>
#include <map>
#include <bitset>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstddef>
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- MfuAct1ComputeInstruction ----
extern uint8_t debug_flag;  // u8 at 0x5cc7b0, also declared in globals.h
// asm: mfu_act1_compute raddr_d1, raddr_s1, raddr_s2, raddr_arg
struct MfuAct1ComputeInstruction : public KInstruction {
    uint8_t raddr_d1_;  // +49 raw[11:7] register: destination address
    uint8_t raddr_s1_;  // +50 raw[16:12] register: source 1 address
    uint8_t raddr_s2_;  // +51 raw[21:17] register: source 2 address
    uint8_t raddr_arg_;  // +52 raw[26:22] register: argument address
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t raddr_d1_val_;  // +56 g_gp_reg[raddr_d1]
    uint32_t raddr_s1_val_;  // +60 g_gp_reg[raddr_s1]
    uint32_t raddr_s2_val_;  // +64 g_gp_reg[raddr_s2]
    uint32_t raddr_arg_val_;  // +68 g_gp_reg[raddr_arg]
    uint32_t raddr_s1_mmu_addr_;  // +72 MMU-translated raddr_s1_val (decoded only)
    uint32_t raddr_s2_mmu_addr_;  // +76 MMU-translated raddr_s2_val (decoded only)
    uint32_t raddr_d1_mmu_addr_;  // +80 MMU-translated raddr_d1_val (decoded only)
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1ComputeInstruction() override;
};
template <> MfuAct1ComputeInstruction Simulator::InstParser<MfuAct1ComputeInstruction, 32>(unsigned char **pc);

// ---- MfuAct1Conf_deqInstruction ----
struct MfuAct1Conf_deqInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rscale_;  // +50 raw[16:12] register: dequant scale
    uint8_t rbias_;  // +51 raw[21:17] register: dequant bias
    uint8_t quant_type_;  // +52 raw[23:22] dequant type (low byte of the 16-bit config)
    uint8_t sid_;  // +53 raw[24] source id: 0 -> source 1, 1 -> source 2 register bank
    uint8_t rshift_bits_;  // +54 raw[29:25] no operand in the dumps (shift, high byte of the 16-bit config)
    uint8_t reserved_30_;  // +55 raw[31:30] reserved, unused
    uint32_t rscale_val_;  // +56 g_gp_reg[rscale]
    uint32_t rbias_val_;  // +60 g_gp_reg[rbias]
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1Conf_deqInstruction() override;
};
template <> MfuAct1Conf_deqInstruction Simulator::InstParser<MfuAct1Conf_deqInstruction, 32>(unsigned char **pc);

// ---- MfuAct1Conf_quantInstruction ----
struct MfuAct1Conf_quantInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint16_t quant_cfg_;  // +50 raw[18:12] 16-bit config: quant_type = raw[13:12] (low byte), raw[18:14] has no operand (high byte, fit shift)
    uint32_t reserved_19_;  // +52 raw[31:19] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1Conf_quantInstruction() override;
};
template <> MfuAct1Conf_quantInstruction Simulator::InstParser<MfuAct1Conf_quantInstruction, 32>(unsigned char **pc);

// ---- MfuAct1ConfDestInstruction ----
struct MfuAct1ConfDestInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rlen_;  // +50 raw[16:12] register: length (stored as dest address, see note)
    uint8_t rshape_;  // +51 raw[19:17] shape register index
    uint16_t reserved_20_;  // +52 raw[31:20] reserved, unused
    uint32_t rlen_val_;  // +56 g_gp_reg[rlen]: destination address
    uint64_t rshape_val_;  // +64 g_shape_reg[rshape]: destination shape
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1ConfDestInstruction() override;
};
template <> MfuAct1ConfDestInstruction Simulator::InstParser<MfuAct1ConfDestInstruction, 32>(unsigned char **pc);

// ---- MfuAct1ConfInstruction ----
struct MfuAct1ConfInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t funct4_;  // +50 raw[15:12] ALU op select (add, ...) -> a1_op_mul
    uint8_t is_by_channel_;  // +51 raw[16] per-channel flag -> a1_per_channel
    uint8_t is_16_segments_;  // +52 raw[17] 16-segments flag -> a1_use_mfu_fit
    uint32_t reserved_18_;  // +56 raw[31:18] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1ConfInstruction() override;
};
template <> MfuAct1ConfInstruction Simulator::InstParser<MfuAct1ConfInstruction, 32>(unsigned char **pc);

// ---- MfuAct1ConfSrc1Instruction ----
struct MfuAct1ConfSrc1Instruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rslice_;  // +50 raw[16:12] register: slice (unused by operation)
    uint8_t rright_repeats_;  // +51 raw[21:17] register: right repeats (unused by operation)
    uint8_t rslice_repeats_;  // +52 raw[26:22] register: slice repeats (used as source address, see note)
    uint8_t sid_;  // +53 raw[27] source id: selects config bank 0/1
    uint8_t slice_loc_;  // +54 raw[28] slice location (l1/l2), stored next to the address
    uint8_t reserved_29_;  // +55 raw[31:29] reserved, unused
    uint32_t rslice_val_;  // +56 g_gp_reg[rslice] (not used by operation)
    uint32_t rright_repeats_val_;  // +60 g_gp_reg[rright_repeats] (not used by operation)
    uint32_t rslice_repeats_val_;  // +64 g_gp_reg[rslice_repeats]: source address
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1ConfSrc1Instruction() override;
};
template <> MfuAct1ConfSrc1Instruction Simulator::InstParser<MfuAct1ConfSrc1Instruction, 32>(unsigned char **pc);

// ---- MfuAct1ConfSrc2Instruction ----
struct MfuAct1ConfSrc2Instruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rleft_repeats_;  // +50 raw[16:12] register: left repeats (stored as source address, see note)
    uint8_t rshape_;  // +51 raw[19:17] shape register index
    uint8_t sid_;  // +52 raw[20] source id: selects config bank 0/1
    uint8_t source_type_;  // +53 raw[21] source type (l2/psum), stored with the shape
    uint16_t reserved_22_;  // +54 raw[31:22] reserved, unused
    uint32_t rleft_repeats_val_;  // +56 g_gp_reg[rleft_repeats]: source address
    uint64_t rshape_val_;  // +64 g_shape_reg[rshape]: source shape
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1ConfSrc2Instruction() override;
};
template <> MfuAct1ConfSrc2Instruction Simulator::InstParser<MfuAct1ConfSrc2Instruction, 32>(unsigned char **pc);

// ---- MfuAct1ConfStrideInstruction ----
struct MfuAct1ConfStrideInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rstride_s1_;  // +50 raw[14:12] source-1 stride shape register index
    uint8_t rstride_s2_;  // +51 raw[17:15] source-2 stride shape register index
    uint8_t rstride_d1_;  // +52 raw[20:18] destination stride shape register index
    uint16_t reserved_21_;  // +54 raw[31:21] reserved, unused
    uint64_t rstride_s1_val_;  // +56 g_shape_reg[rstride_s1] (not used by operation)
    uint64_t rstride_s2_val_;  // +64 g_shape_reg[rstride_s2] (not used by operation)
    uint64_t rstride_d1_val_;  // +72 g_shape_reg[rstride_d1]: stride written to the MeshNet
    void get_next_pc() override;
    void operation() override;
    ~MfuAct1ConfStrideInstruction() override;
};
template <> MfuAct1ConfStrideInstruction Simulator::InstParser<MfuAct1ConfStrideInstruction, 32>(unsigned char **pc);
