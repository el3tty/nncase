#pragma once
// Reconstructed from IDA/Hex-Rays output (inferred declarations).
// asm: mfu_pdp1_compute raddr_d, raddr_s, rshape
// asm: mfu_pdp1_conf1 stride_w, stride_h, rstride_s, funct2, rstride_d
// asm: mfu_pdp1_conf2 rcount_w, rcount_h, rpe_h, rpe_last_h
// asm: mfu_pdp1_conf3 rpe_channels, rpe_last_channels, rpad_value, sspad
// asm: mfu_pdp1_conf4 rwindow_w, rwindow_h, rscale, enable_h2c, enable_bw
// asm: mfu_pdp1_conf_deq rscale, rbias, quant_type
// asm: mfu_pdp1_conf_quant rscale, rbias, quant_type
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

// ---- MfuPdp1ComputeInstruction ----
struct MfuPdp1ComputeInstruction : public KInstruction {
    uint8_t raddr_d_;  // +49 raw[11:7] register: destination address
    uint8_t raddr_s_;  // +50 raw[16:12] register: source address
    uint8_t rshape_;  // +51 raw[19:17] shape register index
    uint16_t reserved_20_;  // +52 raw[31:20] reserved, unused
    uint32_t raddr_d_val_;  // +56 g_gp_reg[raddr_d]
    uint32_t raddr_s_val_;  // +60 g_gp_reg[raddr_s]
    uint64_t rshape_val_;  // +64 g_shape_reg[rshape]
    void get_next_pc() override;
    void operation() override;
    ~MfuPdp1ComputeInstruction() override;
};
template <> MfuPdp1ComputeInstruction Simulator::InstParser<MfuPdp1ComputeInstruction, 32>(unsigned char **pc);

// ---- MfuPdp1Conf1Instruction ----
struct MfuPdp1Conf1Instruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t stride_w_;  // +50 raw[16:12] horizontal stride (immediate, not a register)
    uint8_t stride_h_;  // +51 raw[21:17] vertical stride (immediate, not a register)
    uint8_t rstride_s_;  // +52 raw[24:22] source stride shape register index
    uint8_t funct2_;  // +53 raw[26:25] pooling op select (max, ...)
    uint8_t rstride_d_;  // +54 raw[29:27] destination stride shape register index
    uint8_t reserved_30_;  // +55 raw[31:30] reserved, unused
    uint64_t rstride_s_val_;  // +56 g_shape_reg[rstride_s]
    uint64_t rstride_d_val_;  // +64 g_shape_reg[rstride_d]
    void get_next_pc() override;
    void operation() override;
    ~MfuPdp1Conf1Instruction() override;
};
template <> MfuPdp1Conf1Instruction Simulator::InstParser<MfuPdp1Conf1Instruction, 32>(unsigned char **pc);

// ---- MfuPdp1Conf2Instruction ----
struct MfuPdp1Conf2Instruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rcount_w_;  // +50 raw[16:12] register: window count W
    uint8_t rcount_h_;  // +51 raw[21:17] register: window count H
    uint8_t rpe_h_;  // +52 raw[26:22] register: PE rows (unused by operation)
    uint8_t rpe_last_h_;  // +53 raw[31:27] register: PE rows of last tile (unused by operation)
    uint32_t rcount_w_val_;  // +56 g_gp_reg[rcount_w]
    uint32_t rcount_h_val_;  // +60 g_gp_reg[rcount_h]
    uint32_t rpe_h_val_;  // +64 g_gp_reg[rpe_h] (not used by operation)
    uint32_t rpe_last_h_val_;  // +68 g_gp_reg[rpe_last_h] (not used by operation)
    void get_next_pc() override;
    void operation() override;
    ~MfuPdp1Conf2Instruction() override;
};
template <> MfuPdp1Conf2Instruction Simulator::InstParser<MfuPdp1Conf2Instruction, 32>(unsigned char **pc);

// ---- MfuPdp1Conf3Instruction ----
struct MfuPdp1Conf3Instruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rpe_channels_;  // +50 raw[16:12] register: PE channels
    uint8_t rpe_last_channels_;  // +51 raw[21:17] register: PE channels of last tile
    uint8_t rpad_value_;  // +52 raw[26:22] register: pad value
    uint8_t sspad_;  // +53 raw[29:27] pad shape register index
    uint8_t reserved_30_;  // +54 raw[31:30] reserved, unused
    uint32_t rpe_channels_val_;  // +56 g_gp_reg[rpe_channels]
    uint32_t rpe_last_channels_val_;  // +60 g_gp_reg[rpe_last_channels]
    uint32_t rpad_value_val_;  // +64 g_gp_reg[rpad_value]
    uint64_t sspad_val_;  // +72 g_shape_reg[sspad]
    void get_next_pc() override;
    void operation() override;
    ~MfuPdp1Conf3Instruction() override;
};
template <> MfuPdp1Conf3Instruction Simulator::InstParser<MfuPdp1Conf3Instruction, 32>(unsigned char **pc);

// ---- MfuPdp1Conf4Instruction ----
struct MfuPdp1Conf4Instruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rwindow_w_;  // +50 raw[16:12] register: window width
    uint8_t rwindow_h_;  // +51 raw[21:17] register: window height
    uint8_t rscale_;  // +52 raw[26:22] register: scale
    uint8_t enable_h2c_;  // +53 raw[27] enable H-to-C
    uint8_t enable_bw_;  // +54 raw[28] enable BW
    uint8_t reserved_29_;  // +55 raw[31:29] reserved, unused
    uint32_t rwindow_w_val_;  // +56 g_gp_reg[rwindow_w]
    uint32_t rwindow_h_val_;  // +60 g_gp_reg[rwindow_h]
    uint32_t rscale_val_;  // +64 g_gp_reg[rscale]
    void get_next_pc() override;
    void operation() override;
    ~MfuPdp1Conf4Instruction() override;
};
template <> MfuPdp1Conf4Instruction Simulator::InstParser<MfuPdp1Conf4Instruction, 32>(unsigned char **pc);

// ---- MfuPdp1Conf_deqInstruction ----
struct MfuPdp1Conf_deqInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rscale_;  // +50 raw[16:12] register: dequant scale
    uint8_t rbias_;  // +51 raw[21:17] register: dequant bias
    uint16_t cfg_;  // +52 raw[28:22] 16-bit config: quant_type = raw[23:22] (low byte), raw[28:24] has no operand (high byte)
    uint8_t reserved_29_;  // +54 raw[31:29] reserved, unused
    uint32_t rscale_val_;  // +56 g_gp_reg[rscale]
    uint32_t rbias_val_;  // +60 g_gp_reg[rbias]
    void get_next_pc() override;
    void operation() override;
    ~MfuPdp1Conf_deqInstruction() override;
};
template <> MfuPdp1Conf_deqInstruction Simulator::InstParser<MfuPdp1Conf_deqInstruction, 32>(unsigned char **pc);

// ---- MfuPdp1Conf_quantInstruction ----
struct MfuPdp1Conf_quantInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rscale_;  // +50 raw[16:12] register: quant scale
    uint8_t rbias_;  // +51 raw[21:17] register: quant bias
    uint16_t cfg_;  // +52 raw[28:22] 16-bit config: quant_type = raw[23:22] (low byte), raw[28:24] has no operand (high byte)
    uint8_t reserved_29_;  // +54 raw[31:29] reserved, unused
    uint32_t rscale_val_;  // +56 g_gp_reg[rscale]
    uint32_t rbias_val_;  // +60 g_gp_reg[rbias]
    void get_next_pc() override;
    void operation() override;
    ~MfuPdp1Conf_quantInstruction() override;
};
template <> MfuPdp1Conf_quantInstruction Simulator::InstParser<MfuPdp1Conf_quantInstruction, 32>(unsigned char **pc);
