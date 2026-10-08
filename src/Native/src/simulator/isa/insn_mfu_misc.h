#pragma once
// Reconstructed from IDA/Hex-Rays output (inferred declarations).
// asm: (not in compiler dumps) mfu_memcpy ...
// asm: (not in compiler dumps) mfu_memset ...
// asm: mfu_transpose raddr_d, raddr_s, rshape
// asm: mfu_transpose_conf rstride_d, rstride_s, l2_datatype, permute
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

// ---- MfuMemcpyInstruction ----
struct MfuMemcpyInstruction : public KInstruction {
    uint8_t raddr_d_;  // +49 raw[11:7] register
    uint8_t raddr_s_;  // +50 raw[16:12] register
    uint8_t rstride_d_;  // +51 raw[19:17] shape register index
    uint8_t rstride_s_;  // +52 raw[22:20] shape register index
    uint8_t rshape_;  // +53 raw[25:23] shape register index
    uint8_t reserved_26_;  // +54 raw[31:26] reserved, unused
    uint32_t raddr_d_val_;  // +56 g_gp_reg[rd]
    uint32_t raddr_s_val_;  // +60 g_gp_reg[rs1]
    uint64_t shape_a_;  // +64 g_shape_reg[shape_a_idx]
    uint64_t shape_b_;  // +72 g_shape_reg[shape_b_idx]
    uint64_t shape_c_;  // +80 g_shape_reg[shape_c_idx] (halfwords handed to the MFU)
    void get_next_pc() override;
    void operation() override;
    ~MfuMemcpyInstruction() override;
};
template <> MfuMemcpyInstruction Simulator::InstParser<MfuMemcpyInstruction, 32>(unsigned char **pc);

// ---- MfuMemsetInstruction ----
struct MfuMemsetInstruction : public KInstruction {
    uint8_t raddr_d_;  // +49 raw[11:7] register
    uint8_t rv_;  // +50 raw[16:12] register (fill value)
    uint8_t rstride_;  // +51 raw[19:17] shape register index
    uint8_t rshape_;  // +52 raw[22:20] shape register index
    uint8_t l2_datatype_;  // +53 raw[24:23] mode
    uint8_t reserved_25_;  // +54 raw[31:25] reserved, unused
    uint32_t raddr_d_val_;  // +56 g_gp_reg[rd]
    uint32_t rv_val_;  // +60 g_gp_reg[rs1]: fill value
    uint64_t shape_a_;  // +64 g_shape_reg[shape_a_idx]
    uint64_t shape_b_;  // +72 g_shape_reg[shape_b_idx] (halfwords handed to the MFU)
    uint32_t rd_addr_;  // +80 MMU-translated rd_val (decoded only)
    void get_next_pc() override;
    void operation() override;
    ~MfuMemsetInstruction() override;
};
template <> MfuMemsetInstruction Simulator::InstParser<MfuMemsetInstruction, 32>(unsigned char **pc);

// ---- MfuTransposeInstruction ----
struct MfuTransposeInstruction : public KInstruction {
    uint8_t raddr_d_;  // +49 raw[11:7] register: destination address
    uint8_t raddr_s_;  // +50 raw[16:12] register: source address
    uint8_t rshape_;  // +51 raw[19:17] shape register index
    uint16_t reserved_20_;  // +52 raw[31:20] reserved, unused
    uint32_t raddr_d_val_;  // +56 g_gp_reg[raddr_d]
    uint32_t raddr_s_val_;  // +60 g_gp_reg[raddr_s]
    uint64_t rshape_val_;  // +64 g_shape_reg[rshape]
    void get_next_pc() override;
    void operation() override;
    ~MfuTransposeInstruction() override;
};
template <> MfuTransposeInstruction Simulator::InstParser<MfuTransposeInstruction, 32>(unsigned char **pc);

// ---- MfuTransposeConfInstruction ----
struct MfuTransposeConfInstruction : public KInstruction {
    uint8_t funct5_;  // +49 raw[11:7] mfu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rstride_d_;  // +50 raw[14:12] destination stride shape register index (MFU trans_shape_dst)
    uint8_t rstride_s_;  // +51 raw[17:15] source stride shape register index (MFU trans_shape_src)
    uint16_t cfg_;  // +52 raw[24:18] 16-bit config: l2_datatype = raw[19:18] (low byte, trans_elem16), permute = raw[24:20] (high byte, trans_type)
    uint8_t reserved_25_;  // +54 raw[31:25] reserved, unused
    uint64_t rstride_d_val_;  // +56 g_shape_reg[rstride_d] (MFU trans_shape_dst)
    uint64_t rstride_s_val_;  // +64 g_shape_reg[rstride_s] (MFU trans_shape_src)
    void get_next_pc() override;
    void operation() override;
    ~MfuTransposeConfInstruction() override;
};
template <> MfuTransposeConfInstruction Simulator::InstParser<MfuTransposeConfInstruction, 32>(unsigned char **pc);
