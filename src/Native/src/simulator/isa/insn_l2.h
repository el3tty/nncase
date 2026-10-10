#pragma once
// Lifted from IDA/Hex-Rays output.
// L2 load (DDR -> GLB) configuration: two packed shapes (e.g. source / destination) and two mode bytes.
// asm: l2_load_conf rstride_d, rstride_s, l2_datatype, ddr_datatype
// L2 load: copy a tile from DDR to GLB (uses the shapes configured by L2LoadConf).
// asm: l2_load raddr_d, raddr_s, rshape
// L2 weight load (DDR -> GLB) configuration.
// asm: l2_load_w_conf rlen_compressed, rlen_decompressed, l2_datatype, ddr_datatype, enable_decompress
// L2 weight load: copy weights from DDR to GLB (uses the configuration from L2LoadWConf).
// asm: l2_load_w raddr_d, raddr_s, rvalid_c_num
// L2 store (GLB -> DDR) configuration: two packed shapes and a 16-bit mode word.
// asm: l2_store_conf rstride_d, rstride_s, l2_datatype, ddr_datatype
// L2 store: copy a tile from GLB to DDR (uses the shapes configured by L2StoreConf).
// asm: l2_store raddr_d, raddr_s, rshape
#include <cstddef>
#include <cstdint>
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- L2LoadConfInstruction ----
struct L2LoadConfInstruction : public KInstruction {
    uint8_t rstride_d_;  // +49 raw[9:7] destination (GLB) stride shape register index
    uint8_t rstride_s_;  // +50 raw[12:10] source (DDR) stride shape register index
    uint8_t l2_datatype_;  // +51 raw[14:13] GLB element datatype, stored at L2Load+80
    uint8_t ddr_datatype_;  // +52 raw[17:15] DDR element datatype, stored at L2Load+81
    uint16_t reserved_18_;  // +54 raw[31:18] reserved, unused
    uint64_t rstride_d_val_;  // +56  _G.shape_reg[rstride_d]: dim0 = [47:32], dim1 = [31:16], dim2 = [15:0]
    uint64_t rstride_s_val_;  // +64  _G.shape_reg[rstride_s]
    void get_next_pc() override;
    void operation() override;
    ~L2LoadConfInstruction() override;
};
template <> L2LoadConfInstruction Simulator::InstParser<L2LoadConfInstruction, 32>(unsigned char **pc);

// ---- L2LoadInstruction ----
struct L2LoadInstruction : public KInstruction {
    uint8_t raddr_d_;  // +49 raw[11:7] register: GLB destination address
    uint8_t raddr_s_;  // +50 raw[16:12] register: DDR source offset
    uint8_t rshape_;  // +51 raw[19:17] shape register index
    uint16_t reserved_20_;  // +52 raw[31:20] reserved, unused
    uint32_t raddr_d_val_;  // +56  _G.gp_reg[raddr_d]  bank[31:28] | offset[27:0]
    uint32_t raddr_s_val_;  // +60  _G.gp_reg[raddr_s]
    uint64_t rshape_val_;  // +64  _G.shape_reg[rshape]: four 16-bit dims, dim0 = [63:48] ... dim3 = [15:0]
    uint32_t raddr_d_mmu_addr_;  // +72  MMU-translated GLB address (offset + 32 * MMU item)
    void get_next_pc() override;
    void operation() override;
    ~L2LoadInstruction() override;
};
template <> L2LoadInstruction Simulator::InstParser<L2LoadInstruction, 32>(unsigned char **pc);

// ---- L2LoadWConfInstruction ----
struct L2LoadWConfInstruction : public KInstruction {
    uint8_t rlen_compressed_;  // +49 raw[11:7] register: compressed length (low half of wconf_val)
    uint8_t rlen_decompressed_;  // +50 raw[16:12] register: decompressed length (high half of wconf_val)
    uint8_t l2_datatype_;  // +51 raw[18:17] GLB element datatype, stored at L2Load+82
    uint8_t ddr_datatype_;  // +52 raw[21:19] DDR element datatype, stored at L2Load+83
    uint8_t enable_decompress_;  // +53 raw[22:22] decompress enable, stored at L2Load+56
    uint16_t reserved_23_;  // +54 raw[31:23] reserved, unused
    uint64_t wconf_val_;     // +56  _G.gp_reg[rs_hi] << 32 | _G.gp_reg[rs_lo]; stored at L2Load+48
    void get_next_pc() override;
    void operation() override;
    ~L2LoadWConfInstruction() override;
};
template <> L2LoadWConfInstruction Simulator::InstParser<L2LoadWConfInstruction, 32>(unsigned char **pc);

// ---- L2LoadWInstruction ----
struct L2LoadWInstruction : public KInstruction {
    uint8_t raddr_d_;  // +49 raw[11:7] register: GLB destination address
    uint8_t raddr_s_;  // +50 raw[16:12] register: DDR source offset
    uint8_t rvalid_c_num_;  // +51 raw[21:17] register: number of valid channels, stored at L2Load+60
    uint16_t reserved_22_;  // +52 raw[31:22] reserved, unused
    uint32_t raddr_d_val_;  // +56  _G.gp_reg[raddr_d]  bank[31:28] | offset[27:0]
    uint32_t raddr_s_val_;  // +60  _G.gp_reg[raddr_s]
    uint32_t rvalid_c_num_val_;  // +64  _G.gp_reg[rvalid_c_num]
    uint32_t raddr_d_mmu_addr_;  // +68  MMU-translated GLB address (offset + 32 * MMU item)
    void get_next_pc() override;
    void operation() override;
    ~L2LoadWInstruction() override;
};
template <> L2LoadWInstruction Simulator::InstParser<L2LoadWInstruction, 32>(unsigned char **pc);

// ---- L2StoreConfInstruction ----
struct L2StoreConfInstruction : public KInstruction {
    uint8_t rstride_d_;  // +49 raw[9:7] destination (DDR) stride shape register index
    uint8_t rstride_s_;  // +50 raw[12:10] source (GLB) stride shape register index
    uint8_t l2_datatype_;  // +51 raw[14:13] GLB element datatype (low byte of the mode word at L2Store+32)
    uint8_t ddr_datatype_;  // +52 raw[17:15] DDR element datatype (high byte of the mode word)
    uint16_t reserved_18_;  // +54 raw[31:18] reserved, unused
    uint64_t rstride_d_val_;  // +56  _G.shape_reg[rstride_d]: dim0 = [47:32], dim1 = [31:16], dim2 = [15:0]
    uint64_t rstride_s_val_;  // +64  _G.shape_reg[rstride_s]
    void get_next_pc() override;
    void operation() override;
    ~L2StoreConfInstruction() override;
};
template <> L2StoreConfInstruction Simulator::InstParser<L2StoreConfInstruction, 32>(unsigned char **pc);

// ---- L2StoreInstruction ----
struct L2StoreInstruction : public KInstruction {
    uint8_t raddr_d_;  // +49 raw[11:7] register: DDR destination offset
    uint8_t raddr_s_;  // +50 raw[16:12] register: GLB source address
    uint8_t rshape_;  // +51 raw[19:17] shape register index
    uint16_t reserved_20_;  // +52 raw[31:20] reserved, unused
    uint32_t raddr_d_val_;  // +56  _G.gp_reg[raddr_d]
    uint32_t raddr_s_val_;  // +60  _G.gp_reg[raddr_s]  bank[31:28] | offset[27:0]
    uint64_t rshape_val_;  // +64  _G.shape_reg[rshape]: four 16-bit dims, dim0 = [63:48] ... dim3 = [15:0]
    uint32_t raddr_s_mmu_addr_;  // +72  MMU-translated GLB address (offset + 32 * MMU item)
    void get_next_pc() override;
    void operation() override;
    ~L2StoreInstruction() override;
};
template <> L2StoreInstruction Simulator::InstParser<L2StoreInstruction, 32>(unsigned char **pc);
