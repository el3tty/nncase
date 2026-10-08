#pragma once
// Lifted from IDA/Hex-Rays output; field names are inferred.
// asm: end rs
// asm: (not in compiler dumps) fence ...
// asm: (not in compiler dumps) fence_i ...
// asm: (not in compiler dumps) intr ...
// asm: ccr_clr ccr   (ccr [11:7], verified vs main() decoder and TIR)
// asm: ccr_decl rnum
// asm: ccr_set ccr, value   (ccr [11:7], value [15:12], verified vs main() decoder and TIR)
// asm: mmu_conf rstart, rdepth, mmu_id
// asm: (not in compiler dumps) mmu_setid ...
// asm: ss_pack_shape rn, rc, rh, rw, rss
// asm: ss_pack_stride rn, rc, rh, rss
// Lifted from IDA/Hex-Rays output.
// Write an AI2D configuration register ("external register write").
// reg_addr is a byte address in the AI2D register file (reg_addr >> 2 selects the register);
// writing the register at byte address 140 (index 35) additionally starts the AI2D (ai2d_proc()).
// Decoder: InstParser<...,32> below (inlined into main() in the binary).
// asm: (not in compiler dumps) extrw ...
// Write an AI2D configuration register, restricted variant ("extra" register write): only the first eight
// registers (byte addresses 0..31, the same ones ExtrwInstruction maps to AI2D+136..+172) are handled.
// asm: (not in compiler dumps) extraw ...
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

// ---- EndInstruction ----
struct EndInstruction : public KInstruction {
    uint8_t rs_;  // +49 raw[11:7] register (always x0 in the dumps)
    uint8_t reserved_12_;  // +50 raw[15:12] reserved, decoded but unused
    uint32_t rs_val_;  // +52 g_gp_reg[rs] sampled at decode time
    void get_next_pc() override;
    ~EndInstruction() override;
};
template <> EndInstruction Simulator::InstParser<EndInstruction, 16>(unsigned char **pc);

// ---- FenceInstruction ----
struct FenceInstruction : public KInstruction {
    uint16_t reserved0_;  // +50 raw[15:7] 9-bit field
    void get_next_pc() override;
    ~FenceInstruction() override;
};
template <> FenceInstruction Simulator::InstParser<FenceInstruction, 16>(unsigned char **pc);

// ---- FenceIInstruction ----
struct FenceIInstruction : public KInstruction {
    uint16_t reserved0_;  // +50 raw[15:7] 9-bit field
    void get_next_pc() override;
    ~FenceIInstruction() override;
};
template <> FenceIInstruction Simulator::InstParser<FenceIInstruction, 16>(unsigned char **pc);

// ---- IntrInstruction ----
struct IntrInstruction : public KInstruction {
    uint8_t rs_;  // +49 raw[11:7] register index
    uint8_t reserved0_;  // +50 raw[15:12] 4-bit field
    uint32_t value_;  // +52 g_gp_reg[reg] sampled at decode time
    void get_next_pc() override;
    ~IntrInstruction() override;
};
template <> IntrInstruction Simulator::InstParser<IntrInstruction, 16>(unsigned char **pc);

// ---- CcrClrInstruction ----
struct CcrClrInstruction : public KInstruction {
    uint8_t ccr_;  // +49 raw[11:7] CCR index (compiler range 0..13)
    uint8_t reserved_12_;  // +50 raw[15:12] reserved
    void get_next_pc() override;
    ~CcrClrInstruction() override;
};
template <> CcrClrInstruction Simulator::InstParser<CcrClrInstruction, 16>(unsigned char **pc);

// ---- CcrDeclInstruction ----
struct CcrDeclInstruction : public KInstruction {
    uint8_t rnum_;  // +49 raw[11:7] register holding the CCR number
    uint8_t reserved_12_;  // +50 raw[15:12] reserved, decoded but unused
    uint32_t rnum_val_;  // +52 g_gp_reg[rnum] sampled at decode time
    void get_next_pc() override;
    ~CcrDeclInstruction() override;
};
template <> CcrDeclInstruction Simulator::InstParser<CcrDeclInstruction, 16>(unsigned char **pc);

// ---- CcrSetInstruction ----
struct CcrSetInstruction : public KInstruction {
    uint8_t ccr_;  // +49 raw[11:7] CCR index (compiler range 0..13)
    uint8_t value_;  // +50 raw[15:12] value (compiler range 1..2)
    void get_next_pc() override;
    ~CcrSetInstruction() override;
};
template <> CcrSetInstruction Simulator::InstParser<CcrSetInstruction, 16>(unsigned char **pc);

// ---- MmuConfInstruction ----
struct MmuConfInstruction : public KInstruction {
    uint8_t rstart_;  // +49 raw[11:7] register holding the GLB segment start
    uint8_t rdepth_;  // +50 raw[16:12] register holding the GLB segment depth
    uint8_t mmu_id_;  // +51 raw[20:17] MMU/GLB segment index (compiler range 0..5)
    uint16_t reserved_21_;  // +52 raw[31:21] reserved, decoded but unused
    uint32_t rstart_val_;  // +56 segment rstart_val (value of rstart)
    uint32_t rdepth_val_;  // +60 segment rdepth_val (value of rdepth)
    void get_next_pc() override;
    void operation() override;
    ~MmuConfInstruction() override;
};
template <> MmuConfInstruction Simulator::InstParser<MmuConfInstruction, 32>(unsigned char **pc);

// ---- MmuSetidInstruction ----
struct MmuSetidInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] register whose bits [31:28] are replaced
    uint8_t mmu_id_;  // +50 raw[15:12] new 4-bit MMU/GLB segment id
    uint32_t result_;  // +52 new register value (trace)
    void get_next_pc() override;
    void operation() override;
    void parser_operation();
    ~MmuSetidInstruction() override;
};
template <> MmuSetidInstruction Simulator::InstParser<MmuSetidInstruction, 16>(unsigned char **pc);

// ---- SsPackShapeInstruction ----
struct SsPackShapeInstruction : public KInstruction {
    uint8_t rn_;  // +49 raw[11:7] register: batch (N) dim, packed into [63:48]
    uint8_t rc_;  // +50 raw[16:12] register: channel (C) dim, packed into [47:32]
    uint8_t rh_;  // +51 raw[21:17] register: height (H) dim, packed into [31:16]
    uint8_t rw_;  // +52 raw[26:22] register: width (W) dim, packed into [15:0]
    uint8_t rss_;  // +53 raw[29:27] destination shape register index
    uint8_t reserved_30_;  // +54 raw[31:30] reserved, decoded but unused
    uint32_t rn_val_;  // +56 value of rn
    uint32_t rc_val_;  // +60 value of rc
    uint32_t rh_val_;  // +64 value of rh
    uint32_t rw_val_;  // +68 value of rw
    uint64_t packed_;  // +72 packed shape register value
    void get_next_pc() override;
    void parser_operation();
    ~SsPackShapeInstruction() override;
};
template <> SsPackShapeInstruction Simulator::InstParser<SsPackShapeInstruction, 32>(unsigned char **pc);

// ---- SsPackStrideInstruction ----
struct SsPackStrideInstruction : public KInstruction {
    uint8_t rn_;  // +49 raw[11:7] register: N stride, packed into [47:32]
    uint8_t rc_;  // +50 raw[16:12] register: C stride, packed into [31:16]
    uint8_t rh_;  // +51 raw[21:17] register: H stride, packed into [15:0]
    uint8_t reserved_22_;  // +52 raw[26:22] reserved, decoded but unused
    uint8_t rss_;  // +53 raw[29:27] destination shape (stride) register index
    uint8_t reserved_30_;  // +54 raw[31:30] reserved, decoded but unused
    uint32_t rn_val_;  // +56 value of rn
    uint32_t rc_val_;  // +60 value of rc
    uint32_t rh_val_;  // +64 value of rh
    uint64_t packed_;  // +72 packed stride register value
    void get_next_pc() override;
    void parser_operation();
    ~SsPackStrideInstruction() override;
};
template <> SsPackStrideInstruction Simulator::InstParser<SsPackStrideInstruction, 32>(unsigned char **pc);

// ---- ExtrwInstruction ----
struct ExtrwInstruction : public KInstruction {
    uint16_t extrd_;  // +50 raw[16:7] AI2D register byte address
    uint8_t rs_;  // +52 raw[21:17] register holding the value
    uint16_t imm_;  // +54 raw[31:22] decoded but unused
    uint32_t reg_value_;     // +56  value to write
    void get_next_pc() override;
    void operation() override;
    ~ExtrwInstruction() override;
};
template <> ExtrwInstruction Simulator::InstParser<ExtrwInstruction, 32>(unsigned char **pc);

// ---- ExtrawInstruction ----
struct ExtrawInstruction : public KInstruction {
    uint16_t extrd_;  // +50 raw[16:7] AI2D register byte address
    uint8_t rs_;  // +52 raw[21:17] register holding the value
    uint16_t imm_;  // +54 raw[31:22] decoded but unused
    uint32_t reg_value_;     // +56  value to write
    void get_next_pc() override;
    void operation() override;
    ~ExtrawInstruction() override;
};
template <> ExtrawInstruction Simulator::InstParser<ExtrawInstruction, 32>(unsigned char **pc);
