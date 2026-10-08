#pragma once
// Lifted from IDA/Hex-Rays output.
// DM configuration "broadcast" instruction (16-bit encoding, no operation() body in the binary).
// verified against asm @0x41d820: operation() is an empty function (just ret), so the fields have no consumer; they are named after their bit positions.
// asm: (not in compiler dumps) dm_conf_broadcast ...
// DM: configure the L1 load (packed shape in a shape register + a mode field).
// asm: dm_load_l1_conf tcu_id, pu_id, rstride_s, datatype, l1_type
// DM: load an L1 tile from GLB (see DmLoadL1ConfInstruction for the shape / mode setup).
// asm: dm_load_l1 tcu_id, pu_id, raddr_s, rhtoc_window, rshape, l1_type
// DM: second weight-load configuration (no operation() body in the binary).
// asm: dm_load_w_conf2 tcu_id, pu_id, rgroups, rgoc
// DM: weight-load dequantisation configuration.
// asm: dm_load_w_conf_deq tcu_id, pu_id, quant_type
// DM: weight-load configuration (two immediates and one register value).
// asm: dm_load_w_conf tcu_id, pu_id, kernel_h, kernel_w, rstride_oc
// DM: configure the output-feature (OF) store: packed shape and a mode field.
// asm: dm_store_of_conf tcu_id, pu_id, rstride_d, datatype
#include <cstddef>
#include <cstdint>
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- DmConf_broadcastInstruction ----
struct DmConf_broadcastInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] unknown
    uint8_t broadcast_if_;  // +50 raw[10:10] unknown
    uint8_t broadcast_w_;  // +51 raw[11:11] unknown
    uint8_t psum_cascade_;  // +52 raw[12:12] unknown
    uint8_t reserved0_;  // +53 raw[15:13] unknown
    void get_next_pc() override;   // pc + 2 (compressed encoding)
    ~DmConf_broadcastInstruction() override;
};
template <> DmConf_broadcastInstruction Simulator::InstParser<DmConf_broadcastInstruction, 16>(unsigned char **pc);

// ---- DmLoadAct0 ----
struct DmLoadAct0;   // activation-load descriptor created by Dm::GetLoadAct0()

// DM: load activation-0 parameters (GLB source) and enqueue the load on Conv2D or PDP0.
// asm: dm_load_act0 tcu_id, pu_id, raddr_s, rlen, dest_channel, is_by_channel
struct DmLoadAct0Instruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t raddr_s_;  // +51 raw[17:13] register: GLB source address
    uint8_t rlen_;  // +52 raw[22:18] register: length (captured, unused by operation)
    uint8_t dest_channel_;  // +53 raw[23:23] 1: pdp0, 0: pu (Conv2D)
    uint8_t is_by_channel_;  // +54 raw[24:24] by-channel flag, copied to Dm+105
    uint8_t reserved_25_;  // +55 raw[31:25] reserved, unused
    uint32_t raddr_s_val_;  // +56  g_gp_reg[raddr_s]  GLB address: bank[31:28] | offset[27:0]
    uint32_t rlen_val_;  // +60  g_gp_reg[rlen]
    uint32_t raddr_s_mmu_addr_;  // +64  MMU-translated source address (offset + 32 * MMU item)
    void get_next_pc() override;
    void operation() override;
    ~DmLoadAct0Instruction() override;
};
template <> DmLoadAct0Instruction Simulator::InstParser<DmLoadAct0Instruction, 32>(unsigned char **pc);

// ---- DmLoadL1ConfInstruction ----
struct DmLoadL1ConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] dm_conf variant select (sub_opcode dispatched by main), not an assembly operand
    uint8_t rstride_s_;  // +52 raw[19:17] stride shape register index
    uint8_t datatype_;  // +53 raw[21:20] element datatype, copied to Dm+60
    uint8_t l1_type_;  // +54 raw[23:22] L1 buffer type, unused by operation
    uint8_t reserved_24_;  // +55 raw[31:24] reserved, unused
    uint64_t rstride_s_val_;  // +56  g_shape_reg[rstride_s]: dim0 = [47:32], dim1 = [31:16], dim2 = [15:0]
    void get_next_pc() override;
    void operation() override;
    ~DmLoadL1ConfInstruction() override;
};
template <> DmLoadL1ConfInstruction Simulator::InstParser<DmLoadL1ConfInstruction, 32>(unsigned char **pc);

// ---- DmLoadL1Instruction ----
struct DmLoadL1Instruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t raddr_s_;  // +51 raw[17:13] register: GLB source address
    uint8_t rhtoc_window_;  // +52 raw[22:18] register: H-to-C window
    uint8_t rshape_;  // +53 raw[25:23] shape register index
    uint8_t l1_type_;  // +54 raw[27:26] L1 buffer type (if_, ...), unused by operation
    uint16_t reserved_28_;  // +56 raw[31:28] reserved, unused
    uint32_t raddr_s_val_;  // +60  g_gp_reg[raddr_s]  GLB address: bank[31:28] | offset[27:0]
    uint32_t rhtoc_window_val_;  // +64  g_gp_reg[rhtoc_window]
    uint64_t rshape_val_;  // +72  g_shape_reg[rshape]
    uint32_t raddr_s_mmu_addr_;  // +80  MMU-translated source address (offset + 32 * MMU item)
    void get_next_pc() override;
    void operation() override;   // reconstructed from the assembly @0x423140
    ~DmLoadL1Instruction() override;
};
template <> DmLoadL1Instruction Simulator::InstParser<DmLoadL1Instruction, 32>(unsigned char **pc);

// ---- DmLoadWConf2Instruction ----
struct DmLoadWConf2Instruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] dm_conf variant select (sub_opcode dispatched by main), not an assembly operand
    uint8_t rgroups_;  // +52 raw[21:17] register: number of groups
    uint8_t rgoc_;  // +53 raw[26:22] register: groups of output channels
    uint8_t reserved_27_;  // +54 raw[31:27] reserved, unused
    uint8_t rgroups_val_lo8_;  // +55  low byte of g_gp_reg[rgroups]
                            //      verified against asm @0x4151d0: the original really stores only the low byte (movb %dil,0x37) of rs1_val
    uint64_t rgoc_val_;  // +56  g_gp_reg[rgoc] (zero-extended)
    void get_next_pc() override;
    ~DmLoadWConf2Instruction() override;
};
template <> DmLoadWConf2Instruction Simulator::InstParser<DmLoadWConf2Instruction, 32>(unsigned char **pc);

// ---- DmLoadWConf_deqInstruction ----
struct DmLoadWConf_deqInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] dm_conf variant select (sub_opcode dispatched by main), not an assembly operand
    uint8_t quant_type_;  // +52 raw[18:17] dequantisation type, copied to Dm+16
    uint16_t reserved_19_;  // +54 raw[31:19] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~DmLoadWConf_deqInstruction() override;
};
template <> DmLoadWConf_deqInstruction Simulator::InstParser<DmLoadWConf_deqInstruction, 32>(unsigned char **pc);

// ---- DmLoadWConfInstruction ----
struct DmLoadWConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] dm_conf variant select (sub_opcode dispatched by main), not an assembly operand
    uint8_t kernel_w_;  // +52 raw[26:22] kernel width, copied to Dm+8
    uint8_t kernel_h_;  // +53 raw[21:17] kernel height, copied to Dm+4
    uint8_t rstride_oc_;  // +54 raw[31:27] register: output-channel stride
    uint64_t rstride_oc_val_;  // +56  g_gp_reg[rstride_oc]  copied to Dm+12 (truncated to 32 bits)
                            //      (Dm+12 * Dm+42 is the byte count in Dm::GetLoadW)
    void get_next_pc() override;
    void operation() override;
    ~DmLoadWConfInstruction() override;
};
template <> DmLoadWConfInstruction Simulator::InstParser<DmLoadWConfInstruction, 32>(unsigned char **pc);

// ---- DmLoadW ----
struct DmLoadW;   // weight-load descriptor created by Dm::GetLoadW()

// DM: load weights from GLB (two GLB addresses + a shape) and enqueue the load on Conv2D or PDP0.
// asm: dm_load_w tcu_id, pu_id, raddr_s, raddr_bw, r_iochannels, dest_type
struct DmLoadWInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t raddr_s_;  // +51 raw[17:13] register: first GLB address (weights)
    uint8_t raddr_bw_;  // +52 raw[22:18] register: second GLB address (bias / zero points)
    uint8_t r_iochannels_;  // +53 raw[25:23] shape register index (in/out channels)
    uint8_t dest_type_;  // +54 raw[27:26] non-zero: pdp0, 0: pu (Conv2D); also Dm+40
    uint8_t reserved_28_;  // +55 raw[31:28] reserved, unused
    uint32_t raddr_s_val_;  // +56  g_gp_reg[raddr_s]  GLB address: bank[31:28] | offset[27:0]
    uint32_t raddr_bw_val_;  // +60  g_gp_reg[raddr_bw]
    uint32_t r_iochannels_val_lo32_;  // +64  low 32 bits of g_shape_reg[r_iochannels] (the 64-bit value is truncated by the original)
    uint32_t raddr_s_mmu_addr_;  // +68  MMU-translated src0 (offset + 32 * MMU item)
    uint32_t raddr_bw_mmu_addr_;  // +72  MMU-translated src1
    void get_next_pc() override;
    void operation() override;
    ~DmLoadWInstruction() override;
};
template <> DmLoadWInstruction Simulator::InstParser<DmLoadWInstruction, 32>(unsigned char **pc);

// ---- DmStoreOfConfInstruction ----
struct DmStoreOfConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] dm_conf variant select (sub_opcode dispatched by main), not an assembly operand
    uint8_t rstride_d_;  // +52 raw[19:17] stride shape register index
    uint8_t datatype_;  // +53 raw[21:20] element datatype, copied to Dm+124
    uint16_t reserved_22_;  // +54 raw[31:22] reserved, unused
    uint64_t rstride_d_val_;  // +56  g_shape_reg[rstride_d]: dim0 = [47:32], dim1 = [31:16], dim2 = [15:0]
    void get_next_pc() override;
    void operation() override;
    ~DmStoreOfConfInstruction() override;
};
template <> DmStoreOfConfInstruction Simulator::InstParser<DmStoreOfConfInstruction, 32>(unsigned char **pc);

// ---- DmStoreOf ----
struct DmStoreOf;   // OF-store descriptor created by Dm::GetStoreOf()

// DM: store output features to GLB, enqueue the store on Conv2D or PDP0 and kick off the compute.
// asm: dm_store_of tcu_id, pu_id, raddr_d, rshape, src_channel
struct DmStoreOfInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t raddr_d_;  // +51 raw[17:13] register: GLB destination address
    uint8_t rshape_;  // +52 raw[20:18] shape register index
    uint8_t src_channel_;  // +53 raw[21:21] 1: pdp0, 0: pu (Conv2D); also Dm+156
    uint16_t reserved_22_;  // +54 raw[31:22] reserved, unused
    uint32_t raddr_d_val_;  // +56  g_gp_reg[raddr_d]  GLB address: bank[31:28] | offset[27:0]
    uint64_t rshape_val_;  // +64  g_shape_reg[rshape]: four 16-bit dims, dim0 = [63:48] ... dim3 = [15:0]
    uint32_t raddr_d_mmu_addr_;  // +72  MMU-translated destination (offset + 32 * MMU item)
    void get_next_pc() override;
    void operation() override;
    ~DmStoreOfInstruction() override;
};
template <> DmStoreOfInstruction Simulator::InstParser<DmStoreOfInstruction, 32>(unsigned char **pc);
