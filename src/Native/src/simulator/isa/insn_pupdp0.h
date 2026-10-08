#pragma once
// Lifted from IDA/Hex-Rays output. Fields carry their byte offset in the object.
// asm: (not in compiler dumps) pu_pdp0_compute ...
// asm: (not in compiler dumps) pu_pdp0_conf_deq ...
// asm: (not in compiler dumps) pu_pdp0_fetchif_conf1 ...
// asm: (not in compiler dumps) pu_pdp0_fetchif_conf2 ...   (operation() writes nothing)
// asm: (not in compiler dumps) pu_pdp0_fetchif_conf3 ...
// asm: (not in compiler dumps) pu_pdp0_fetchif_conf4 ...
// asm: (not in compiler dumps) pu_pdp0_mode_conf ...
// asm: (not in compiler dumps) pu_pdp0_of_conf ...
// asm: (not in compiler dumps) pu_pdp0_w_conf ...
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- PuPdp0ComputeInstruction ----
struct PuPdp0ComputeInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id
    uint8_t rs_;  // +50 raw[14:10] register holding the compute parameter
    uint8_t reserved_15_;  // +51 raw[15] reserved, unused
    uint32_t compute_param_;  // +52 g_gp_reg[rs], stored to PDP0+65632
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0ComputeInstruction() override;
};
template <> PuPdp0ComputeInstruction Simulator::InstParser<PuPdp0ComputeInstruction, 16>(unsigned char **pc);

// ---- PuPdp0Conf_deqInstruction ----
struct PuPdp0Conf_deqInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rs1_;  // +52 raw[21:17] register: input zero point
    uint8_t unsigned_flag_;  // +53 raw[23:22] unsigned flag (PDP0 regs.unsigned_flag)
    uint8_t reserved_24_;  // +54 raw[31:24] reserved, unused
    uint32_t rs1_val_;  // +56 g_gp_reg[rs1]
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0Conf_deqInstruction() override;
};
template <> PuPdp0Conf_deqInstruction Simulator::InstParser<PuPdp0Conf_deqInstruction, 32>(unsigned char **pc);

// ---- PuPdp0FetchifConf1Instruction ----
struct PuPdp0FetchifConf1Instruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t imm17_;  // +52 raw[21:17] immediate -> PDP0 word 16385
    uint8_t imm22_;  // +53 raw[26:22] immediate -> PDP0 word 16386
    uint8_t reserved_27_;  // +54 raw[31:27] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0FetchifConf1Instruction() override;
};
template <> PuPdp0FetchifConf1Instruction Simulator::InstParser<PuPdp0FetchifConf1Instruction, 32>(unsigned char **pc);

// ---- PuPdp0FetchifConf2Instruction ----
struct PuPdp0FetchifConf2Instruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rs1_;  // +52 raw[21:17] register (unused)
    uint8_t rs2_;  // +53 raw[26:22] register (unused)
    uint8_t reserved_27_;  // +54 raw[31:27] reserved, unused
    uint32_t rs1_val_;  // +56 g_gp_reg[rs1]
    uint32_t rs2_val_;  // +60 g_gp_reg[rs2]
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0FetchifConf2Instruction() override;
};
template <> PuPdp0FetchifConf2Instruction Simulator::InstParser<PuPdp0FetchifConf2Instruction, 32>(unsigned char **pc);

// ---- PuPdp0FetchifConf3Instruction ----
struct PuPdp0FetchifConf3Instruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint16_t reserved_17_;  // +52 raw[26:17] decoded, unused
    uint8_t shape_idx_;  // +54 raw[29:27] shape register index
    uint8_t reserved_30_;  // +55 raw[31:30] reserved, unused
    uint64_t shape_;  // +56 g_shape_reg[shape_idx]
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0FetchifConf3Instruction() override;
};
template <> PuPdp0FetchifConf3Instruction Simulator::InstParser<PuPdp0FetchifConf3Instruction, 32>(unsigned char **pc);

// ---- PuPdp0FetchifConf4Instruction ----
struct PuPdp0FetchifConf4Instruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t rs1_;  // +52 raw[21:17] register -> PDP0 word 16391
    uint8_t shape_idx_;  // +53 raw[24:22] shape register index
    uint8_t reserved_25_;  // +54 raw[31:25] reserved, unused
    uint32_t rs1_val_;  // +56 g_gp_reg[rs1]
    uint64_t shape_;  // +64 g_shape_reg[shape_idx]
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0FetchifConf4Instruction() override;
};
template <> PuPdp0FetchifConf4Instruction Simulator::InstParser<PuPdp0FetchifConf4Instruction, 32>(unsigned char **pc);

// ---- PuPdp0ModeConfInstruction ----
struct PuPdp0ModeConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t mode_;  // +52 raw[19:17] PDP0 mode register
    uint16_t reserved_20_;  // +54 raw[31:20] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0ModeConfInstruction() override;
};
template <> PuPdp0ModeConfInstruction Simulator::InstParser<PuPdp0ModeConfInstruction, 32>(unsigned char **pc);

// ---- PuPdp0OfConfInstruction ----
struct PuPdp0OfConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t shape_idx0_;  // +52 raw[19:17] shape register index 0
    uint8_t shape_idx1_;  // +53 raw[22:20] shape register index 1
    uint16_t reserved_23_;  // +54 raw[31:23] reserved, unused
    uint64_t shape0_;  // +56 g_shape_reg[shape_idx0]
    uint64_t shape1_;  // +64 g_shape_reg[shape_idx1]
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0OfConfInstruction() override;
};
template <> PuPdp0OfConfInstruction Simulator::InstParser<PuPdp0OfConfInstruction, 32>(unsigned char **pc);

// ---- PuPdp0WConfInstruction ----
struct PuPdp0WConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // +50 raw[12:10] PU id (compiler range 0)
    uint8_t funct4_;  // +51 raw[16:13] pu_conf variant select (sub-opcode dispatched by main), not an assembly operand
    uint8_t imm17_;  // +52 raw[21:17] immediate -> PDP0+65596
    uint8_t imm22_;  // +53 raw[26:22] immediate -> PDP0+65592
    uint8_t reserved_27_;  // +54 raw[31:27] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~PuPdp0WConfInstruction() override;
};
template <> PuPdp0WConfInstruction Simulator::InstParser<PuPdp0WConfInstruction, 32>(unsigned char **pc);
