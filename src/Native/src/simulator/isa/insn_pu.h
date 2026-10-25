#pragma once
// asm: pu_compute tcu_id, of_shift_mode
// asm: pu_compute_conf tcu_id, pu_id, load_psum, clr_psum, dest_target, release_if, mode
// asm: pu_fetchif_conf1 tcu_id, pu_id, stride_w, stride_h, rstride_s
// asm: pu_fetchif_conf2 tcu_id, pu_id, rgic, rgic_last   (operation is an empty stub)
// asm: pu_fetchif_conf3 tcu_id, pu_id, raddr_s, rgroups, rshape
// asm: pu_fetchif_conf4 tcu_id, pu_id, rpad_value, sspad   (sspad at [29:27] per the decoder; operand order would suggest [26:22])
// asm: pu_fetchif_conf_deq tcu_id, pu_id, ric, rbx, quant_type
// asm: pu_forward_psum tcu_id, pu_id, raddr, rlen   (after the last pass of a transposed convolution; a board only forwards after an accumulating pu_compute, see README)
// asm: pu_of_conf1 tcu_id, pu_id, rgoc, rgoc_last, rstride_d
// asm: pu_of_conf2 tcu_id, pu_id, raddr_d, rshape_d
// asm: pu_w_conf tcu_id, pu_id, kernel_h, kernel_w
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- PuComputeInstruction ----
struct PuComputeInstruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t of_shift_mode_;  // raw[11:10] output shift mode (none / left_shift_4 / right_shift_4), stored to
    uint8_t reserved_;  // raw[15:12] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~PuComputeInstruction() override;
};
template <> PuComputeInstruction Simulator::InstParser<PuComputeInstruction, 16>(unsigned char **pc);

// ---- PuComputeConfInstruction ----
struct PuComputeConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t load_psum_;  // raw[17] accumulate into the PSUM already present ->
    uint8_t clr_psum_;  // raw[18] zero PSUM before computing ->
    uint8_t dest_target_;  // raw[19] 1: act0 (result forwarded to ACT0), 0: psum ->
    uint8_t release_if_;  // raw[20] release the IF buffer ->
    uint8_t mode_;  // raw[21] pu mode (pu_mode_normal) ->
    uint16_t reserved_;  // raw[29:22] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~PuComputeConfInstruction() override;
};
template <> PuComputeConfInstruction Simulator::InstParser<PuComputeConfInstruction, 32>(unsigned char **pc);

// ---- PuFetchifConf1Instruction ----
struct PuFetchifConf1Instruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t stride_w_;  // raw[21:17] horizontal stride (immediate) -> Conv2D cfg[0]
    uint8_t stride_h_;  // raw[26:22] vertical stride (immediate) -> Conv2D cfg[1]
    uint8_t rstride_s_;  // raw[29:27] stride shape register index
    uint8_t reserved_;  // raw[31:30] reserved, unused
    uint64_t rstride_s_val_;  // _G.shape_reg[rstride_s]
    void get_next_pc() override;
    void operation() override;
    ~PuFetchifConf1Instruction() override;
};
template <> PuFetchifConf1Instruction Simulator::InstParser<PuFetchifConf1Instruction, 32>(unsigned char **pc);

// ---- PuFetchifConf2Instruction ----
struct PuFetchifConf2Instruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t rgic_;  // raw[21:17] register: number of input-channel groups
    uint8_t rgic_last_;  // raw[26:22] register: input channels of the last group
    uint8_t reserved_;  // raw[31:27] reserved, unused
    uint32_t rgic_val_;  // _G.gp_reg[rgic]
    uint32_t rgic_last_val_;  // _G.gp_reg[rgic_last]
    void get_next_pc() override;
    ~PuFetchifConf2Instruction() override;
};
template <> PuFetchifConf2Instruction Simulator::InstParser<PuFetchifConf2Instruction, 32>(unsigned char **pc);

// ---- PuFetchifConf3Instruction ----
struct PuFetchifConf3Instruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t raddr_s_;  // raw[21:17] register: IF source address -> Conv2D cfg[7]
    uint8_t rgroups_;  // raw[26:22] register: number of groups -> Conv2D cfg[6]
    uint8_t rshape_;  // raw[29:27] shape register index
    uint8_t reserved_;  // raw[31:30] reserved, unused
    uint32_t raddr_s_val_;  // _G.gp_reg[raddr_s]
    uint32_t rgroups_val_;  // _G.gp_reg[rgroups]
    uint64_t rshape_val_;  // _G.shape_reg[rshape]
    void get_next_pc() override;
    void operation() override;
    ~PuFetchifConf3Instruction() override;
};
template <> PuFetchifConf3Instruction Simulator::InstParser<PuFetchifConf3Instruction, 32>(unsigned char **pc);

// ---- PuFetchifConf4Instruction ----
struct PuFetchifConf4Instruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t rpad_value_;  // raw[21:17] register: pad value (low byte used)
    uint8_t sspad_;  // raw[29:27] pad shape register index
    uint8_t reserved1_;  // raw[26:22] decoded but unused
    uint8_t reserved2_;  // raw[31:30] reserved, unused
    uint32_t rpad_value_val_;  // _G.gp_reg[rpad_value]
    uint64_t sspad_val_;  // _G.shape_reg[sspad]
    void get_next_pc() override;
    void operation() override;
    ~PuFetchifConf4Instruction() override;
};
template <> PuFetchifConf4Instruction Simulator::InstParser<PuFetchifConf4Instruction, 32>(unsigned char **pc);

// ---- PuFetchifConf_deqInstruction ----
struct PuFetchifConf_deqInstruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t ric_;  // raw[21:17] register: index of the dequant byte slot (31 = all)
    uint8_t rbx_;  // raw[26:22] register: dequant byte value
    uint8_t quant_type_;  // raw[28:27] IF quantisation type ->
    uint8_t reserved_;  // raw[31:29] reserved, unused
    uint32_t ric_val_;  // _G.gp_reg[ric]: index of the dequant byte slot (31 = all)
    uint32_t rbx_val_;  // _G.gp_reg[rbx]: dequant byte value
    void get_next_pc() override;
    void operation() override;
    ~PuFetchifConf_deqInstruction() override;
};
template <> PuFetchifConf_deqInstruction Simulator::InstParser<PuFetchifConf_deqInstruction, 32>(unsigned char **pc);

// ---- PuForward_psumInstruction ----
struct PuForward_psumInstruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id
    uint8_t pu_id_;  // raw[12:10] PU id
    uint8_t raddr_;  // raw[17:13] register: source byte offset in _G.PSUM_L1
    uint8_t rlen_;  // raw[22:18] register: words copied per row
    uint16_t reserved_;  // raw[29:21] decoded, unused (overlaps rs2; bit 22 masked out)
    uint32_t raddr_val_;  // _G.gp_reg[rs1]: source byte offset in _G.PSUM_L1
    uint64_t rlen_val_;  // _G.gp_reg[rs2]: words copied per row
    void get_next_pc() override;
    void operation() override;
    ~PuForward_psumInstruction() override;
};
template <> PuForward_psumInstruction Simulator::InstParser<PuForward_psumInstruction, 32>(unsigned char **pc);

// ---- PuOfConf1Instruction ----
struct PuOfConf1Instruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t rgoc_;  // raw[21:17] register: groups of output channels (unused by operation)
    uint8_t rgoc_last_;  // raw[26:22] register: last group of output channels (unused by operation)
    uint8_t rstride_d_;  // raw[29:27] output stride shape register index
    uint8_t reserved_;  // raw[31:30] reserved, unused
    uint32_t rgoc_val_;  // _G.gp_reg[rgoc] (unused)
    uint32_t rgoc_last_val_;  // _G.gp_reg[rgoc_last] (unused)
    uint64_t rstride_d_val_;  // _G.shape_reg[rstride_d]
    void get_next_pc() override;
    void operation() override;
    ~PuOfConf1Instruction() override;
};
template <> PuOfConf1Instruction Simulator::InstParser<PuOfConf1Instruction, 32>(unsigned char **pc);

// ---- PuOfConf2Instruction ----
struct PuOfConf2Instruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t raddr_d_;  // raw[21:17] register: output (PSUM) byte offset -> Conv2D cfg[34]
    uint8_t reserved1_;  // raw[26:22] decoded but unused
    uint8_t rshape_d_;  // raw[29:27] output shape register index
    uint8_t reserved2_;  // raw[31:30] reserved, unused
    uint32_t raddr_d_val_;  // _G.gp_reg[raddr_d] -> Conv2D cfg[34]
    uint64_t rshape_d_val_;  // _G.shape_reg[rshape_d]
    void get_next_pc() override;
    void operation() override;
    ~PuOfConf2Instruction() override;
};
template <> PuOfConf2Instruction Simulator::InstParser<PuOfConf2Instruction, 32>(unsigned char **pc);

// ---- PuWConfInstruction ----
struct PuWConfInstruction : public KInstruction {
    uint8_t tcu_id_;  // raw[9:7] TCU id (compiler range 0)
    uint8_t pu_id_;  // raw[12:10] PU id (compiler range 0)
    uint8_t conf_id_;  // raw[16:13] pu_conf register select (dispatched by main), not an assembly operand
    uint8_t kernel_h_;  // raw[21:17] kernel height (immediate) ->
    uint8_t kernel_w_;  // raw[26:22] kernel width (immediate) ->
    uint8_t reserved_;  // raw[31:27] reserved, unused
    void get_next_pc() override;
    void operation() override;
    ~PuWConfInstruction() override;
};
template <> PuWConfInstruction Simulator::InstParser<PuWConfInstruction, 32>(unsigned char **pc);
