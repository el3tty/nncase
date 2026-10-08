#pragma once
// Lifted from IDA/Hex-Rays output.
// ACT0 source-1 configuration: rshape_val + parameter of the activation unit, then selects (creates) the
// Act0Compute descriptor on Conv2D or PDP0.
// asm: act0_src1_conf tcu_id, pu_id, channel, rshape
#include <cstddef>
#include <cstdint>
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- Act0Compute ----
struct Act0Compute;   // activation-0 compute descriptor (see Act0Src1ConfInstruction / Act0::GetAct0Compute)

// ACT0 compute: fill in the current Act0Compute descriptor and enqueue it on Conv2D or PDP0.
// asm: act0_compute raddr_d, tcu_id, channel, target, dest_datatype, is_by_channel
struct Act0ComputeInstruction : public KInstruction {
    uint8_t  reserved_7_;  // +49 raw[11:7] no operand, unused
    uint8_t  raddr_d_;  // +50 raw[16:12] register: destination address, copied to Act0Compute+24
    uint8_t  tcu_id_;  // +51 raw[19:17] TCU id (compiler range 0), unused
    uint8_t  channel_;  // +52 raw[20] 1: pdp0, 0: pu (Conv2D)
    uint8_t  target_;  // +53 raw[22:21] output target (dm / psum), stored at Act0Compute+28
    uint8_t  dest_datatype_;  // +54 raw[24:23] destination datatype, stored at Act0Compute+32
    uint8_t  is_by_channel_;  // +55 raw[25] stored at Act0Compute+36
    uint8_t  reserved_26_;  // +56 raw[31:26] reserved, unused
    uint32_t raddr_d_val_;       // +60  g_gp_reg[raddr_d]
    void get_next_pc() override;
    void operation() override;   // the IDA prototype carried a stray int64_t argument (a register leftover)
    ~Act0ComputeInstruction() override;
};
template <> Act0ComputeInstruction Simulator::InstParser<Act0ComputeInstruction, 32>(unsigned char **pc);

// ---- Act0Src1ConfInstruction ----
struct Act0Src1ConfInstruction : public KInstruction {
    uint8_t  tcu_id_;  // +49 raw[9:7] TCU id (compiler range 0), unused
    uint8_t  pu_id_;  // +50 raw[12:10] PU id (compiler range 0), unused
    uint8_t  funct3_;  // +51 raw[15:13] act0_conf variant select, not an assembly operand
    uint8_t  channel_;  // +52 raw[16] 1: pdp0, 0: pu (Conv2D); also stored in the Act0 singleton
    uint8_t  rshape_;  // +53 raw[19:17] shape register index
    uint8_t  src1_param_;  // +54 raw[24:20] no operand in the dumps; stored in the Act0 singleton
    uint16_t reserved_25_;  // +56 raw[31:25] reserved, unused
    uint64_t rshape_val_;         // +64  g_shape_reg[rshape]: four 16-bit dims, dim0 = [63:48] ... dim3 = [15:0]
    void get_next_pc() override;
    void operation() override;
    ~Act0Src1ConfInstruction() override;
};
template <> Act0Src1ConfInstruction Simulator::InstParser<Act0Src1ConfInstruction, 32>(unsigned char **pc);
