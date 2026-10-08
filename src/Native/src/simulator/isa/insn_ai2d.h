#pragma once
// Lifted from IDA/Hex-Rays output.
// AI2D compute: 16-bit instruction without own operands or operation() body in the binary.
// asm: (not in compiler dumps) ai2d_compute ...
#include <cstddef>
#include <cstdint>
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- Ai2dComputeInstruction ----
struct Ai2dComputeInstruction : public KInstruction {
    uint16_t imm9_;  // +50 raw[15:7] 9-bit field
    void get_next_pc() override;   // pc + 2
    ~Ai2dComputeInstruction() override;
};
template <> Ai2dComputeInstruction Simulator::InstParser<Ai2dComputeInstruction, 16>(unsigned char **pc);
