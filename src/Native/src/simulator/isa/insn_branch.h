#pragma once
// Lifted from IDA/Hex-Rays output; field names are inferred.
// asm: (not in compiler dumps) beq ...
// asm: (not in compiler dumps) bne ...
// asm: (not in compiler dumps) blt ...
// asm: (not in compiler dumps) bltu ...
// asm: (not in compiler dumps) bge ...
// asm: (not in compiler dumps) bgeu ...
// asm: (not in compiler dumps) jal ...
// asm: (not in compiler dumps) jalr ...
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

// ---- BeqInstruction ----
struct BeqInstruction : public KInstruction {
    uint8_t rs1_;  // +49 raw[11:7] source register 1
    uint8_t rs2_;  // +50 raw[16:12] source register 2
    uint8_t funct3_;  // +51 raw[19:17] branch condition select (dispatch field of main)
    uint16_t imm_;  // +52 raw[31:20] 12-bit branch offset (raw)
    uint32_t rs1_val_;  // +56 value of rs1 (read at decode time)
    uint32_t rs2_val_;  // +60 value of rs2 (read at decode time)
    void operation() override;
    void get_next_pc() override;
    ~BeqInstruction() override;
};
template <> BeqInstruction Simulator::InstParser<BeqInstruction, 32>(unsigned char **pc);

// ---- BneInstruction ----
struct BneInstruction : public KInstruction {
    uint8_t rs1_;  // +49 raw[11:7] source register 1
    uint8_t rs2_;  // +50 raw[16:12] source register 2
    uint8_t funct3_;  // +51 raw[19:17] branch condition select (dispatch field of main)
    uint16_t imm_;  // +52 raw[31:20] 12-bit branch offset (raw)
    uint32_t rs1_val_;  // +56 value of rs1 (read at decode time)
    uint32_t rs2_val_;  // +60 value of rs2 (read at decode time)
    void operation() override;
    void get_next_pc() override;
    ~BneInstruction() override;
};
template <> BneInstruction Simulator::InstParser<BneInstruction, 32>(unsigned char **pc);

// ---- BltInstruction ----
struct BltInstruction : public KInstruction {
    uint8_t rs1_;  // +49 raw[11:7] source register 1
    uint8_t rs2_;  // +50 raw[16:12] source register 2
    uint8_t funct3_;  // +51 raw[19:17] branch condition select (dispatch field of main)
    uint16_t imm_;  // +52 raw[31:20] 12-bit branch offset (raw)
    uint32_t rs1_val_;  // +56 value of rs1 (read at decode time)
    uint32_t rs2_val_;  // +60 value of rs2 (read at decode time)
    void operation() override;
    void get_next_pc() override;
    ~BltInstruction() override;
};
template <> BltInstruction Simulator::InstParser<BltInstruction, 32>(unsigned char **pc);

// ---- BltuInstruction ----
struct BltuInstruction : public KInstruction {
    uint8_t rs1_;  // +49 raw[11:7] source register 1
    uint8_t rs2_;  // +50 raw[16:12] source register 2
    uint8_t funct3_;  // +51 raw[19:17] branch condition select (dispatch field of main)
    uint16_t imm_;  // +52 raw[31:20] 12-bit branch offset (raw)
    uint32_t rs1_val_;  // +56 value of rs1 (read at decode time)
    uint32_t rs2_val_;  // +60 value of rs2 (read at decode time)
    void operation() override;
    void get_next_pc() override;
    ~BltuInstruction() override;
};
template <> BltuInstruction Simulator::InstParser<BltuInstruction, 32>(unsigned char **pc);

// ---- BgeInstruction ----
struct BgeInstruction : public KInstruction {
    uint8_t rs1_;  // +49 raw[11:7] source register 1
    uint8_t rs2_;  // +50 raw[16:12] source register 2
    uint8_t funct3_;  // +51 raw[19:17] branch condition select (dispatch field of main)
    uint16_t imm_;  // +52 raw[31:20] 12-bit branch offset (raw)
    uint32_t rs1_val_;  // +56 value of rs1 (read at decode time)
    uint32_t rs2_val_;  // +60 value of rs2 (read at decode time)
    void operation() override;
    void get_next_pc() override;
    ~BgeInstruction() override;
};
template <> BgeInstruction Simulator::InstParser<BgeInstruction, 32>(unsigned char **pc);

// ---- BgeuInstruction ----
struct BgeuInstruction : public KInstruction {
    uint8_t rs1_;  // +49 raw[11:7] source register 1
    uint8_t rs2_;  // +50 raw[16:12] source register 2
    uint8_t funct3_;  // +51 raw[19:17] branch condition select (dispatch field of main)
    uint16_t imm_;  // +52 raw[31:20] 12-bit branch offset (raw)
    uint32_t rs1_val_;  // +56 value of rs1 (read at decode time)
    uint32_t rs2_val_;  // +60 value of rs2 (read at decode time)
    void operation() override;
    void get_next_pc() override;
    ~BgeuInstruction() override;
};
template <> BgeuInstruction Simulator::InstParser<BgeuInstruction, 32>(unsigned char **pc);

// ---- JalInstruction ----
struct JalInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] link register
    uint32_t imm_;  // +52 raw[31:12] 20-bit jump offset (raw)
    uint32_t link_;  // +56 return address (pc + 4)
    void operation() override;
    void get_next_pc() override;
    void parser_operation();
    ~JalInstruction() override;
};
template <> JalInstruction Simulator::InstParser<JalInstruction, 32>(unsigned char **pc);

// ---- JalrInstruction ----
struct JalrInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] link register
    uint8_t rs1_;  // +50 raw[16:12] base register
    uint8_t funct3_;  // +51 raw[19:17] must be 0 (dispatch field of main)
    uint16_t imm_;  // +52 raw[31:20] 12-bit signed offset (raw)
    uint32_t link_val_;  // +56 pc + 4 (value written to rd)
    uint32_t rs1_val_;  // +60 value of rs1 (g_gp_reg[rs1], loaded by parser_operation)
    void operation() override;
    void get_next_pc() override;
    void parser_operation();
    ~JalrInstruction() override;
};
template <> JalrInstruction Simulator::InstParser<JalrInstruction, 32>(unsigned char **pc);
