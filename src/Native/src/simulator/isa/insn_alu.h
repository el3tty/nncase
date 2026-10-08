#pragma once
// Lifted from IDA/Hex-Rays output; field names are inferred.
// asm: add rd, rs1, rs2
// asm: sub rd, rs1, rs2
// asm: mul rd, rs1, rs2
// asm: div rd, rs1, rs2
// asm: divu rd, rs1, rs2
// asm: rem rd, rs1, rs2
// asm: remu rd, rs1, rs2
// asm: addi rd, rs, imm
// asm: lui rd, imm
// asm: (not in compiler dumps) auipc rd, imm
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

// ---- AddInstruction ----
struct AddInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs1_;  // +50 raw[16:12] source register 1
    uint8_t funct5_;  // +51 raw[21:17] ALU function select (dispatch field of main), not an assembly operand
    uint8_t rs2_;  // +52 raw[26:22] source register 2
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs1_val_;  // +60 value of rs1 (trace)
    uint32_t rs2_val_;  // +64 value of rs2 (trace)
    void get_next_pc() override;
    void parser_operation();
    ~AddInstruction() override;
};
template <> AddInstruction Simulator::InstParser<AddInstruction, 32>(unsigned char **pc);

// ---- SubInstruction ----
struct SubInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs1_;  // +50 raw[16:12] source register 1
    uint8_t funct5_;  // +51 raw[21:17] ALU function select (dispatch field of main), not an assembly operand
    uint8_t rs2_;  // +52 raw[26:22] source register 2
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs1_val_;  // +60 value of rs1 (trace)
    uint32_t rs2_val_;  // +64 value of rs2 (trace)
    void operation() override;
    void get_next_pc() override;
    void parser_operation();
    ~SubInstruction() override;
};
template <> SubInstruction Simulator::InstParser<SubInstruction, 32>(unsigned char **pc);

// ---- MulInstruction ----
struct MulInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs1_;  // +50 raw[16:12] source register 1
    uint8_t funct5_;  // +51 raw[21:17] ALU function select (dispatch field of main), not an assembly operand
    uint8_t rs2_;  // +52 raw[26:22] source register 2
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs1_val_;  // +60 value of rs1 (trace)
    uint32_t rs2_val_;  // +64 value of rs2 (trace)
    void get_next_pc() override;
    void parser_operation();
    ~MulInstruction() override;
};
template <> MulInstruction Simulator::InstParser<MulInstruction, 32>(unsigned char **pc);

// ---- DivInstruction ----
struct DivInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs1_;  // +50 raw[16:12] source register 1
    uint8_t funct5_;  // +51 raw[21:17] ALU function select (dispatch field of main), not an assembly operand
    uint8_t rs2_;  // +52 raw[26:22] source register 2
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs1_val_;  // +60 value of rs1 (trace)
    uint32_t rs2_val_;  // +64 value of rs2 (trace)
    void get_next_pc() override;
    void parser_operation();
    ~DivInstruction() override;
};
template <> DivInstruction Simulator::InstParser<DivInstruction, 32>(unsigned char **pc);

// ---- DivuInstruction ----
struct DivuInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs1_;  // +50 raw[16:12] source register 1
    uint8_t funct5_;  // +51 raw[21:17] ALU function select (dispatch field of main), not an assembly operand
    uint8_t rs2_;  // +52 raw[26:22] source register 2
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs1_val_;  // +60 value of rs1 (trace)
    uint32_t rs2_val_;  // +64 value of rs2 (trace)
    void get_next_pc() override;
    void parser_operation();
    ~DivuInstruction() override;
};
template <> DivuInstruction Simulator::InstParser<DivuInstruction, 32>(unsigned char **pc);

// ---- RemInstruction ----
struct RemInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs1_;  // +50 raw[16:12] source register 1
    uint8_t funct5_;  // +51 raw[21:17] ALU function select (dispatch field of main), not an assembly operand
    uint8_t rs2_;  // +52 raw[26:22] source register 2
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs1_val_;  // +60 value of rs1 (trace)
    uint32_t rs2_val_;  // +64 value of rs2 (trace)
    void get_next_pc() override;
    void parser_operation();
    ~RemInstruction() override;
};
template <> RemInstruction Simulator::InstParser<RemInstruction, 32>(unsigned char **pc);

// ---- RemuInstruction ----
struct RemuInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs1_;  // +50 raw[16:12] source register 1
    uint8_t funct5_;  // +51 raw[21:17] ALU function select (dispatch field of main), not an assembly operand
    uint8_t rs2_;  // +52 raw[26:22] source register 2
    uint8_t reserved_27_;  // +53 raw[31:27] reserved, decoded but unused
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs1_val_;  // +60 value of rs1 (trace)
    uint32_t rs2_val_;  // +64 value of rs2 (trace)
    void get_next_pc() override;
    void parser_operation();
    ~RemuInstruction() override;
};
template <> RemuInstruction Simulator::InstParser<RemuInstruction, 32>(unsigned char **pc);

// ---- AddiInstruction ----
struct AddiInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs_;  // +50 raw[16:12] source register
    uint8_t funct5_;  // +51 raw[19:17] must be 0 (dispatch field of main), not an assembly operand
    uint16_t imm_;  // +52 raw[31:20] 12-bit signed immediate (raw)
    uint32_t result_;  // +56 value written to rd (trace)
    uint32_t rs_val_;  // +60 value of rs (trace)
    void get_next_pc() override;
    void parser_operation();
    ~AddiInstruction() override;
};
template <> AddiInstruction Simulator::InstParser<AddiInstruction, 32>(unsigned char **pc);

// ---- LuiInstruction ----
struct LuiInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint32_t imm_;  // +52 raw[31:12] 20-bit upper immediate (value << 12)
    uint32_t result_;  // +56 value written to rd (trace)
    void get_next_pc() override;
    void parser_operation();
    ~LuiInstruction() override;
};
template <> LuiInstruction Simulator::InstParser<LuiInstruction, 32>(unsigned char **pc);

// ---- AuipcInstruction ----
struct AuipcInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint32_t imm_;  // +52 raw[31:12] 20-bit upper immediate
    uint32_t result_;  // +56 value written to rd (trace)
    void get_next_pc() override;
    void parser_operation();
    ~AuipcInstruction() override;
};
template <> AuipcInstruction Simulator::InstParser<AuipcInstruction, 32>(unsigned char **pc);
