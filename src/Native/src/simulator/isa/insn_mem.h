#pragma once
// Lifted from IDA/Hex-Rays output; field names are inferred.
// asm: lb rd, rs, offset   (by analogy with lw; only lw appears in the dumps)
// asm: lbu rd, rs, offset   (by analogy with lw; only lw appears in the dumps)
// asm: lh rd, rs, offset   (by analogy with lw; only lw appears in the dumps)
// asm: lhu rd, rs, offset   (by analogy with lw; only lw appears in the dumps)
// asm: lw rd, rs, offset
// asm: (not in compiler dumps) sb ...
// asm: (not in compiler dumps) sh ...
// asm: (not in compiler dumps) sw ...
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

// ---- LbInstruction ----
struct LbInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs_;  // +50 raw[16:12] base register
    uint8_t funct3_;  // +51 raw[19:17] load width select (dispatch field of main), not an assembly operand
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed byte offset (raw)
    uint32_t result_;  // +56 loaded value after extension (trace)
    uint32_t rs_val_;  // +60 value of rs (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~LbInstruction() override;
};
template <> LbInstruction Simulator::InstParser<LbInstruction, 32>(unsigned char **pc);

// ---- LbuInstruction ----
struct LbuInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs_;  // +50 raw[16:12] base register
    uint8_t funct3_;  // +51 raw[19:17] load width select (dispatch field of main), not an assembly operand
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed byte offset (raw)
    uint32_t result_;  // +56 loaded value after extension (trace)
    uint32_t rs_val_;  // +60 value of rs (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~LbuInstruction() override;
};
template <> LbuInstruction Simulator::InstParser<LbuInstruction, 32>(unsigned char **pc);

// ---- LhInstruction ----
struct LhInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs_;  // +50 raw[16:12] base register
    uint8_t funct3_;  // +51 raw[19:17] load width select (dispatch field of main), not an assembly operand
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed byte offset (raw)
    uint32_t result_;  // +56 loaded value after extension (trace)
    uint32_t rs_val_;  // +60 value of rs (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~LhInstruction() override;
};
template <> LhInstruction Simulator::InstParser<LhInstruction, 32>(unsigned char **pc);

// ---- LhuInstruction ----
struct LhuInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs_;  // +50 raw[16:12] base register
    uint8_t funct3_;  // +51 raw[19:17] load width select (dispatch field of main), not an assembly operand
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed byte offset (raw)
    uint32_t result_;  // +56 loaded value after extension (trace)
    uint32_t rs_val_;  // +60 value of rs (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~LhuInstruction() override;
};
template <> LhuInstruction Simulator::InstParser<LhuInstruction, 32>(unsigned char **pc);

// ---- LwInstruction ----
struct LwInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] destination register
    uint8_t rs_;  // +50 raw[16:12] base register
    uint8_t funct3_;  // +51 raw[19:17] load width select (dispatch field of main), not an assembly operand
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed byte offset (raw)
    uint32_t result_;  // +56 loaded value after extension (trace)
    uint32_t rs_val_;  // +60 value of rs (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~LwInstruction() override;
};
template <> LwInstruction Simulator::InstParser<LwInstruction, 32>(unsigned char **pc);

// ---- SbInstruction ----
struct SbInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] base register
    uint8_t rs_;  // +50 raw[16:12] data register
    uint8_t funct3_;  // +51 raw[19:17] store width select (dispatch field of main)
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed offset (raw)
    uint32_t rd_val_;  // +56 value of rs1 (trace)
    uint32_t rs_val_;  // +60 value of rs2 (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~SbInstruction() override;
};
template <> SbInstruction Simulator::InstParser<SbInstruction, 32>(unsigned char **pc);

// ---- ShInstruction ----
struct ShInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] base register
    uint8_t rs_;  // +50 raw[16:12] data register
    uint8_t funct3_;  // +51 raw[19:17] store width select (dispatch field of main)
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed offset (raw)
    uint32_t rd_val_;  // +56 value of rs1 (trace)
    uint32_t rs_val_;  // +60 value of rs2 (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~ShInstruction() override;
};
template <> ShInstruction Simulator::InstParser<ShInstruction, 32>(unsigned char **pc);

// ---- SwInstruction ----
struct SwInstruction : public KInstruction {
    uint8_t rd_;  // +49 raw[11:7] base register
    uint8_t rs_;  // +50 raw[16:12] data register
    uint8_t funct3_;  // +51 raw[19:17] store width select (dispatch field of main)
    uint16_t offset_;  // +52 raw[31:20] 12-bit signed offset (raw)
    uint32_t rd_val_;  // +56 value of rs1 (trace)
    uint32_t rs_val_;  // +60 value of rs2 (trace)
    uint32_t mem_addr_;  // +64 MMU-translated address (trace)
    void get_next_pc() override;
    void parser_operation();
    ~SwInstruction() override;
};
template <> SwInstruction Simulator::InstParser<SwInstruction, 32>(unsigned char **pc);
