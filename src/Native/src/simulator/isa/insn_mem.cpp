// Lifted from IDA/Hex-Rays output; names and types are inferred.
// Operand/result fields are named after their role; "+NN" in the header gives the byte offset.
// Simulator6.cpp  @0x411600
// Simulator7.cpp  @0x411830
// Simulator4.cpp  @0x4111a0
// Simulator5.cpp  @0x4113d0
// Simulator3.cpp  @0x410f70
// Simulator10.cpp  @0x411ec0
// Simulator9.cpp  @0x411c90
// Simulator8.cpp  @0x411a60
#include "isa/insn_mem.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/memaccessor.h"
#include "engines/simulator.h"

// ---- LbInstruction ----
template <>
LbInstruction Simulator::InstParser<LbInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  LbInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// LbInstruction1.cpp  @0x41dab0
void LbInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// LbInstruction2.cpp  @0x422450
void LbInstruction::parser_operation()
{
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rs_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  MemAccessor mem(_G.GLB[seg]);
  int32_t value = (int8_t)mem.MemAt<uint8_t>(off);   // LB: sign-extend byte
  result_ = value;
  set_g_gp_reg(rd_, value);
  mem_addr_ = 32 * _G.MMU_MMUItem[2 * seg] + off;
}

// LbInstruction3.cpp  @0x429360
LbInstruction::~LbInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// LbInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- LbuInstruction ----
template <>
LbuInstruction Simulator::InstParser<LbuInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  LbuInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// LbuInstruction1.cpp  @0x41dac0
void LbuInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// LbuInstruction2.cpp  @0x422510
void LbuInstruction::parser_operation()
{
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rs_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  MemAccessor mem(_G.GLB[seg]);
  uint32_t value = mem.MemAt<uint8_t>(off);   // LBU: zero-extend byte
  result_ = value;
  set_g_gp_reg(rd_, value);
  mem_addr_ = 32 * _G.MMU_MMUItem[2 * seg] + off;
}

// LbuInstruction3.cpp  @0x429300
LbuInstruction::~LbuInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// LbuInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- LhInstruction ----
template <>
LhInstruction Simulator::InstParser<LhInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  LhInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// LhInstruction1.cpp  @0x41da90
void LhInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// LhInstruction2.cpp  @0x4222f0
void LhInstruction::parser_operation()
{
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rs_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  MemAccessor mem(_G.GLB[seg]);
  int32_t value = (int16_t)mem.MemAt<uint16_t>(off);   // LH: sign-extend halfword
  result_ = value;
  set_g_gp_reg(rd_, value);
  mem_addr_ = 32 * _G.MMU_MMUItem[2 * seg] + off;
}

// LhInstruction3.cpp  @0x4267b0
LhInstruction::~LhInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// LhInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- LhuInstruction ----
template <>
LhuInstruction Simulator::InstParser<LhuInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  LhuInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// LhuInstruction1.cpp  @0x41daa0
void LhuInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// LhuInstruction2.cpp  @0x4223b0
void LhuInstruction::parser_operation()
{
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rs_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  MemAccessor mem(_G.GLB[seg]);
  uint32_t value = mem.MemAt<uint16_t>(off);   // LHU: zero-extend halfword
  result_ = value;
  set_g_gp_reg(rd_, value);
  mem_addr_ = 32 * _G.MMU_MMUItem[2 * seg] + off;
}

// LhuInstruction3.cpp  @0x426760
LhuInstruction::~LhuInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// LhuInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- LwInstruction ----
template <>
LwInstruction Simulator::InstParser<LwInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  LwInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// LwInstruction1.cpp  @0x41da80
void LwInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// LwInstruction2.cpp  @0x420990
void LwInstruction::parser_operation()
{
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rs_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  MemAccessor mem(_G.GLB[seg]);
  uint32_t value = mem.MemAt<uint32_t>(off);   // LW: 32-bit little-endian word
  result_ = value;
  set_g_gp_reg(rd_, value);
  mem_addr_ = 32 * _G.MMU_MMUItem[2 * seg] + off;
}

// LwInstruction3.cpp  @0x426800
LwInstruction::~LwInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// LwInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- SbInstruction ----
template <>
SbInstruction Simulator::InstParser<SbInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  SbInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// SbInstruction1.cpp  @0x41daf0
void SbInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// SbInstruction2.cpp  @0x420e30
void SbInstruction::parser_operation()
{
  rd_val_ = _G.gp_reg[rd_];
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rd_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  uint8_t *glb = _G.GLB[seg];            // base pointer wrapped by MemAccessor(_G.GLB[seg])
  glb[off + 0] = (uint8_t)(rs_val_ >> 0);   // little-endian store
  mem_addr_ = off + 32 * _G.MMU_MMUItem[2 * seg];
}

// SbInstruction3.cpp  @0x4291e0
SbInstruction::~SbInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// SbInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- ShInstruction ----
template <>
ShInstruction Simulator::InstParser<ShInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  ShInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ShInstruction1.cpp  @0x41dae0
void ShInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// ShInstruction2.cpp  @0x420d80
void ShInstruction::parser_operation()
{
  rd_val_ = _G.gp_reg[rd_];
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rd_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  uint8_t *glb = _G.GLB[seg];            // base pointer wrapped by MemAccessor(_G.GLB[seg])
  glb[off + 0] = (uint8_t)(rs_val_ >> 0);
  glb[off + 1] = (uint8_t)(rs_val_ >> 8);   // little-endian store
  mem_addr_ = off + 32 * _G.MMU_MMUItem[2 * seg];
}

// ShInstruction3.cpp  @0x4266c0
ShInstruction::~ShInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// ShInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- SwInstruction ----
template <>
SwInstruction Simulator::InstParser<SwInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  SwInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct3_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x300000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// SwInstruction1.cpp  @0x41dad0
void SwInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// SwInstruction2.cpp  @0x420cb0
void SwInstruction::parser_operation()
{
  rd_val_ = _G.gp_reg[rd_];
  rs_val_ = _G.gp_reg[rs_];
  int32_t simm = offset_;
  if (offset_ & 0x800)   // sign-extend the 12-bit offset
    simm -= 4096;
  uint32_t addr = simm + rd_val_;
  uint32_t seg = addr >> 28;            // GLB segment selected by the top 4 bits
  uint32_t off = addr & 0xFFFFFFF;      // offset inside the segment
  uint8_t *glb = _G.GLB[seg];            // base pointer wrapped by MemAccessor(_G.GLB[seg])
  glb[off + 0] = (uint8_t)(rs_val_ >> 0);
  glb[off + 1] = (uint8_t)(rs_val_ >> 8);
  glb[off + 2] = (uint8_t)(rs_val_ >> 16);
  glb[off + 3] = (uint8_t)(rs_val_ >> 24);   // little-endian store
  mem_addr_ = off + 32 * _G.MMU_MMUItem[2 * seg];
}

// SwInstruction3.cpp  @0x426710
SwInstruction::~SwInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// SwInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.
