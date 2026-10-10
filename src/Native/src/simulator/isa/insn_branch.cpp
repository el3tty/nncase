#include "isa/insn_branch.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- BeqInstruction ----
template <>
BeqInstruction Simulator::InstParser<BeqInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  BeqInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rs1_ = kinst_bits(raw, 7, 5);
  inst.rs2_ = kinst_bits(raw, 12, 5);
  inst.cond_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.rs1_val_ = _G.gp_reg[inst.rs1_];
  inst.rs2_val_ = _G.gp_reg[inst.rs2_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

void BeqInstruction::operation() {
  int32_t offset = offset_;
  if (offset_ >> 11)   // sign-extend
    offset -= 4096;
  if (rs1_val_ == rs2_val_) {
    taken_ = 1;
    next_pc_ = pc_ + offset;
  } else {
    taken_ = 0;
    next_pc_ = 0;
  }
}

void BeqInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

BeqInstruction::~BeqInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// BeqInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- BneInstruction ----
template <>
BneInstruction Simulator::InstParser<BneInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  BneInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rs1_ = kinst_bits(raw, 7, 5);
  inst.rs2_ = kinst_bits(raw, 12, 5);
  inst.cond_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.rs1_val_ = _G.gp_reg[inst.rs1_];
  inst.rs2_val_ = _G.gp_reg[inst.rs2_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

void BneInstruction::operation() {
  int32_t offset = offset_;
  if (offset_ >> 11)   // sign-extend
    offset -= 4096;
  if (rs1_val_ != rs2_val_) {
    taken_ = 1;
    next_pc_ = pc_ + offset;
  } else {
    taken_ = 0;
    next_pc_ = 0;
  }
}

void BneInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

BneInstruction::~BneInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// BneInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- BltInstruction ----
template <>
BltInstruction Simulator::InstParser<BltInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  BltInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rs1_ = kinst_bits(raw, 7, 5);
  inst.rs2_ = kinst_bits(raw, 12, 5);
  inst.cond_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.rs1_val_ = _G.gp_reg[inst.rs1_];
  inst.rs2_val_ = _G.gp_reg[inst.rs2_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

void BltInstruction::operation() {
  int32_t offset = offset_;
  if (offset_ >> 11)   // sign-extend
    offset -= 4096;
  if ((int32_t)rs1_val_ < (int32_t)rs2_val_) {
    taken_ = 1;
    next_pc_ = pc_ + offset;
  } else {
    taken_ = 0;
    next_pc_ = 0;
  }
  // verified against asm: signed compare (jge on the inverted condition); offset = imm - 4096 if bit 11 set
}

void BltInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

BltInstruction::~BltInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// BltInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- BltuInstruction ----
template <>
BltuInstruction Simulator::InstParser<BltuInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  BltuInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rs1_ = kinst_bits(raw, 7, 5);
  inst.rs2_ = kinst_bits(raw, 12, 5);
  inst.cond_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.rs1_val_ = _G.gp_reg[inst.rs1_];
  inst.rs2_val_ = _G.gp_reg[inst.rs2_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

void BltuInstruction::operation() {
  int32_t offset = offset_;
  if (offset_ >> 11)   // sign-extend
    offset -= 4096;
  if (rs1_val_ < rs2_val_) {
    taken_ = 1;
    next_pc_ = pc_ + offset;
  } else {
    taken_ = 0;
    next_pc_ = 0;
  }
}

void BltuInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

BltuInstruction::~BltuInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// BltuInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- BgeInstruction ----
template <>
BgeInstruction Simulator::InstParser<BgeInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  BgeInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rs1_ = kinst_bits(raw, 7, 5);
  inst.rs2_ = kinst_bits(raw, 12, 5);
  inst.cond_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.rs1_val_ = _G.gp_reg[inst.rs1_];
  inst.rs2_val_ = _G.gp_reg[inst.rs2_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

void BgeInstruction::operation() {
  int32_t offset = offset_;
  if (offset_ >> 11)   // sign-extend
    offset -= 4096;
  if ((int32_t)rs1_val_ >= (int32_t)rs2_val_) {
    taken_ = 1;
    next_pc_ = pc_ + offset;
  } else {
    taken_ = 0;
    next_pc_ = 0;
  }
  // verified against asm: signed compare (jl on the inverted condition); offset = imm - 4096 if bit 11 set
}

void BgeInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

BgeInstruction::~BgeInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// BgeInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- BgeuInstruction ----
template <>
BgeuInstruction Simulator::InstParser<BgeuInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  BgeuInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rs1_ = kinst_bits(raw, 7, 5);
  inst.rs2_ = kinst_bits(raw, 12, 5);
  inst.cond_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.rs1_val_ = _G.gp_reg[inst.rs1_];
  inst.rs2_val_ = _G.gp_reg[inst.rs2_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

void BgeuInstruction::operation() {
  int32_t offset = offset_;
  if (offset_ >> 11)   // sign-extend
    offset -= 4096;
  if (rs1_val_ >= rs2_val_) {
    taken_ = 1;
    next_pc_ = pc_ + offset;
  } else {
    taken_ = 0;
    next_pc_ = 0;
  }
}

void BgeuInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

BgeuInstruction::~BgeuInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// BgeuInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- JalInstruction ----
void JalInstruction::operation() {
  uint32_t offset = offset_;
  taken_ = 1;
  if (offset >> 19)   // sign-extend 20-bit offset
    offset -= 0x100000;
  next_pc_ = pc_ + offset;
}

void JalInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void JalInstruction::parser_operation() {
  link_ = pc_ + 4;
  set_g_gp_reg(rd_, link_);
}

JalInstruction::~JalInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// JalInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main in the binary (no standalone symbol); reconstructed from the decoder at main @0x40e635.
template <>
JalInstruction Simulator::InstParser<JalInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  JalInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.offset_ = kinst_bits(raw, 12, 20);   // bit-by-bit copy loop of raw[31:12]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- JalrInstruction ----
template <>
JalrInstruction Simulator::InstParser<JalrInstruction, 32>(uint8_t ** pc) {
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  JalrInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.reserved_ = kinst_bits(raw, 17, 3);
  inst.offset_ = kinst_bits(raw, 20, 12);   // verified against asm: 12-bit field raw[31:20]
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0x100000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

void JalrInstruction::operation() {
  int32_t offset = offset_;
  taken_ = 1;
  if (offset_ >> 11)   // sign-extend
    offset -= 4096;
  next_pc_ = rs_val_ + offset;   // verified against asm @0x41d9c0: taken=1, next_pc = rs1_val + sext12(imm), no "& ~1"
}

void JalrInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void JalrInstruction::parser_operation() {
  // verified against asm @0x420ef0: stores {pc+4, _G.gp_reg[rs1]} as one qword at +56/+60, then set_g_gp_reg(rd, pc+4)
  link_val_ = pc_ + 4;
  rs_val_ = _G.gp_reg[rs_];
  set_g_gp_reg(rd_, link_val_);
}

JalrInstruction::~JalrInstruction() {
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// JalrInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.
