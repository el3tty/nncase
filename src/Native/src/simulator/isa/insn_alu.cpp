// Lifted from IDA/Hex-Rays output; names and types are inferred.
// Operand/result fields are named after their role; "+NN" in the header gives the byte offset.
// AddInstruction1.cpp  @0x41da10
// SubInstruction1.cpp  @0x41d820
// MulInstruction1.cpp  @0x41da30
// DivInstruction1.cpp  @0x41da40
// DivuInstruction1.cpp  @0x41da50
// RemInstruction1.cpp  @0x41da60
// Simulator2.cpp  @0x410d50
// AddiInstruction1.cpp  @0x41da00
// LuiInstruction1.cpp  @0x41d9e0
// AuipcInstruction1.cpp  @0x41d9f0
#include "isa/insn_alu.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/simulator.h"

// ---- AddInstruction ----
void AddInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// AddInstruction2.cpp  @0x4207a0
void AddInstruction::parser_operation()
{
  rs1_val_ = g_gp_reg[rs1_];
  rs2_val_ = g_gp_reg[rs2_];
  set_g_gp_reg(rd_, rs1_val_ + rs2_val_);
  result_ = g_gp_reg[rd_];
}

// AddInstruction3.cpp  @0x426a30
AddInstruction::~AddInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// AddInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40bb3d.
template <>
AddInstruction Simulator::InstParser<AddInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  AddInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs1_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 5);
  inst.rs2_ = kinst_bits(raw, 22, 5);
  inst.reserved_27_ = kinst_bits(raw, 27, 5);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- SubInstruction ----
void SubInstruction::operation()
{
}

// SubInstruction2.cpp  @0x41da20
void SubInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// SubInstruction3.cpp  @0x4207f0
void SubInstruction::parser_operation()
{
  rs1_val_ = g_gp_reg[rs1_];
  rs2_val_ = g_gp_reg[rs2_];
  set_g_gp_reg(rd_, rs1_val_ - rs2_val_);
  result_ = g_gp_reg[rd_];
}

// SubInstruction4.cpp  @0x4269e0
SubInstruction::~SubInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// SubInstruction5: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40bfd0.
template <>
SubInstruction Simulator::InstParser<SubInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  SubInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs1_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 5);
  inst.rs2_ = kinst_bits(raw, 22, 5);
  inst.reserved_27_ = kinst_bits(raw, 27, 5);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- MulInstruction ----
void MulInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// MulInstruction2.cpp  @0x420840
void MulInstruction::parser_operation()
{
  rs1_val_ = g_gp_reg[rs1_];
  rs2_val_ = g_gp_reg[rs2_];
  set_g_gp_reg(rd_, rs2_val_ * rs1_val_);
  result_ = g_gp_reg[rd_];
}

// MulInstruction3.cpp  @0x426990
MulInstruction::~MulInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// MulInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40c396.
template <>
MulInstruction Simulator::InstParser<MulInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  MulInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs1_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 5);
  inst.rs2_ = kinst_bits(raw, 22, 5);
  inst.reserved_27_ = kinst_bits(raw, 27, 5);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- DivInstruction ----
void DivInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DivInstruction2.cpp  @0x420890
void DivInstruction::parser_operation()
{
  rs1_val_ = g_gp_reg[rs1_];
  rs2_val_ = g_gp_reg[rs2_];
  // verified against asm @0x420890: plain idivl, no zero / INT_MIN/-1 check (host SIGFPE)
  set_g_gp_reg(rd_, (int32_t)rs1_val_ / (int32_t)rs2_val_);
  result_ = g_gp_reg[rd_];
}

// DivInstruction3.cpp  @0x426940
DivInstruction::~DivInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// DivInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40c76e.
template <>
DivInstruction Simulator::InstParser<DivInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  DivInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs1_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 5);
  inst.rs2_ = kinst_bits(raw, 22, 5);
  inst.reserved_27_ = kinst_bits(raw, 27, 5);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- DivuInstruction ----
void DivuInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DivuInstruction2.cpp  @0x4208e0
void DivuInstruction::parser_operation()
{
  rs1_val_ = g_gp_reg[rs1_];
  rs2_val_ = g_gp_reg[rs2_];
  // verified against asm @0x4208e0: plain divl, no zero check (host SIGFPE)
  set_g_gp_reg(rd_, rs1_val_ / rs2_val_);
  result_ = g_gp_reg[rd_];
}

// DivuInstruction3.cpp  @0x4268f0
DivuInstruction::~DivuInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// DivuInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40c9a6.
template <>
DivuInstruction Simulator::InstParser<DivuInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  DivuInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs1_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 5);
  inst.rs2_ = kinst_bits(raw, 22, 5);
  inst.reserved_27_ = kinst_bits(raw, 27, 5);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- RemInstruction ----
void RemInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// RemInstruction2.cpp  @0x420930
void RemInstruction::parser_operation()
{
  rs1_val_ = g_gp_reg[rs1_];
  rs2_val_ = g_gp_reg[rs2_];
  // verified against asm @0x420930: plain idivl, no zero / INT_MIN%-1 check (host SIGFPE)
  result_ = (int32_t)rs1_val_ % (int32_t)rs2_val_;
  set_g_gp_reg(rd_, result_);
}

// RemInstruction3.cpp  @0x4268a0
RemInstruction::~RemInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// RemInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40cbd6.
template <>
RemInstruction Simulator::InstParser<RemInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  RemInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs1_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 5);
  inst.rs2_ = kinst_bits(raw, 22, 5);
  inst.reserved_27_ = kinst_bits(raw, 27, 5);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- RemuInstruction ----
template <>
RemuInstruction Simulator::InstParser<RemuInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  RemuInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs1_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 5);
  inst.rs2_ = kinst_bits(raw, 22, 5);
  inst.reserved_27_ = kinst_bits(raw, 27, 5);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// RemuInstruction1.cpp  @0x41da70
void RemuInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// RemuInstruction2.cpp  @0x420960
void RemuInstruction::parser_operation()
{
  rs1_val_ = g_gp_reg[rs1_];
  rs2_val_ = g_gp_reg[rs2_];
  // verified against asm @0x420960: plain divl, no zero check (host SIGFPE)
  result_ = (uint32_t)rs1_val_ % (uint32_t)rs2_val_;
  set_g_gp_reg(rd_, result_);
}

// RemuInstruction3.cpp  @0x426850
RemuInstruction::~RemuInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// RemuInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- AddiInstruction ----
void AddiInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// AddiInstruction2.cpp  @0x420750
void AddiInstruction::parser_operation()
{
  rs_val_ = g_gp_reg[rs_];
  int32_t simm = imm_;
  if (imm_ & 0x800)   // sign-extend the 12-bit immediate
    simm -= 4096;
  set_g_gp_reg(rd_, simm + rs_val_);
  result_ = g_gp_reg[rd_];
}

// AddiInstruction3.cpp  @0x428ee0
AddiInstruction::~AddiInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// AddiInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40bd60.
template <>
AddiInstruction Simulator::InstParser<AddiInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  AddiInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.rs_ = kinst_bits(raw, 12, 5);
  inst.funct5_ = kinst_bits(raw, 17, 3);
  inst.imm_ = kinst_bits(raw, 20, 12);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- LuiInstruction ----
void LuiInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// LuiInstruction2.cpp  @0x420700
void LuiInstruction::parser_operation()
{
  set_g_gp_reg(rd_, imm_ << 12);
  result_ = g_gp_reg[rd_];
}

// LuiInstruction3.cpp  @0x426ad0
LuiInstruction::~LuiInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// LuiInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40c200.
template <>
LuiInstruction Simulator::InstParser<LuiInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  LuiInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.imm_ = kinst_bits(raw, 12, 20);   // bit-by-bit copy loop of raw[31:12]
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- AuipcInstruction ----
void AuipcInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// AuipcInstruction2.cpp  @0x420730
void AuipcInstruction::parser_operation()
{
  result_ = pc_ + (imm_ << 12);
  set_g_gp_reg(rd_, result_);
}

// AuipcInstruction3.cpp  @0x426a80
AuipcInstruction::~AuipcInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// AuipcInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40c5d0.
template <>
AuipcInstruction Simulator::InstParser<AuipcInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  AuipcInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.imm_ = kinst_bits(raw, 12, 20);   // bit-by-bit copy loop of raw[31:12]
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 6;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}
