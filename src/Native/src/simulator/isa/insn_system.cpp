// Lifted from IDA/Hex-Rays output; this instruction carries no operands.
// EndInstruction1.cpp  @0x41db90
// FenceInstruction1.cpp  @0x41dba0
// FenceIInstruction1.cpp  @0x41dbb0
// IntrInstruction1.cpp  @0x41db80
// CcrClrInstruction1.cpp  @0x41dc00
// CcrDeclInstruction1.cpp  @0x41dbe0
// CcrSetInstruction1.cpp  @0x41dbf0
// Lifted from IDA/Hex-Rays output; names and types are inferred.
// Operand/result fields are named after their role; "+NN" in the header gives the byte offset.
// Simulator18.cpp  @0x4131c0
// MmuSetidInstruction1.cpp  @0x41dc20
// Simulator19.cpp  @0x413410
// Simulator20.cpp  @0x413640
// ExtrawInstruction1.cpp  @0x41dbd0
#include "isa/insn_system.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/simulator.h"
#include "engines/ai2d.h"

// ---- EndInstruction ----
void EndInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// EndInstruction2.cpp  @0x426490
EndInstruction::~EndInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// EndInstruction3.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40db5e.
template <>
EndInstruction Simulator::InstParser<EndInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  EndInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rs_ = kinst_bits(raw, 7, 5);
  inst.reserved_12_ = kinst_bits(raw, 12, 4);
  inst.rs_val_ = g_gp_reg[inst.rs_];
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x200000006LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- FenceInstruction ----
void FenceInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// FenceInstruction2.cpp  @0x426440
FenceInstruction::~FenceInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// FenceInstruction3.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40e1d2.
template <>
FenceInstruction Simulator::InstParser<FenceInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  FenceInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.reserved0_ = kinst_bits(raw, 7, 9);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000000LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- FenceIInstruction ----
void FenceIInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// FenceIInstruction2.cpp  @0x4263f0
FenceIInstruction::~FenceIInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// FenceIInstruction3.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40dd3f.
template <>
FenceIInstruction Simulator::InstParser<FenceIInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  FenceIInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.reserved0_ = kinst_bits(raw, 7, 9);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000000LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- IntrInstruction ----
void IntrInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// IntrInstruction2.cpp  @0x4264e0
IntrInstruction::~IntrInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// IntrInstruction3.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40e47b.
template <>
IntrInstruction Simulator::InstParser<IntrInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  IntrInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rs_ = kinst_bits(raw, 7, 5);
  inst.reserved0_ = kinst_bits(raw, 12, 4);
  inst.value_ = g_gp_reg[inst.rs_];
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x200000006LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- CcrClrInstruction ----
void CcrClrInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// CcrClrInstruction2.cpp  @0x426260
CcrClrInstruction::~CcrClrInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// CcrClrInstruction3.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40de1e.
template <>
CcrClrInstruction Simulator::InstParser<CcrClrInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  CcrClrInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.ccr_ = kinst_bits(raw, 7, 5);
  inst.reserved_12_ = kinst_bits(raw, 12, 4);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000000LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- CcrDeclInstruction ----
void CcrDeclInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// CcrDeclInstruction2.cpp  @0x426300
CcrDeclInstruction::~CcrDeclInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// CcrDeclInstruction3.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40d7d0.
template <>
CcrDeclInstruction Simulator::InstParser<CcrDeclInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  CcrDeclInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rnum_ = kinst_bits(raw, 7, 5);
  inst.reserved_12_ = kinst_bits(raw, 12, 4);
  inst.rnum_val_ = g_gp_reg[inst.rnum_];
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000000LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- CcrSetInstruction ----
void CcrSetInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// CcrSetInstruction2.cpp  @0x4262b0
CcrSetInstruction::~CcrSetInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// CcrSetInstruction3.cpp: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40e341.
template <>
CcrSetInstruction Simulator::InstParser<CcrSetInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  CcrSetInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.ccr_ = kinst_bits(raw, 7, 5);
  inst.value_ = kinst_bits(raw, 12, 4);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000000LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- MmuConfInstruction ----
template <>
MmuConfInstruction Simulator::InstParser<MmuConfInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  MmuConfInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rstart_ = kinst_bits(raw, 7, 5);
  inst.rdepth_ = kinst_bits(raw, 12, 5);
  inst.mmu_id_ = kinst_bits(raw, 17, 4);
  inst.reserved_21_ = kinst_bits(raw, 21, 11);   // verified against asm: raw[31:21]
  inst.rstart_val_ = g_gp_reg[inst.rstart_];
  inst.rdepth_val_ = g_gp_reg[inst.rdepth_];
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// MmuConfInstruction1.cpp  @0x41dc10
void MmuConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// MmuConfInstruction2.cpp  @0x41e840
void MmuConfInstruction::operation()
{
  AI2D_Ai2dInst[mmu_id_ + 110] = rstart_val_;
  AI2D_Ai2dInst[mmu_id_ + 126] = rdepth_val_;
  MMU_MMUItem[2 * mmu_id_] = rstart_val_;
  MMU_MMUItem[2 * mmu_id_ + 1] = rdepth_val_;
  g_glb_start[mmu_id_] = rstart_val_;
  g_glb_depth[mmu_id_] = rdepth_val_;
}

// MmuConfInstruction3.cpp  @0x426210
MmuConfInstruction::~MmuConfInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// MmuConfInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- MmuSetidInstruction ----
void MmuSetidInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// MmuSetidInstruction2.cpp  @0x41fc80
void MmuSetidInstruction::operation()
{
  set_g_gp_reg(rd_, (g_gp_reg[rd_] & 0xFFFFFFF) | (mmu_id_ << 28));
  result_ = g_gp_reg[rd_];
}

// MmuSetidInstruction3.cpp  @0x420f20
void MmuSetidInstruction::parser_operation()
{
  // Same effect as operation(), but writes g_gp_reg directly (bypasses set_g_gp_reg).
  result_ = (g_gp_reg[rd_] & 0xFFFFFFF) | (mmu_id_ << 28);
  g_gp_reg[rd_] = result_;
}

// MmuSetidInstruction4.cpp  @0x4261c0
MmuSetidInstruction::~MmuSetidInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// MmuSetidInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40d97b.
template <>
MmuSetidInstruction Simulator::InstParser<MmuSetidInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  MmuSetidInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.rd_ = kinst_bits(raw, 7, 5);
  inst.mmu_id_ = kinst_bits(raw, 12, 4);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000006LL;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}

// ---- SsPackShapeInstruction ----
template <>
SsPackShapeInstruction Simulator::InstParser<SsPackShapeInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  SsPackShapeInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rn_ = kinst_bits(raw, 7, 5);
  inst.rc_ = kinst_bits(raw, 12, 5);
  inst.rh_ = kinst_bits(raw, 17, 5);
  inst.rw_ = kinst_bits(raw, 22, 5);
  inst.rss_ = kinst_bits(raw, 27, 3);
  inst.reserved_30_ = kinst_bits(raw, 30, 2);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// SsPackShapeInstruction1.cpp  @0x41dc30
void SsPackShapeInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// SsPackShapeInstruction2.cpp  @0x420f50
void SsPackShapeInstruction::parser_operation()
{
  rn_val_ = g_gp_reg[rn_];
  rc_val_ = g_gp_reg[rc_];
  rh_val_ = g_gp_reg[rh_];
  rw_val_ = g_gp_reg[rw_];
  // verified against asm @0x420f50: (rs1<<48)|(rs2<<32) in 64-bit, rs3<<16 in 32-bit arithmetic, rs4 zero-extended; no masking
  packed_ = (uint64_t)(uint32_t)(rh_val_ << 16) | rw_val_ | ((uint64_t)rn_val_ << 48) | ((uint64_t)rc_val_ << 32);
  g_shape_reg[rss_] = packed_;
}

// SsPackShapeInstruction3.cpp  @0x426170
SsPackShapeInstruction::~SsPackShapeInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// SsPackShapeInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- SsPackStrideInstruction ----
template <>
SsPackStrideInstruction Simulator::InstParser<SsPackStrideInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // 32-bit instruction word (upper half is ignored)
  SsPackStrideInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  // verified against asm @0x4120f0 (and sibling parsers): register field is raw[11:7] (5 bits)
  inst.rn_ = kinst_bits(raw, 7, 5);
  inst.rc_ = kinst_bits(raw, 12, 5);
  inst.rh_ = kinst_bits(raw, 17, 5);
  inst.reserved_22_ = kinst_bits(raw, 22, 5);
  inst.rss_ = kinst_bits(raw, 27, 3);
  inst.reserved_30_ = kinst_bits(raw, 30, 2);
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0x400000006LL;   // (kind << 32) | type 6
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  inst.parser_operation();
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// SsPackStrideInstruction1.cpp  @0x41dc40
void SsPackStrideInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// SsPackStrideInstruction2.cpp  @0x420fb0
void SsPackStrideInstruction::parser_operation()
{
  rn_val_ = g_gp_reg[rn_];
  rc_val_ = g_gp_reg[rc_];
  rh_val_ = g_gp_reg[rh_];
  // verified against asm @0x420fb0: (rs1<<32)|(rs2<<16)|rs3 in 64-bit, no masking; stored in g_shape_reg (0x9f2a20)
  packed_ = (uint64_t)rh_val_ | ((uint64_t)rc_val_ << 16) | ((uint64_t)rn_val_ << 32);
  g_shape_reg[rss_] = packed_;
}

// SsPackStrideInstruction3.cpp  @0x426120
SsPackStrideInstruction::~SsPackStrideInstruction()
{
  /* vptr reset to KInstruction vtable; std::string name released by member dtor */
}

// SsPackStrideInstruction4: duplicate of the preceding definition (C1/C2 constructor alias) omitted.

// ---- ExtrwInstruction ----
namespace {
// Extract `width` bits of `word` starting at bit `lo`.
inline uint32_t bits(uint32_t word, unsigned lo, unsigned width)
{
  return (word >> lo) & ((1u << width) - 1u);
}
// Replace byte `index` (0 = least significant) of a 32-bit register.
inline void set_byte(uint32_t &word, unsigned index, uint8_t value)
{
  word = (word & ~(0xFFu << (8 * index))) | ((uint32_t)value << (8 * index));
}
}  // namespace

// ExtrwInstruction1.cpp  @0x41dbc0
void ExtrwInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// ExtrwInstruction2.cpp  @0x41ef40
void ExtrwInstruction::operation()
{
  // AI2D::GetAI2D() singleton (lazy init elided). The singleton is kept as a raw word array in
  // globals.h (AI2D_Ai2dInst); word index n is the AI2D member at byte offset 4*n.
  // TODO(layout): name the AI2D registers (ai2d.h has them as fNNN members).
  uint32_t *ai2d = AI2D_Ai2dInst;
  const uint32_t v = reg_value_;

  if (extrd_ <= 0x8F) {
    switch (extrd_ >> 2) {
      case 0:  ai2d[34] = v; break;                          // AI2D+136
      case 1:  ai2d[35] = v; break;                          // AI2D+140
      case 2:  ai2d[36] = v; break;                          // AI2D+144
      case 3:  ai2d[37] = v; break;                          // AI2D+148
      case 4:  ai2d[40] = v; break;                          // AI2D+160
      case 5:  ai2d[41] = v; break;                          // AI2D+164
      case 6:  ai2d[42] = v; break;                          // AI2D+168
      case 7:  ai2d[43] = v; break;                          // AI2D+172
      case 8:  ai2d[63] = bits(v, 0, 16); ai2d[64] = bits(v, 16, 16); break;   // AI2D+252 / +256
      case 9:  ai2d[65] = bits(v, 0, 16); ai2d[66] = bits(v, 16, 16); break;   // AI2D+260 / +264
      case 10: ai2d[67] = bits(v, 0, 16); ai2d[68] = bits(v, 16, 16); break;   // AI2D+268 / +272
      case 11: ai2d[69] = bits(v, 0, 16); ai2d[70] = bits(v, 16, 16); break;   // AI2D+276 / +280
      case 12: ai2d[46] = v; break;                          // AI2D+184
      case 13: ai2d[47] = v; break;                          // AI2D+188
      case 14: ai2d[49] = v; break;                          // AI2D+196
      case 15: ai2d[50] = v; break;                          // AI2D+200
      case 16:
      case 17:
      case 27:
        break;                                               // no register behind these addresses
      case 18: {
        ai2d[78] = v >> 28;                                  // AI2D+312
        ai2d[77] = bits(v, 24, 4);                           // AI2D+308
        ai2d[84] = bits(v, 20, 4);                           // AI2D+336
        // 8-bit field [19:12]; if any of its upper nibble [19:16] is set it is treated as negative (value - 32)
        uint32_t field = bits(v, 12, 8);
        ai2d[83] = (bits(v, 16, 4) != 0) ? (uint32_t)((int)field - 32) : field;   // AI2D+332
        ai2d[90] = bits(v, 10, 2);                           // AI2D+360
        ai2d[58] = bits(v, 8, 2);                            // AI2D+232
        ai2d[59] = bits(v, 6, 2);                            // AI2D+236
        ai2d[62] = bits(v, 3, 3);                            // AI2D+248
        ai2d[61] = bits(v, 0, 3);                            // AI2D+244
        ai2d[60] = bits(v, 0, 3);                            // AI2D+240
        break;
      }
      case 19:
        *reinterpret_cast<uint8_t *>(&ai2d[79]) = bits(v, 16, 1);   // AI2D+316 (byte store)
        ai2d[85] = bits(v, 0, 16);                           // AI2D+340
        break;
      case 20: ai2d[94] = bits(v, 0, 12);  ai2d[95] = bits(v, 12, 12);  break;   // AI2D+376 / +380
      case 21: ai2d[96] = bits(v, 0, 12);  ai2d[97] = bits(v, 12, 12);  break;   // AI2D+384 / +388
      case 22: ai2d[98] = bits(v, 0, 12);  ai2d[99] = bits(v, 12, 12);  break;   // AI2D+392 / +396
      case 23: ai2d[100] = bits(v, 0, 12); ai2d[101] = bits(v, 12, 12); break;   // AI2D+400 / +404
      case 24:
        ai2d[104] = bits(v, 0, 12);                          // AI2D+416
        ai2d[105] = bits(v, 12, 12);                         // AI2D+420
        set_byte(ai2d[92], 0, bits(v, 24, 8));               // AI2D+368 byte 0
        break;
      case 25:
        *reinterpret_cast<uint8_t *>(&ai2d[93]) = bits(v, 27, 1);                      // AI2D+372 byte 0
        *(reinterpret_cast<uint8_t *>(&ai2d[93]) + 1) = bits(v, 26, 1);                // AI2D+372 byte 1
        ai2d[82] = bits(v, 25, 1);                           // AI2D+328
        ai2d[81] = bits(v, 24, 1);                           // AI2D+324
        set_byte(ai2d[92], 3, bits(v, 16, 8));               // AI2D+368 byte 3
        set_byte(ai2d[92], 2, bits(v, 8, 8));                // AI2D+368 byte 2
        set_byte(ai2d[92], 1, bits(v, 0, 8));                // AI2D+368 byte 1
        break;
      case 26: ai2d[102] = bits(v, 0, 12); ai2d[103] = bits(v, 12, 12); break;   // AI2D+408 / +412
      case 28: ai2d[89] = bits(v, 16, 10); ai2d[88] = bits(v, 0, 10); break;     // AI2D+356 / +352
      case 29: ai2d[87] = bits(v, 16, 10); ai2d[86] = bits(v, 0, 10); break;     // AI2D+348 / +344
      case 30: ai2d[71] = bits(v, 16, 13); ai2d[72] = bits(v, 0, 13); break;     // AI2D+284 / +288
      case 31:
        ai2d[80]  = v >> 31;                                 // AI2D+320
        ai2d[106] = bits(v, 30, 1);                          // AI2D+424
        ai2d[73]  = bits(v, 16, 13);                         // AI2D+292
        ai2d[74]  = bits(v, 0, 13);                          // AI2D+296
        break;
      case 32: ai2d[48] = v; break;                          // AI2D+192
      case 33: ai2d[51] = v; break;                          // AI2D+204
      case 34: ai2d[38] = bits(v, 0, 13); ai2d[39] = bits(v, 16, 13); break;     // AI2D+152 / +156
      case 35:
        ai2d[107] = v >> 31;                                 // AI2D+428
        ai2d[45]  = bits(v, 16, 13);                         // AI2D+180
        ai2d[44]  = bits(v, 0, 13);                          // AI2D+176
        break;
    }
  }

  // Only register 35 (byte address 140) ever reaches this: writing it starts the AI2D engine.
  if (extrd_ == 140)
    reinterpret_cast<AI2D *>(AI2D_Ai2dInst)->ai2d_proc();
}

// ExtrwInstruction3.cpp  @0x4263a0
ExtrwInstruction::~ExtrwInstruction() = default;

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40e064. The inlined code stores no info word (stack slot left as is); 0 is used here.
template <>
ExtrwInstruction Simulator::InstParser<ExtrwInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  ExtrwInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.extrd_ = kinst_bits(raw, 7, 10);
  inst.rs_ = kinst_bits(raw, 17, 5);
  inst.imm_ = kinst_bits(raw, 22, 10);
  inst.reg_value_ = g_gp_reg[inst.rs_];
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}

// ---- ExtrawInstruction ----
void ExtrawInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// ExtrawInstruction2.cpp  @0x41ec50
void ExtrawInstruction::operation()
{
  // AI2D::GetAI2D() singleton (lazy init elided), kept as a raw word array in globals.h;
  // word index n is the AI2D member at byte offset 4*n.
  // TODO(layout): name the AI2D registers (ai2d.h has them as fNNN members).
  uint32_t *ai2d = AI2D_Ai2dInst;

  if (extrd_ <= 0x1F) {
    // verified against asm @0x41ec50: jump table (0x48046c) is indexed by reg_addr >> 2 (16-bit), the stored
    // value is reg_value (+0x38); targets map to AI2D+0x88,0x8c,0x90,0x94,0xa0,0xa4,0xa8,0xac.
    switch (extrd_ >> 2) {
      case 0: ai2d[34] = reg_value_; break;   // AI2D+136
      case 1: ai2d[35] = reg_value_; break;   // AI2D+140
      case 2: ai2d[36] = reg_value_; break;   // AI2D+144
      case 3: ai2d[37] = reg_value_; break;   // AI2D+148
      case 4: ai2d[40] = reg_value_; break;   // AI2D+160
      case 5: ai2d[41] = reg_value_; break;   // AI2D+164
      case 6: ai2d[42] = reg_value_; break;   // AI2D+168
      case 7: ai2d[43] = reg_value_; break;   // AI2D+172
    }
  }
}

// ExtrawInstruction3.cpp  @0x426350
ExtrawInstruction::~ExtrawInstruction() = default;

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40f8c1. The inlined code stores no info word (stack slot left as is); 0 is used here.
template <>
ExtrawInstruction Simulator::InstParser<ExtrawInstruction, 32>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 4 below)
  const uint64_t raw = **pcw;          // only the low 32 bits are consumed
  ExtrawInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.extrd_ = kinst_bits(raw, 7, 10);
  inst.rs_ = kinst_bits(raw, 17, 5);
  inst.imm_ = kinst_bits(raw, 22, 10);
  inst.reg_value_ = g_gp_reg[inst.rs_];
  inst.pc_ = (uint32_t)(uintptr_t)*pcw - (uint32_t)(uintptr_t)g_DDR;   // offset inside DDR image
  inst.info_ = 0;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}
