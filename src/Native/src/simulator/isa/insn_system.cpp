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
  inst.rs_val_ = _G.gp_reg[inst.rs_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.value_ = _G.gp_reg[inst.rs_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.rnum_val_ = _G.gp_reg[inst.rnum_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.rstart_val_ = _G.gp_reg[inst.rstart_];
  inst.rdepth_val_ = _G.gp_reg[inst.rdepth_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  AI2D *ai2d = AI2D::GetAI2D();
  ai2d->glb_start_[mmu_id_] = rstart_val_;
  ai2d->glb_depth_[mmu_id_] = rdepth_val_;
  _G.MMU_MMUItem[2 * mmu_id_] = rstart_val_;
  _G.MMU_MMUItem[2 * mmu_id_ + 1] = rdepth_val_;
  _G.glb_start[mmu_id_] = rstart_val_;
  _G.glb_depth[mmu_id_] = rdepth_val_;
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
  set_g_gp_reg(rd_, (_G.gp_reg[rd_] & 0xFFFFFFF) | (mmu_id_ << 28));
  result_ = _G.gp_reg[rd_];
}

// MmuSetidInstruction3.cpp  @0x420f20
void MmuSetidInstruction::parser_operation()
{
  // Same effect as operation(), but writes _G.gp_reg directly (bypasses set_g_gp_reg).
  result_ = (_G.gp_reg[rd_] & 0xFFFFFFF) | (mmu_id_ << 28);
  _G.gp_reg[rd_] = result_;
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
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  rn_val_ = _G.gp_reg[rn_];
  rc_val_ = _G.gp_reg[rc_];
  rh_val_ = _G.gp_reg[rh_];
  rw_val_ = _G.gp_reg[rw_];
  // verified against asm @0x420f50: (rs1<<48)|(rs2<<32) in 64-bit, rs3<<16 in 32-bit arithmetic, rs4 zero-extended; no masking
  packed_ = (uint64_t)(uint32_t)(rh_val_ << 16) | rw_val_ | ((uint64_t)rn_val_ << 48) | ((uint64_t)rc_val_ << 32);
  _G.shape_reg[rss_] = packed_;
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
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  rn_val_ = _G.gp_reg[rn_];
  rc_val_ = _G.gp_reg[rc_];
  rh_val_ = _G.gp_reg[rh_];
  // verified against asm @0x420fb0: (rs1<<32)|(rs2<<16)|rs3 in 64-bit, no masking; stored in _G.shape_reg (0x9f2a20)
  packed_ = (uint64_t)rh_val_ | ((uint64_t)rc_val_ << 16) | ((uint64_t)rn_val_ << 32);
  _G.shape_reg[rss_] = packed_;
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
}  // namespace

// ExtrwInstruction1.cpp  @0x41dbc0
void ExtrwInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// ExtrwInstruction2.cpp  @0x41ef40
void ExtrwInstruction::operation()
{
  AI2D *ai2d = AI2D::GetAI2D();
  const uint32_t v = reg_value_;

  if (extrd_ <= 0x8F) {
    switch (extrd_ >> 2) {
      case 0:  ai2d->src_ch_ptr_[0] = v; break;
      case 1:  ai2d->src_ch_ptr_[1] = v; break;
      case 2:  ai2d->src_ch_ptr_[2] = v; break;
      case 3:  ai2d->src_ch_ptr_[3] = v; break;
      case 4:  ai2d->dst_ch_ptr_[0] = v; break;
      case 5:  ai2d->dst_ch_ptr_[1] = v; break;
      case 6:  ai2d->dst_ch_ptr_[2] = v; break;
      case 7:  ai2d->dst_ch_ptr_[3] = v; break;
      case 8:  ai2d->src_width_layout_[0] = bits(v, 0, 16); ai2d->src_width_layout_[1] = bits(v, 16, 16); break;
      case 9:  ai2d->src_width_layout_[2] = bits(v, 0, 16); ai2d->src_width_layout_[3] = bits(v, 16, 16); break;
      case 10: ai2d->dst_width_layout_[0] = bits(v, 0, 16); ai2d->dst_width_layout_[1] = bits(v, 16, 16); break;
      case 11: ai2d->dst_width_layout_[2] = bits(v, 0, 16); ai2d->dst_width_layout_[3] = bits(v, 16, 16); break;
      case 12: ai2d->m_raw_[0] = v; break;
      case 13: ai2d->m_raw_[1] = v; break;
      case 14: ai2d->m_raw_[3] = v; break;
      case 15: ai2d->m_raw_[4] = v; break;
      case 16:
      case 17:
      case 27:
        break;                                               // no register behind these addresses
      case 18: {
        ai2d->dst_format_ = v >> 28;
        ai2d->src_format_ = bits(v, 24, 4);
        ai2d->bound_ind_ = bits(v, 20, 4);
        // 8-bit field [19:12]; if any of its upper nibble [19:16] is set it is treated as negative (value - 32)
        uint32_t field = bits(v, 12, 8);
        ai2d->shift_ = (bits(v, 16, 4) != 0) ? ((int)field - 32) : (int)field;
        ai2d->pad_mod_ = bits(v, 10, 2);
        ai2d->interpolation_ = bits(v, 8, 2);
        ai2d->cord_round_ = bits(v, 6, 2);
        ai2d->dst_channel_ = bits(v, 3, 3);
        ai2d->channel_cfg_ = bits(v, 0, 3);
        ai2d->channel_ = bits(v, 0, 3);
        break;
      }
      case 19:
        ai2d->bound_smooth_ = bits(v, 16, 1);
        ai2d->bound_val_ = bits(v, 0, 16);
        break;
      case 20: ai2d->yuv2rgb_coef_[0] = bits(v, 0, 12);  ai2d->yuv2rgb_coef_[1] = bits(v, 12, 12);  break;
      case 21: ai2d->yuv2rgb_coef_[2] = bits(v, 0, 12);  ai2d->yuv2rgb_coef_[3] = bits(v, 12, 12);  break;
      case 22: ai2d->yuv2rgb_coef_[4] = bits(v, 0, 12);  ai2d->yuv2rgb_coef_[5] = bits(v, 12, 12);  break;
      case 23: ai2d->yuv2rgb_coef_[6] = bits(v, 0, 12); ai2d->yuv2rgb_coef_[7] = bits(v, 12, 12); break;
      case 24:
        ai2d->yuv2rgb_coef_[10] = bits(v, 0, 12);
        ai2d->yuv2rgb_coef_[11] = bits(v, 12, 12);
        ai2d->const_pad_ch_[0] = bits(v, 24, 8);
        break;
      case 25:
        ai2d->is_signed_ = bits(v, 27, 1);
        ai2d->cmd_id_ = bits(v, 26, 1);
        ai2d->dst_ind_ = bits(v, 25, 1);
        ai2d->src_ind_ = bits(v, 24, 1);
        ai2d->const_pad_ch_[3] = bits(v, 16, 8);
        ai2d->const_pad_ch_[2] = bits(v, 8, 8);
        ai2d->const_pad_ch_[1] = bits(v, 0, 8);
        break;
      case 26: ai2d->yuv2rgb_coef_[8] = bits(v, 0, 12); ai2d->yuv2rgb_coef_[9] = bits(v, 12, 12); break;
      case 28: ai2d->pad_b_ = bits(v, 16, 10); ai2d->pad_t_ = bits(v, 0, 10); break;
      case 29: ai2d->pad_r_ = bits(v, 16, 10); ai2d->pad_l_ = bits(v, 0, 10); break;
      case 30: ai2d->src_height_shape_ = bits(v, 16, 13); ai2d->src_width_shape_ = bits(v, 0, 13); break;
      case 31:
        ai2d->csc_en_  = v >> 31;
        ai2d->intr_mask_ = bits(v, 30, 1);
        ai2d->dst_height_shape_  = bits(v, 16, 13);
        ai2d->dst_width_shape_  = bits(v, 0, 13);
        break;
      case 32: ai2d->m_raw_[2] = v; break;
      case 33: ai2d->m_raw_[5] = v; break;
      case 34: ai2d->src_x_ = bits(v, 0, 13); ai2d->src_y_ = bits(v, 16, 13); break;
      case 35:
        ai2d->calc_enable_ = v >> 31;
        ai2d->dst_y_  = bits(v, 16, 13);
        ai2d->dst_x_  = bits(v, 0, 13);
        break;
    }
  }

  // Only register 35 (byte address 140) ever reaches this: writing it starts the AI2D engine.
  if (extrd_ == 140)
    ai2d->ai2d_proc();
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
  inst.reg_value_ = _G.gp_reg[inst.rs_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
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
  AI2D *ai2d = AI2D::GetAI2D();

  if (extrd_ <= 0x1F) {
    // verified against asm @0x41ec50: jump table (0x48046c) is indexed by reg_addr >> 2 (16-bit), the stored
    // value is reg_value (+0x38); targets map to AI2D+0x88,0x8c,0x90,0x94,0xa0,0xa4,0xa8,0xac.
    switch (extrd_ >> 2) {
      case 0: ai2d->src_ch_ptr_[0] = reg_value_; break;
      case 1: ai2d->src_ch_ptr_[1] = reg_value_; break;
      case 2: ai2d->src_ch_ptr_[2] = reg_value_; break;
      case 3: ai2d->src_ch_ptr_[3] = reg_value_; break;
      case 4: ai2d->dst_ch_ptr_[0] = reg_value_; break;
      case 5: ai2d->dst_ch_ptr_[1] = reg_value_; break;
      case 6: ai2d->dst_ch_ptr_[2] = reg_value_; break;
      case 7: ai2d->dst_ch_ptr_[3] = reg_value_; break;
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
  inst.reg_value_ = _G.gp_reg[inst.rs_];
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  inst.info_ = 0;
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 4);
  return inst;
}
