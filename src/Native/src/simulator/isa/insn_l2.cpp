// Lifted from IDA/Hex-Rays output; names and types are inferred.
#include <cstring>
#include "isa/insn_l2.h"
#include "globals.h"
#include "engines/l2load.h"
#include "isa/kinstruction.h"
#include "engines/simulator.h"
#include "isa/pu_common.h"
#include "engines/l2store.h"

// ---- L2LoadConfInstruction ----
namespace {
inline uint32_t bits(uint32_t word, unsigned lo, unsigned width)
{
  return (word >> lo) & ((1u << width) - 1u);
}
// Reinterpret a (mis-sized) global scalar as the 32-bit field it really is.
template <class T> inline uint32_t &as_u32(T &x) { return reinterpret_cast<uint32_t &>(x); }
}  // namespace

// Simulator21.cpp  @0x413870
template <>
L2LoadConfInstruction Simulator::InstParser<L2LoadConfInstruction, 32>(uint8_t ** pc)
{
  L2LoadConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.rstride_d_  = bits(raw, 7, 3);
  inst.rstride_s_  = bits(raw, 10, 3);
  inst.l2_datatype_      = bits(raw, 13, 2);
  inst.ddr_datatype_      = bits(raw, 15, 3);
  inst.reserved_18_  = bits(raw, 18, 14);
  inst.rstride_d_val_     = _G.shape_reg[inst.rstride_d_];
  inst.rstride_s_val_     = _G.shape_reg[inst.rstride_s_];
  inst.info_   = 0x400000001ULL;   // {kind = 1 (L2 load), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// L2LoadConfInstruction1.cpp  @0x41dc50
void L2LoadConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// L2LoadConfInstruction2.cpp  @0x41ea60
void L2LoadConfInstruction::operation()
{
  // L2Load::GetL2Load() singleton (lazy init elided); the object lives at 0x54C440.
  // verified against asm @0x41ea60: dword stores at +0/+4/+8/+12/+16/+20/+24/+28, byte stores at +0x50/+0x51
  L2Load &l2 = *reinterpret_cast<L2Load *>(_G.L2Load_L2LoadInst);
  l2.dst_dim0_  = (rstride_d_val_ >> 32) & 0xFFFF;
  l2.dst_dim1_  = (rstride_d_val_ >> 16) & 0xFFFF;
  l2.dst_pitch_ = rstride_d_val_ & 0xFFFF;
  l2.dst_dim3_  = 0;
  l2.src_dim0_  = (rstride_s_val_ >> 32) & 0xFFFF;
  l2.src_dim1_  = (rstride_s_val_ >> 16) & 0xFFFF;
  l2.src_pitch_ = rstride_s_val_ & 0xFFFF;
  l2.src_dim3_  = 0;
  l2.mode1_     = ddr_datatype_;
  l2.mode0_     = l2_datatype_;
}

// L2LoadConfInstruction3.cpp  @0x4260d0
L2LoadConfInstruction::~L2LoadConfInstruction() = default;

// ---- L2LoadInstruction ----
// Simulator25.cpp  @0x414250
template <>
L2LoadInstruction Simulator::InstParser<L2LoadInstruction, 32>(uint8_t ** pc)
{
  L2LoadInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.raddr_d_     = bits(raw, 7, 5);
  inst.raddr_s_     = bits(raw, 12, 5);
  inst.rshape_   = bits(raw, 17, 3);
  inst.reserved_20_  = bits(raw, 20, 12);
  inst.raddr_d_val_   = _G.gp_reg[inst.raddr_d_];
  inst.raddr_s_val_ = _G.gp_reg[inst.raddr_s_];
  inst.rshape_val_      = _G.shape_reg[inst.rshape_];
  inst.raddr_d_mmu_addr_ = (inst.raddr_d_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_d_val_ >> 28)];
  inst.info_   = 0x400000001ULL;   // {kind = 1 (L2 load), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// L2LoadInstruction1.cpp  @0x41dc90
void L2LoadInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// L2LoadInstruction2.cpp  @0x41e260
void L2LoadInstruction::operation()
{
  // L2Load::GetL2Load() singleton (lazy init elided), object at 0x54C440.
  // verified against asm @0x41e260: qword stores at +0x40/+0x48, dword stores at +0x54/+0x5c/+0x20/+0x24/+0x28/+0x2c/+0x58
  L2Load &l2 = *reinterpret_cast<L2Load *>(_G.L2Load_L2LoadInst);
  const uint32_t bank = raddr_d_val_ >> 28;
  l2.ddr_ptr_      = _G.DDR + raddr_s_val_;
  l2.glb_ptr_      = _G.GLB[bank] + (raddr_d_val_ & 0xFFFFFFF);
  l2.ddr_offset_   = raddr_s_val_;
  l2.glb_bank_     = bank;
  l2.batch_count_  = (rshape_val_ >> 48) & 0xFFFF;
  l2.plane_count_  = (rshape_val_ >> 32) & 0xFFFF;
  l2.row_count_    = (rshape_val_ >> 16) & 0xFFFF;
  l2.row_len_      = rshape_val_ & 0xFFFF;
  l2.glb_mmu_addr_ = (raddr_d_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * bank];
  l2.Load();
}

// L2LoadInstruction3.cpp  @0x425f90
L2LoadInstruction::~L2LoadInstruction() = default;

// ---- L2LoadWConfInstruction ----
// Simulator22.cpp  @0x413af0
template <>
L2LoadWConfInstruction Simulator::InstParser<L2LoadWConfInstruction, 32>(uint8_t ** pc)
{
  L2LoadWConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.rlen_compressed_      = bits(raw, 7, 5);
  inst.rlen_decompressed_      = bits(raw, 12, 5);
  inst.l2_datatype_      = bits(raw, 17, 2);
  inst.ddr_datatype_      = bits(raw, 19, 3);
  inst.enable_decompress_  = bits(raw, 22, 1);
  inst.reserved_23_  = bits(raw, 23, 9);
  inst.wconf_val_  = ((uint64_t)_G.gp_reg[inst.rlen_decompressed_] << 32) | _G.gp_reg[inst.rlen_compressed_];
  inst.info_   = 0x400000001ULL;   // {kind = 1 (L2 load), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// L2LoadWConfInstruction1.cpp  @0x41dc60
void L2LoadWConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// L2LoadWConfInstruction2.cpp  @0x41e790
void L2LoadWConfInstruction::operation()
{
  // L2Load::GetL2Load() singleton (lazy init elided), object at 0x54C440.
  // verified against asm @0x41e790: setne byte at +0x38, bytes at +0x53/+0x52, qword at +0x30
  L2Load &l2 = *reinterpret_cast<L2Load *>(_G.L2Load_L2LoadInst);
  l2.compressed_ = enable_decompress_ != 0;
  l2.w_mode1_    = ddr_datatype_;
  l2.w_mode0_    = l2_datatype_;
  std::memcpy(&l2.wconf_lo_, &wconf_val_, sizeof wconf_val_);   // wconf_lo + weight_count
}

// L2LoadWConfInstruction3.cpp  @0x426080
L2LoadWConfInstruction::~L2LoadWConfInstruction() = default;

// ---- L2LoadWInstruction ----
// Simulator23.cpp  @0x413d50
template <>
L2LoadWInstruction Simulator::InstParser<L2LoadWInstruction, 32>(uint8_t ** pc)
{
  L2LoadWInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.raddr_d_     = bits(raw, 7, 5);
  inst.raddr_s_     = bits(raw, 12, 5);
  inst.rvalid_c_num_     = bits(raw, 17, 5);
  inst.reserved_22_  = bits(raw, 22, 10);
  inst.raddr_d_val_   = _G.gp_reg[inst.raddr_d_];
  inst.raddr_s_val_ = _G.gp_reg[inst.raddr_s_];
  inst.rvalid_c_num_val_    = _G.gp_reg[inst.rvalid_c_num_];
  inst.raddr_d_mmu_addr_ = (inst.raddr_d_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_d_val_ >> 28)];
  inst.info_   = 0x400000001ULL;   // {kind = 1 (L2 load), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// L2LoadWInstruction1.cpp  @0x41dc70
void L2LoadWInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// L2LoadWInstruction2.cpp  @0x41eb50
void L2LoadWInstruction::operation()
{
  if (flag_)
    _G.debug_flag = 1;

  // L2Load::GetL2Load() singleton (lazy init elided), object at 0x54C440.
  // verified against asm @0x41eb50: qword stores at +0x40/+0x48, dword stores at +0x54/+0x3c/+0x5c/+0x58
  L2Load &l2 = *reinterpret_cast<L2Load *>(_G.L2Load_L2LoadInst);
  const uint32_t bank = raddr_d_val_ >> 28;
  l2.ddr_ptr_      = _G.DDR + raddr_s_val_;
  l2.glb_ptr_      = _G.GLB[bank] + (raddr_d_val_ & 0xFFFFFFF);
  l2.ddr_offset_   = raddr_s_val_;
  l2.row_len_m1_   = rvalid_c_num_val_;
  l2.glb_bank_     = bank;
  l2.glb_mmu_addr_ = (raddr_d_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * bank];
  l2.LoadW();
}

// L2LoadWInstruction3.cpp  @0x426030
L2LoadWInstruction::~L2LoadWInstruction() = default;

// ---- L2StoreConfInstruction ----
// Simulator24.cpp  @0x413fd0
template <>
L2StoreConfInstruction Simulator::InstParser<L2StoreConfInstruction, 32>(uint8_t ** pc)
{
  L2StoreConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.rstride_d_  = bits(raw, 7, 3);
  inst.rstride_s_  = bits(raw, 10, 3);
  inst.l2_datatype_    = bits(raw, 13, 2);
  inst.ddr_datatype_    = bits(raw, 15, 3);
  inst.reserved_18_  = bits(raw, 18, 14);
  inst.rstride_d_val_     = _G.shape_reg[inst.rstride_d_];
  inst.rstride_s_val_     = _G.shape_reg[inst.rstride_s_];
  inst.info_   = 0x400000002ULL;   // {kind = 2 (L2 store), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// L2StoreConfInstruction1.cpp  @0x41dc80
void L2StoreConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// L2StoreConfInstruction2.cpp  @0x41e6b0
void L2StoreConfInstruction::operation()
{
  // L2Store::GetL2Store() singleton (lazy init elided): the object at 0x53A6C0.
  L2Store * st = reinterpret_cast<L2Store *>(_G.L2Store_L2StoreInst);
  st->ddr_dim0_ = (rstride_d_val_ >> 32) & 0xFFFF;   // +0
  st->ddr_dim1_ = (rstride_d_val_ >> 16) & 0xFFFF;   // +4
  st->ddr_pitch_ = rstride_d_val_ & 0xFFFF;          // +8
  st->ddr_dim3_ = 0;                         // +12
  st->glb_dim0_ = (rstride_s_val_ >> 32) & 0xFFFF;   // +16
  st->glb_dim1_ = (rstride_s_val_ >> 16) & 0xFFFF;   // +20
  st->glb_pitch_ = rstride_s_val_ & 0xFFFF;          // +24
  st->glb_dim3_ = 0;                         // +28
  // The original stores the 16-bit mode word (mode_lo | mode_hi << 8) as a zero-extended 64-bit value.
  const uint16_t mode = (uint16_t)(l2_datatype_ | (ddr_datatype_ << 8));
  std::memset(&st->mode0_, 0, 8);
  st->mode0_ = (uint8_t)(mode & 0xFF);       // +32
  st->mode1_ = (uint8_t)(mode >> 8);         // +33
}

// L2StoreConfInstruction3.cpp  @0x425fe0
L2StoreConfInstruction::~L2StoreConfInstruction() = default;

// ---- L2StoreInstruction ----
// Simulator26.cpp  @0x4144e0
template <>
L2StoreInstruction Simulator::InstParser<L2StoreInstruction, 32>(uint8_t ** pc)
{
  L2StoreInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.raddr_d_     = bits(raw, 7, 5);
  inst.raddr_s_     = bits(raw, 12, 5);
  inst.rshape_   = bits(raw, 17, 3);
  inst.reserved_20_  = bits(raw, 20, 12);
  inst.raddr_d_val_ = _G.gp_reg[inst.raddr_d_];
  inst.raddr_s_val_   = _G.gp_reg[inst.raddr_s_];
  inst.rshape_val_      = _G.shape_reg[inst.rshape_];
  inst.raddr_s_mmu_addr_ = (inst.raddr_s_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_s_val_ >> 28)];
  inst.info_   = 0x400000002ULL;   // {kind = 2 (L2 store), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// L2StoreInstruction1.cpp  @0x41dca0
void L2StoreInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// L2StoreInstruction2.cpp  @0x41e380
void L2StoreInstruction::operation()
{
  // L2Store::GetL2Store() singleton (lazy init elided): the object at 0x53A6C0.
  L2Store * st = reinterpret_cast<L2Store *>(_G.L2Store_L2StoreInst);
  const uint32_t bank = raddr_s_val_ >> 28;
  st->ddr_ptr_ = _G.DDR + raddr_d_val_;                                        // +40
  st->glb_ptr_ = _G.GLB[bank] + (raddr_s_val_ & 0xFFFFFFF);                      // +48
  st->ddr_offset_ = raddr_d_val_;                                             // +72
  st->glb_bank_ = bank;                                                     // +80
  st->batch_count_ = (rshape_val_ >> 48) & 0xFFFF;                                // +56
  st->plane_count_ = (rshape_val_ >> 32) & 0xFFFF;                                // +60
  st->row_count_ = (rshape_val_ >> 16) & 0xFFFF;                                  // +64
  st->row_len_ = rshape_val_ & 0xFFFF;                                            // +68
  st->glb_mmu_addr_ = (raddr_s_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * bank]; // +76
  st->Store();
}

// L2StoreInstruction3.cpp  @0x425f40
L2StoreInstruction::~L2StoreInstruction() = default;
