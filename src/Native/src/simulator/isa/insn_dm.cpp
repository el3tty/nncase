// Lifted from IDA/Hex-Rays output; names and types are inferred.
#include "isa/insn_dm.h"
#include "globals.h"
#include "isa/kinstruction.h"
#include "engines/simulator.h"
#include "engines/conv2d.h"
#include "engines/dm.h"
#include "engines/pdp0.h"
#include "isa/pu_common.h"

// ---- DmConf_broadcastInstruction ----
namespace {
// Extract `width` bits of `word` starting at bit `lo`.
inline uint32_t bits(uint32_t word, unsigned lo, unsigned width)
{
  return (word >> lo) & ((1u << width) - 1u);
}
}  // namespace

// Simulator27.cpp  @0x414770
template <>
DmConf_broadcastInstruction Simulator::InstParser<DmConf_broadcastInstruction, 16>(uint8_t ** pc)
{
  DmConf_broadcastInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));   // the original loaded 64 bits; only the low bits are consumed

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_    = bits(raw, 7, 3);
  inst.broadcast_if_   = bits(raw, 10, 1);
  inst.broadcast_w_   = bits(raw, 11, 1);
  inst.psum_cascade_   = bits(raw, 12, 1);
  inst.reserved0_ = bits(raw, 13, 3);
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 2;   // 16-bit instruction (asm: InstParser<...,16>, addq $2)
  return inst;
}

// DmConf_broadcastInstruction1.cpp  @0x41dcb0
void DmConf_broadcastInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// DmConf_broadcastInstruction2.cpp  @0x425ef0
DmConf_broadcastInstruction::~DmConf_broadcastInstruction() = default;

// ---- DmLoadAct0 ----
namespace {

// TODO(layout): Dm / Conv2D / PDP0 singletons are accessed by raw byte offset.
template <class T> inline T &at(void *base, size_t off)
{
  return *reinterpret_cast<T *>(static_cast<char *>(base) + off);
}
using LoadAct0Queue = std::deque<std::shared_ptr<DmLoadAct0>>;
}  // namespace

// Simulator35.cpp  @0x415950
template <>
DmLoadAct0Instruction Simulator::InstParser<DmLoadAct0Instruction, 32>(uint8_t ** pc)
{
  DmLoadAct0Instruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.raddr_s_     = bits(raw, 13, 5);
  inst.rlen_     = bits(raw, 18, 5);
  inst.dest_channel_   = bits(raw, 23, 1);
  inst.is_by_channel_  = bits(raw, 24, 1);
  inst.reserved_25_  = bits(raw, 25, 7);
  inst.raddr_s_val_   = _G.gp_reg[inst.raddr_s_];
  inst.rlen_val_    = _G.gp_reg[inst.rlen_];
  inst.raddr_s_mmu_addr_ = (inst.raddr_s_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_s_val_ >> 28)];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmLoadAct0Instruction1.cpp  @0x41dd30
void DmLoadAct0Instruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmLoadAct0Instruction2.cpp  @0x422690
void DmLoadAct0Instruction::operation()
{
  char *dm = Dm::GetDm();
  // Dm singleton: LoadAct0 section
  at<uint64_t>(dm, 96)  = (uint64_t)(uintptr_t)_G.GLB[raddr_s_val_ >> 28] + (raddr_s_val_ & 0xFFFFFFF);  // GLB source pointer
  at<uint8_t>(dm, 104)  = dest_channel_;          // TODO(layout): queue selector stored in the Dm
  at<uint8_t>(dm, 105)  = is_by_channel_ != 0;

  std::shared_ptr<DmLoadAct0> load = reinterpret_cast<Dm *>(dm)->GetLoadAct0();
  if (dest_channel_) {
    // TODO(layout): PDP0 + 65800 is its std::deque<std::shared_ptr<DmLoadAct0>>
    static_cast<PDP0 *>(PDP0::GetPDP0())->load_act0_queue_.push_back(load);
  } else {
    // TODO(layout): Conv2D + 400 is its std::deque<std::shared_ptr<DmLoadAct0>>
    Conv2D::GetConv2D()->act0_param_queue_.push_back(load);
  }
}

// DmLoadAct0Instruction3.cpp  @0x425c70
DmLoadAct0Instruction::~DmLoadAct0Instruction() = default;

// ---- DmLoadL1ConfInstruction ----
// Simulator28.cpp  @0x4148d0
template <>
DmLoadL1ConfInstruction Simulator::InstParser<DmLoadL1ConfInstruction, 32>(uint8_t ** pc)
{
  DmLoadL1ConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.funct4_  = bits(raw, 13, 4);
  inst.rstride_s_   = bits(raw, 17, 3);
  inst.datatype_       = bits(raw, 20, 2);
  inst.l1_type_  = bits(raw, 22, 2);
  inst.reserved_24_  = bits(raw, 24, 8);
  inst.rstride_s_val_      = _G.shape_reg[inst.rstride_s_];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmLoadL1ConfInstruction1.cpp  @0x41dcc0
void DmLoadL1ConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmLoadL1ConfInstruction2.cpp  @0x41e030
void DmLoadL1ConfInstruction::operation()
{
  // TODO(layout): Dm singleton fields, accessed as 32-bit words (index = byte offset / 4).
  uint32_t *dm = reinterpret_cast<uint32_t *>(Dm::GetDm());
  dm[11] = (rstride_s_val_ >> 32) & 0xFFFF;   // Dm+44: L1 load shape dim0
  dm[12] = (rstride_s_val_ >> 16) & 0xFFFF;   // Dm+48: L1 load shape dim1
  dm[13] = rstride_s_val_ & 0xFFFF;           // Dm+52: L1 load shape dim2
  dm[14] = 0;                        // Dm+56: dim3 (always cleared)
  dm[15] = datatype_;                     // Dm+60: L1 load mode (consumed by Dm::GetDmLoadL1)
}

// DmLoadL1ConfInstruction3.cpp  @0x425ea0
DmLoadL1ConfInstruction::~DmLoadL1ConfInstruction() = default;

// ---- DmLoadL1Instruction ----
// Simulator33.cpp  @0x415420
template <>
DmLoadL1Instruction Simulator::InstParser<DmLoadL1Instruction, 32>(uint8_t ** pc)
{
  DmLoadL1Instruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.raddr_s_     = bits(raw, 13, 5);
  inst.rhtoc_window_     = bits(raw, 18, 5);
  inst.rshape_   = bits(raw, 23, 3);
  inst.l1_type_  = bits(raw, 26, 2);
  inst.reserved_28_  = bits(raw, 28, 4);
  inst.raddr_s_val_   = _G.gp_reg[inst.raddr_s_];
  inst.rhtoc_window_val_    = _G.gp_reg[inst.rhtoc_window_];
  inst.rshape_val_      = _G.shape_reg[inst.rshape_];
  inst.raddr_s_mmu_addr_ = (inst.raddr_s_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_s_val_ >> 28)];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmLoadL1Instruction1.cpp  @0x41dd10
void DmLoadL1Instruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmLoadL1Instruction::operation  @0x423140 (reconstructed from the assembly, IDA had no body)
void DmLoadL1Instruction::operation()
{
  if (flag_)
    _G.debug_flag = 1;   // byte at 0x5cc7b0

  Dm *dm = reinterpret_cast<Dm *>(Dm::GetDm());
  // Dm+64..76: the four 16-bit dims of the shape register (dim0 = [63:48] ... dim3 = [15:0])
  dm->l1_dims_[0] = (uint32_t)(rshape_val_ >> 48);
  dm->l1_dims_[1] = (uint32_t)((rshape_val_ >> 32) & 0xFFFF);
  dm->l1_dims_[2] = (uint32_t)((rshape_val_ >> 16) & 0xFFFF);
  dm->l1_dims_[3] = (uint32_t)(rshape_val_ & 0xFFFF);
  // Dm+80: GLB source pointer (bank[31:28] | offset[27:0])
  dm->l1_src_ = (const uint8_t *)(_G.GLB[raddr_s_val_ >> 28] + (raddr_s_val_ & 0xFFFFFFF));
  // Dm+88: the value of the second register (rs_aux), not the MMU-translated address
  dm->l1_layout_ = rhtoc_window_val_;

  // The tile is loaded into IF_L1 right away from a first snapshot ...
  Dm::LoadL1(dm->GetDmLoadL1());
  // ... and a second snapshot is queued on Conv2D (std::deque<shared_ptr<DmLoadL1>> at Conv2D+160).
  Conv2D::GetConv2D()->if_queue_.push_back(dm->GetDmLoadL1());
}

// DmLoadL1Instruction2.cpp  @0x425d10
DmLoadL1Instruction::~DmLoadL1Instruction() = default;

// ---- DmLoadWConf2Instruction ----
// Simulator32.cpp  @0x4151d0
template <>
DmLoadWConf2Instruction Simulator::InstParser<DmLoadWConf2Instruction, 32>(uint8_t ** pc)
{
  DmLoadWConf2Instruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.funct4_  = bits(raw, 13, 4);
  inst.rgroups_        = bits(raw, 17, 5);
  inst.rgoc_        = bits(raw, 22, 5);
  inst.reserved_27_  = bits(raw, 27, 5);
  inst.rgroups_val_lo8_ = (uint8_t)_G.gp_reg[inst.rgroups_];
  inst.rgoc_val_    = _G.gp_reg[inst.rgoc_];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmLoadWConf2Instruction1.cpp  @0x41dcd0
void DmLoadWConf2Instruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmLoadWConf2Instruction2.cpp  @0x425e50
DmLoadWConf2Instruction::~DmLoadWConf2Instruction() = default;

// ---- DmLoadWConf_deqInstruction ----
// Simulator30.cpp  @0x414d50
template <>
DmLoadWConf_deqInstruction Simulator::InstParser<DmLoadWConf_deqInstruction, 32>(uint8_t ** pc)
{
  DmLoadWConf_deqInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.funct4_  = bits(raw, 13, 4);
  inst.quant_type_   = bits(raw, 17, 2);
  inst.reserved_19_  = bits(raw, 19, 13);
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmLoadWConf_deqInstruction1.cpp  @0x41dcf0
void DmLoadWConf_deqInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmLoadWConf_deqInstruction2.cpp  @0x41e010
void DmLoadWConf_deqInstruction::operation()
{
  // TODO(layout): Dm singleton, +16 = weight dequantisation mode
  *reinterpret_cast<uint32_t *>(Dm::GetDm() + 16) = quant_type_;
}

// DmLoadWConf_deqInstruction3.cpp  @0x425db0
DmLoadWConf_deqInstruction::~DmLoadWConf_deqInstruction() = default;

// ---- DmLoadWConfInstruction ----
// Simulator29.cpp  @0x414b20
template <>
DmLoadWConfInstruction Simulator::InstParser<DmLoadWConfInstruction, 32>(uint8_t ** pc)
{
  DmLoadWConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.funct4_  = bits(raw, 13, 4);
  inst.kernel_h_     = bits(raw, 17, 5);
  inst.kernel_w_     = bits(raw, 22, 5);
  inst.rstride_oc_     = bits(raw, 27, 5);
  inst.rstride_oc_val_    = _G.gp_reg[inst.rstride_oc_];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmLoadWConfInstruction1.cpp  @0x41dce0
void DmLoadWConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmLoadWConfInstruction2.cpp  @0x41dff0
void DmLoadWConfInstruction::operation()
{
  // TODO(layout): Dm singleton weight-load section, accessed as 32-bit words
  uint32_t *dm = reinterpret_cast<uint32_t *>(Dm::GetDm());
  dm[1] = kernel_h_;    // Dm+4
  dm[2] = kernel_w_;    // Dm+8
  dm[3] = (uint32_t)rstride_oc_val_;   // Dm+12
}

// DmLoadWConfInstruction3.cpp  @0x425e00
DmLoadWConfInstruction::~DmLoadWConfInstruction() = default;

// ---- DmLoadW ----
namespace {
using LoadWQueue = std::deque<std::shared_ptr<DmLoadW>>;

inline uint32_t mmu_translate(uint32_t glb_addr)
{
  return (glb_addr & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (glb_addr >> 28)];
}
}  // namespace

// Simulator34.cpp  @0x4156b0
template <>
DmLoadWInstruction Simulator::InstParser<DmLoadWInstruction, 32>(uint8_t ** pc)
{
  DmLoadWInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.raddr_s_    = bits(raw, 13, 5);
  inst.raddr_bw_    = bits(raw, 18, 5);
  inst.r_iochannels_   = bits(raw, 23, 3);
  inst.dest_type_   = bits(raw, 26, 2);
  inst.reserved_28_  = bits(raw, 28, 4);
  inst.raddr_s_val_  = _G.gp_reg[inst.raddr_s_];
  inst.raddr_bw_val_  = _G.gp_reg[inst.raddr_bw_];
  inst.r_iochannels_val_lo32_ = (uint32_t)_G.shape_reg[inst.r_iochannels_];
  inst.raddr_s_mmu_addr_ = mmu_translate(inst.raddr_s_val_);
  inst.raddr_bw_mmu_addr_ = mmu_translate(inst.raddr_bw_val_);
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmLoadWInstruction1.cpp  @0x41dd20
void DmLoadWInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmLoadWInstruction2.cpp  @0x4225b0
void DmLoadWInstruction::operation()
{
  if (flag_)
    _G.debug_flag = 1;

  char *dm = Dm::GetDm();
  // Dm singleton: LoadW section
  at<uint64_t>(dm, 24) = (uint64_t)(uintptr_t)_G.GLB[raddr_s_val_ >> 28] + (raddr_s_val_ & 0xFFFFFFF);  // GLB pointer 0
  at<uint64_t>(dm, 32) = (uint64_t)(uintptr_t)_G.GLB[raddr_bw_val_ >> 28] + (raddr_bw_val_ & 0xFFFFFFF);  // GLB pointer 1
  at<uint8_t>(dm, 40)  = dest_type_;
  at<uint16_t>(dm, 42) = (uint16_t)r_iochannels_val_lo32_;   // line count (byte size = Dm+12 * Dm+42)

  std::shared_ptr<DmLoadW> load = reinterpret_cast<Dm *>(dm)->GetLoadW();
  if (dest_type_) {
    // TODO(layout): PDP0 + 65640 is its std::deque<std::shared_ptr<DmLoadW>>
    static_cast<PDP0 *>(PDP0::GetPDP0())->weight_queue_.push_back(load);
  } else {
    // TODO(layout): Conv2D + 240 is its std::deque<std::shared_ptr<DmLoadW>>
    Conv2D::GetConv2D()->weight_queue_.push_back(load);
  }
}

// DmLoadWInstruction3.cpp  @0x425cc0
DmLoadWInstruction::~DmLoadWInstruction() = default;

// ---- DmStoreOfConfInstruction ----
// Simulator31.cpp  @0x414f80
template <>
DmStoreOfConfInstruction Simulator::InstParser<DmStoreOfConfInstruction, 32>(uint8_t ** pc)
{
  DmStoreOfConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.funct4_  = bits(raw, 13, 4);
  inst.rstride_d_   = bits(raw, 17, 3);
  inst.datatype_       = bits(raw, 20, 2);
  inst.reserved_22_  = bits(raw, 22, 10);
  inst.rstride_d_val_      = _G.shape_reg[inst.rstride_d_];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmStoreOfConfInstruction1.cpp  @0x41dd00
void DmStoreOfConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmStoreOfConfInstruction2.cpp  @0x41e070
void DmStoreOfConfInstruction::operation()
{
  // TODO(layout): Dm singleton OF-store section, accessed as 32-bit words (index = byte offset / 4).
  uint32_t *dm = reinterpret_cast<uint32_t *>(Dm::GetDm());
  dm[27] = (rstride_d_val_ >> 32) & 0xFFFF;   // Dm+108: OF store shape dim0
  dm[28] = (rstride_d_val_ >> 16) & 0xFFFF;   // Dm+112: dim1
  dm[29] = rstride_d_val_ & 0xFFFF;           // Dm+116: dim2
  dm[30] = 0;                        // Dm+120: dim3 (always cleared)
  dm[31] = datatype_;                     // Dm+124: OF store mode (consumed by Dm::GetStoreOf)
}

// DmStoreOfConfInstruction3.cpp  @0x425d60
DmStoreOfConfInstruction::~DmStoreOfConfInstruction() = default;

// ---- DmStoreOf ----
// Simulator36.cpp  @0x415bc0
template <>
DmStoreOfInstruction Simulator::InstParser<DmStoreOfInstruction, 32>(uint8_t ** pc)
{
  DmStoreOfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.raddr_d_     = bits(raw, 13, 5);
  inst.rshape_   = bits(raw, 18, 3);
  inst.src_channel_   = bits(raw, 21, 1);
  inst.reserved_22_  = bits(raw, 22, 10);
  inst.raddr_d_val_   = _G.gp_reg[inst.raddr_d_];
  inst.rshape_val_      = _G.shape_reg[inst.rshape_];
  inst.raddr_d_mmu_addr_ = (inst.raddr_d_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_d_val_ >> 28)];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// DmStoreOfInstruction1.cpp  @0x41dd40
void DmStoreOfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// DmStoreOfInstruction2.cpp  @0x422750
void DmStoreOfInstruction::operation()
{
  Dm * dm = reinterpret_cast<Dm *>(Dm::GetDm());
  const uint32_t bank = raddr_d_val_ >> 28;

  // Dm singleton: OF-store section
  dm->of_dst_ = _G.GLB[bank] + (raddr_d_val_ & 0xFFFFFFF);          // +128  GLB destination pointer
  dm->of_full_shape_[0] = (rshape_val_ >> 48) & 0xFFFF;              // +136  shape dim0
  dm->of_full_shape_[1] = (rshape_val_ >> 32) & 0xFFFF;              // +140  shape dim1
  dm->of_full_shape_[2] = (rshape_val_ >> 16) & 0xFFFF;              // +144  shape dim2
  dm->of_full_shape_[3] = rshape_val_ & 0xFFFF;                      // +148  shape dim3
  // verified against asm @0x422750: the binary masks the offset with 0xFFFFFF (24 bits) here, while the decoder
  // (dst_mmu_addr) uses 0xFFFFFFF (28 bits); Dm+152 is recomputed from dst_addr.
  dm->of_mmu_addr_ = 32 * _G.MMU_MMUItem[2 * bank] + (raddr_d_val_ & 0xFFFFFF);   // +152
  dm->of_use_pdp0_ = src_channel_;                                 // +156

  std::shared_ptr<DmStoreOf> store = dm->GetStoreOf();
  if (src_channel_) {
    PDP0 * pdp0 = static_cast<PDP0 *>(PDP0::GetPDP0());
    pdp0->store_of_queue_.push_back(store);
    if (!pdp0->store_of_queue_.empty())
      pdp0->Compute();
  } else {
    Conv2D * conv = Conv2D::GetConv2D();
    conv->store_queue_.push_back(store);
    if (!conv->store_queue_.empty())
      conv->Compute();
  }
}

// DmStoreOfInstruction3.cpp  @0x425c20
DmStoreOfInstruction::~DmStoreOfInstruction() = default;
