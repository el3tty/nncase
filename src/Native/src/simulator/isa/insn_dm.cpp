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
inline uint32_t bits(uint32_t word, unsigned lo, unsigned width) {
  return (word >> lo) & ((1u << width) - 1u);
}
}  // namespace

template <>
DmConf_broadcastInstruction Simulator::InstParser<DmConf_broadcastInstruction, 16>(uint8_t ** pc) {
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
  inst.reserved_ = bits(raw, 13, 3);
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 2;   // 16-bit instruction (asm: InstParser<...,16>, addq $2)
  return inst;
}

void DmConf_broadcastInstruction::get_next_pc() {
  next_pc_ = pc_ + 2;
}

DmConf_broadcastInstruction::~DmConf_broadcastInstruction() = default;

// ---- DmLoadAct0 ----
namespace {

using LoadAct0Queue = std::deque<std::shared_ptr<DmLoadAct0>>;
}  // namespace

template <>
DmLoadAct0Instruction Simulator::InstParser<DmLoadAct0Instruction, 32>(uint8_t ** pc) {
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
  inst.reserved_  = bits(raw, 25, 7);
  inst.raddr_s_val_   = _G.gp_reg[inst.raddr_s_];
  inst.rlen_val_    = _G.gp_reg[inst.rlen_];
  inst.raddr_s_mmu_addr_ = (inst.raddr_s_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_s_val_ >> 28)];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

void DmLoadAct0Instruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void DmLoadAct0Instruction::operation() {
  Dm *dm = Dm::GetDm();
  // Dm singleton: LoadAct0 section
  dm->loadact0_src_ = reinterpret_cast<const int16_t *>(_G.GLB[raddr_s_val_ >> 28] + (raddr_s_val_ & 0xFFFFFFF));  // GLB source pointer
  dm->loadact0_use_pdp0_ = dest_channel_;          // queue selector
  dm->loadact0_flag_ = is_by_channel_ != 0;

  std::shared_ptr<DmLoadAct0> load = dm->GetLoadAct0();
  if (dest_channel_) {
    PDP0::GetPDP0()->load_act0_queue_.push_back(load);
  } else {
    Conv2D::GetConv2D()->act0_param_queue_.push_back(load);
  }
}

DmLoadAct0Instruction::~DmLoadAct0Instruction() = default;

// ---- DmLoadL1ConfInstruction ----
template <>
DmLoadL1ConfInstruction Simulator::InstParser<DmLoadL1ConfInstruction, 32>(uint8_t ** pc) {
  DmLoadL1ConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.conf_id_  = bits(raw, 13, 4);
  inst.rstride_s_   = bits(raw, 17, 3);
  inst.datatype_       = bits(raw, 20, 2);
  inst.l1_type_  = bits(raw, 22, 2);
  inst.reserved_  = bits(raw, 24, 8);
  inst.rstride_s_val_      = _G.shape_reg[inst.rstride_s_];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

void DmLoadL1ConfInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void DmLoadL1ConfInstruction::operation() {
  Dm *dm = Dm::GetDm();
  dm->l1_shape_[0] = (rstride_s_val_ >> 32) & 0xFFFF;   // L1 load shape dim0
  dm->l1_shape_[1] = (rstride_s_val_ >> 16) & 0xFFFF;   // L1 load shape dim1
  dm->l1_shape_[2] = rstride_s_val_ & 0xFFFF;           // L1 load shape dim2
  dm->l1_shape_[3] = 0;                                 // dim3 (always cleared)
  dm->l1_mode_ = datatype_;                             // L1 load mode (consumed by Dm::GetDmLoadL1)
}

DmLoadL1ConfInstruction::~DmLoadL1ConfInstruction() = default;

// ---- DmLoadL1Instruction ----
template <>
DmLoadL1Instruction Simulator::InstParser<DmLoadL1Instruction, 32>(uint8_t ** pc) {
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
  inst.reserved_  = bits(raw, 28, 4);
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

void DmLoadL1Instruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

// DmLoadL1Instruction::operation  @0x423140 (reconstructed from the assembly)
void DmLoadL1Instruction::operation() {
  if (flag_)
    _G.debug_flag = 1;   // byte at 0x5cc7b0

  Dm *dm = Dm::GetDm();
  // the four 16-bit dims of the shape register (dim0 = [63:48] ... dim3 = [15:0])
  dm->l1_dims_[0] = (uint32_t)(rshape_val_ >> 48);
  dm->l1_dims_[1] = (uint32_t)((rshape_val_ >> 32) & 0xFFFF);
  dm->l1_dims_[2] = (uint32_t)((rshape_val_ >> 16) & 0xFFFF);
  dm->l1_dims_[3] = (uint32_t)(rshape_val_ & 0xFFFF);
  // GLB source pointer (bank[31:28] | offset[27:0])
  dm->l1_src_ = (const uint8_t *)(_G.GLB[raddr_s_val_ >> 28] + (raddr_s_val_ & 0xFFFFFFF));
  // the value of the second register (rs_aux), not the MMU-translated address
  dm->l1_layout_ = rhtoc_window_val_;

  // The tile is loaded into IF_L1 right away from a first snapshot ...
  Dm::LoadL1(dm->GetDmLoadL1());
  // .. and a second snapshot is queued on Conv2D (std::deque<shared_ptr<DmLoadL1>> at ).
  Conv2D::GetConv2D()->if_queue_.push_back(dm->GetDmLoadL1());
}

DmLoadL1Instruction::~DmLoadL1Instruction() = default;

// ---- DmLoadWConf2Instruction ----
template <>
DmLoadWConf2Instruction Simulator::InstParser<DmLoadWConf2Instruction, 32>(uint8_t ** pc) {
  DmLoadWConf2Instruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.conf_id_  = bits(raw, 13, 4);
  inst.rgroups_        = bits(raw, 17, 5);
  inst.rgoc_        = bits(raw, 22, 5);
  inst.reserved_  = bits(raw, 27, 5);
  inst.rgroups_val_lo8_ = (uint8_t)_G.gp_reg[inst.rgroups_];
  inst.rgoc_val_    = _G.gp_reg[inst.rgoc_];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

void DmLoadWConf2Instruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

DmLoadWConf2Instruction::~DmLoadWConf2Instruction() = default;

// ---- DmLoadWConf_deqInstruction ----
template <>
DmLoadWConf_deqInstruction Simulator::InstParser<DmLoadWConf_deqInstruction, 32>(uint8_t ** pc) {
  DmLoadWConf_deqInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.conf_id_  = bits(raw, 13, 4);
  inst.quant_type_   = bits(raw, 17, 2);
  inst.reserved_  = bits(raw, 19, 13);
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

void DmLoadWConf_deqInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void DmLoadWConf_deqInstruction::operation() {
  Dm::GetDm()->loadw_deq_mode_ = quant_type_;
}

DmLoadWConf_deqInstruction::~DmLoadWConf_deqInstruction() = default;

// ---- DmLoadWConfInstruction ----
template <>
DmLoadWConfInstruction Simulator::InstParser<DmLoadWConfInstruction, 32>(uint8_t ** pc) {
  DmLoadWConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.conf_id_  = bits(raw, 13, 4);
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

void DmLoadWConfInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void DmLoadWConfInstruction::operation() {
  Dm *dm = Dm::GetDm();
  dm->conf_a_ = kernel_h_;
  dm->conf_b_ = kernel_w_;
  dm->loadw_len_ = (uint32_t)rstride_oc_val_;
}

DmLoadWConfInstruction::~DmLoadWConfInstruction() = default;

// ---- DmLoadW ----
namespace {
using LoadWQueue = std::deque<std::shared_ptr<DmLoadW>>;

inline uint32_t mmu_translate(uint32_t glb_addr) {
  return (glb_addr & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (glb_addr >> 28)];
}
}  // namespace

template <>
DmLoadWInstruction Simulator::InstParser<DmLoadWInstruction, 32>(uint8_t ** pc) {
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
  inst.reserved_  = bits(raw, 28, 4);
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

void DmLoadWInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void DmLoadWInstruction::operation() {
  if (flag_)
    _G.debug_flag = 1;

  Dm *dm = Dm::GetDm();
  // Dm singleton: LoadW section
  dm->loadw_src0_ = _G.GLB[raddr_s_val_ >> 28] + (raddr_s_val_ & 0xFFFFFFF);  // GLB pointer 0
  dm->loadw_src1_ = _G.GLB[raddr_bw_val_ >> 28] + (raddr_bw_val_ & 0xFFFFFFF);  // GLB pointer 1
  dm->loadw_use_pdp0_ = dest_type_;
  dm->loadw_lines_ = (uint16_t)r_iochannels_val_lo32_;   // line count (byte size = loadw_len_ * loadw_lines_)

  std::shared_ptr<DmLoadW> load = dm->GetLoadW();
  if (dest_type_) {
    PDP0::GetPDP0()->weight_queue_.push_back(load);
  } else {
    Conv2D::GetConv2D()->weight_queue_.push_back(load);
  }
}

DmLoadWInstruction::~DmLoadWInstruction() = default;

// ---- DmStoreOfConfInstruction ----
template <>
DmStoreOfConfInstruction Simulator::InstParser<DmStoreOfConfInstruction, 32>(uint8_t ** pc) {
  DmStoreOfConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.conf_id_  = bits(raw, 13, 4);
  inst.rstride_d_   = bits(raw, 17, 3);
  inst.datatype_       = bits(raw, 20, 2);
  inst.reserved_  = bits(raw, 22, 10);
  inst.rstride_d_val_      = _G.shape_reg[inst.rstride_d_];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

void DmStoreOfConfInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void DmStoreOfConfInstruction::operation() {
  Dm *dm = Dm::GetDm();
  dm->of_shape_[0] = (rstride_d_val_ >> 32) & 0xFFFF;   // OF store shape dim0
  dm->of_shape_[1] = (rstride_d_val_ >> 16) & 0xFFFF;   // dim1
  dm->of_shape_[2] = rstride_d_val_ & 0xFFFF;           // dim2
  dm->of_shape_[3] = 0;                                 // dim3 (always cleared)
  dm->of_mode_ = datatype_;                             // OF store mode (consumed by Dm::GetStoreOf)
}

DmStoreOfConfInstruction::~DmStoreOfConfInstruction() = default;

// ---- DmStoreOf ----
template <>
DmStoreOfInstruction Simulator::InstParser<DmStoreOfInstruction, 32>(uint8_t ** pc) {
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
  inst.reserved_  = bits(raw, 22, 10);
  inst.raddr_d_val_   = _G.gp_reg[inst.raddr_d_];
  inst.rshape_val_      = _G.shape_reg[inst.rshape_];
  inst.raddr_d_mmu_addr_ = (inst.raddr_d_val_ & 0xFFFFFFF) + 32 * _G.MMU_MMUItem[2 * (inst.raddr_d_val_ >> 28)];
  inst.info_   = 0x400000004ULL;   // {kind = 4, type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

void DmStoreOfInstruction::get_next_pc() {
  next_pc_ = pc_ + 4;
}

void DmStoreOfInstruction::operation() {
  Dm * dm = Dm::GetDm();
  const uint32_t bank = raddr_d_val_ >> 28;

  // Dm singleton: OF-store section
  dm->of_dst_ = _G.GLB[bank] + (raddr_d_val_ & 0xFFFFFFF);          // GLB destination pointer
  dm->of_full_shape_[0] = (rshape_val_ >> 48) & 0xFFFF;              // shape dim0
  dm->of_full_shape_[1] = (rshape_val_ >> 32) & 0xFFFF;              // shape dim1
  dm->of_full_shape_[2] = (rshape_val_ >> 16) & 0xFFFF;              // shape dim2
  dm->of_full_shape_[3] = rshape_val_ & 0xFFFF;                      // shape dim3
  // verified against asm @0x422750: the binary masks the offset with 0xFFFFFF (24 bits) here, while the decoder
  // (dst_mmu_addr) uses 0xFFFFFFF (28 bits); is recomputed from dst_addr.
  dm->of_mmu_addr_ = 32 * _G.MMU_MMUItem[2 * bank] + (raddr_d_val_ & 0xFFFFFF);
  dm->of_use_pdp0_ = src_channel_;

  std::shared_ptr<DmStoreOf> store = dm->GetStoreOf();
  if (src_channel_) {
    PDP0 * pdp0 = PDP0::GetPDP0();
    pdp0->store_of_queue_.push_back(store);
    if (!pdp0->store_of_queue_.empty())
      pdp0->Compute();
  } else {
    Conv2D * conv = Conv2D::GetConv2D();
    conv->store_queue_.push_back(store);
    // Only release a compute that is already waiting (mode 1 defers it until its store is queued);
    // otherwise the store just stays queued until PuCompute / Act0 consume it.
    if (!conv->compute_queue_.empty())
      conv->Compute();
  }
}

DmStoreOfInstruction::~DmStoreOfInstruction() = default;
