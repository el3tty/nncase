// Lifted from IDA/Hex-Rays output; names and types are inferred.
#include "isa/insn_act0.h"
#include "globals.h"
#include "engines/conv2d.h"
#include "engines/pdp0.h"
#include "isa/kinstruction.h"
#include "engines/simulator.h"
#include "engines/act0.h"

// ---- Act0Compute ----
namespace {
inline uint32_t bits(uint32_t word, unsigned lo, unsigned width)
{
  return (word >> lo) & ((1u << width) - 1u);
}

using Act0ComputeQueue = std::deque<std::shared_ptr<Act0Compute>>;
}  // namespace

// Simulator56.cpp  @0x418970
template <>
Act0ComputeInstruction Simulator::InstParser<Act0ComputeInstruction, 32>(uint8_t ** pc)
{
  Act0ComputeInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.reserved_7_     = bits(raw, 7, 5);
  inst.raddr_d_     = bits(raw, 12, 5);
  inst.tcu_id_  = bits(raw, 17, 3);
  inst.channel_   = bits(raw, 20, 1);
  inst.target_     = bits(raw, 21, 2);
  inst.dest_datatype_     = bits(raw, 23, 2);
  inst.is_by_channel_     = bits(raw, 25, 1);
  inst.reserved_26_  = bits(raw, 26, 6);
  inst.raddr_d_val_    = _G.gp_reg[inst.raddr_d_];
  inst.info_   = 0x400000003ULL;   // {kind = 3 (activation), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// Act0ComputeInstruction1.cpp  @0x41dea0
void Act0ComputeInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// Act0ComputeInstruction2.cpp  @0x423010
void Act0ComputeInstruction::operation()
{
  // The "current" Act0Compute descriptor was installed by Act0Src1ConfInstruction.
  Conv2D *conv = Conv2D::GetConv2D();
  PDP0 *pdp0 = PDP0::GetPDP0();
  std::shared_ptr<Act0Compute> compute = channel_ ? pdp0->cur_act0_ : conv->cur_act0_;
  compute->out_base_ = raddr_d_val_;
  compute->out_route_ = target_;
  compute->out_type_ = dest_datatype_;
  compute->per_channel_ = is_by_channel_;

  (channel_ ? pdp0->act0_queue_ : conv->act0_queue_).push_back(compute);
}

// Act0ComputeInstruction3.cpp  @0x425540
Act0ComputeInstruction::~Act0ComputeInstruction() = default;

// ---- Act0Src1ConfInstruction ----
namespace {
template <class T> inline uint32_t &as_u32(T &x) { return reinterpret_cast<uint32_t &>(x); }
}  // namespace

// Simulator55.cpp  @0x418720
template <>
Act0Src1ConfInstruction Simulator::InstParser<Act0Src1ConfInstruction, 32>(uint8_t ** pc)
{
  Act0Src1ConfInstruction inst;
  const uint8_t *cur = *pc;
  uint32_t raw;
  std::memcpy(&raw, cur, sizeof(raw));

  inst.taken_  = 0;
  inst.flag_   = 0;
  inst.opcode_ = bits(raw, 0, 7);
  inst.tcu_id_     = bits(raw, 7, 3);
  inst.pu_id_  = bits(raw, 10, 3);
  inst.funct3_  = bits(raw, 13, 3);
  inst.channel_   = bits(raw, 16, 1);
  inst.rshape_   = bits(raw, 17, 3);
  inst.rshift_bits_ = bits(raw, 20, 5);
  inst.reserved_25_  = bits(raw, 25, 7);
  inst.rshape_val_      = _G.shape_reg[inst.rshape_];
  inst.info_   = 0x400000003ULL;   // {kind = 3 (activation), type = 4}
  inst.pc_     = KPU_PC(cur);
  inst.pc_rel_ = inst.pc_ - start_pc_;
  *pc += 4;
  return inst;
}

// Act0Src1ConfInstruction1.cpp  @0x41de90
void Act0Src1ConfInstruction::get_next_pc()
{
  next_pc_ = pc_ + 4;
}

// Act0Src1ConfInstruction2.cpp  @0x422ec0
void Act0Src1ConfInstruction::operation()
{
  // Act0::GetAct0() singleton (lazy init elided); verified against asm @0x422ec0: dword stores at Act0+0x40000..+0x40014
  Act0 *act0 = reinterpret_cast<Act0 *>(_G.Act0_act0);
  act0->batch_    = (rshape_val_ >> 48) & 0xFFFF;
  act0->channels_ = (rshape_val_ >> 32) & 0xFFFF;
  act0->height_   = (rshape_val_ >> 16) & 0xFFFF;
  act0->width_    = rshape_val_ & 0xFFFF;
  act0->shift_    = rshift_bits_;          // PSUM scale: 2^-shift
  act0->engine_   = channel_;             // target engine (0: Conv2D, 1: PDP0)

  // Install a fresh Act0Compute descriptor as the "current" one of the selected engine
  // (consumed later by Act0ComputeInstruction::operation()).
  if (channel_)
    PDP0::GetPDP0()->cur_act0_ = act0->GetAct0Compute();
  else
    Conv2D::GetConv2D()->cur_act0_ = act0->GetAct0Compute();
}

// Act0Src1ConfInstruction3.cpp  @0x425590
Act0Src1ConfInstruction::~Act0Src1ConfInstruction() = default;
