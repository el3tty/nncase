// Lifted from IDA/Hex-Rays output; names and types are inferred.
// Ai2dComputeInstruction1.cpp  @0x41dfe0
#include "isa/insn_ai2d.h"
#include "isa/kinstruction.h"
#include "engines/simulator.h"
#include "globals.h"

// ---- Ai2dComputeInstruction ----
void Ai2dComputeInstruction::get_next_pc()
{
  next_pc_ = pc_ + 2;
}

// Ai2dComputeInstruction2.cpp  @0x424f00
Ai2dComputeInstruction::~Ai2dComputeInstruction() = default;

// Inlined into main() in the binary (no standalone symbol); reconstructed from the decoder at main @0x40fb5a.
template <>
Ai2dComputeInstruction Simulator::InstParser<Ai2dComputeInstruction, 16>(uint8_t ** pc)
{
  uint64_t **pcw = (uint64_t **)pc;   // instruction stream cursor (advanced by 2 below)
  const uint64_t raw = **pcw;          // only the low 16 bits are consumed
  Ai2dComputeInstruction inst;
  inst.taken_ = 0;
  inst.flag_ = 0;
  inst.opcode_ = raw & 0x7F;
  inst.reserved0_ = kinst_bits(raw, 7, 9);
  inst.pc_ = KPU_PC(*pcw);   // offset inside DDR image
  // verified against asm: inst.info is not written by the decoder (left as constructed)
  inst.pc_rel_ = inst.pc_ - start_pc_;   // TODO(layout): Simulator::start_pc (code base offset)
  *pcw = (uint64_t *)((char *)*pcw + 2);
  return inst;
}
