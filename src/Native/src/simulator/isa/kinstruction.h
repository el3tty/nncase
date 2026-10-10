#pragma once
// Reconstructed from IDA/Hex-Rays output (inferred declarations).
#include <cstddef>
#include <typeinfo>
#include <xmmintrin.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <deque>
#include <map>
#include <bitset>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
// Extract `width` bits of the instruction word starting at bit `lo` (decoder helper).
inline uint32_t kinst_bits(uint64_t raw, unsigned lo, unsigned width)
{
    return (uint32_t)((raw >> lo) & ((1ull << width) - 1));
}

// Instruction word layout (as seen in the compiler dumps and in main.cpp's dispatch):
//   [6:0]   opcode class (compiler "opcode" field); 32-bit words use [31:7] for operands, 16-bit words [15:7].
//   Operands are packed in assembly order from bit 7 upward (register operand = 5 bits, shape-register operand = 3 bits).
//   Selector fields inside a class (not assembly operands): funct3 [19:17] (addi/load/store/branch/jalr),
//   funct5 [21:17] (ALU), funct4 [16:13] (dm_conf / pu_conf / pu_pdp0_conf), funct5 [11:7] (mfu_conf).
//   Per-instruction field names follow the compiler operand names; see each class header for `// asm:` syntax.
// Common base of every simulated instruction.
// Object layout (byte offsets): vptr@0, opcode@8, info@16, pc@24, pc_rel@28, next_pc@32,
// taken@36, name@40 (std::string, always empty and never read; not ported), flag@48; derived classes start their fields at +49.
// The instruction word is 32 bits (opcode = raw & 0x7F); "short" instructions advance pc by 2.
struct KInstruction {
    uint32_t opcode_;  // +8   raw & 0x7F
    uint64_t info_;  // +16  two u32: low = instruction type (6 for decoded 32-bit insns), high = kind/class (1 branch/jump, 3 mem, 4 conf)
    uint32_t pc_;  // +24  address of this instruction relative to _G.DDR
    uint32_t pc_rel_;  // +28  pc relative to the code base (Simulator::start_pc_)
    uint32_t next_pc_;  // +32  filled by get_next_pc()/operation()
    uint8_t taken_;  // +36  set by control-flow instructions when the branch/jump is taken
    uint8_t flag_;  // +48
    virtual ~KInstruction();
    virtual void operation();      // default: no-op
    virtual void get_next_pc();    // default: no-op (derived: pc + 4, or pc + 2 for 16-bit insns)
};
