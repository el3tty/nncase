#pragma once
// Shared helpers for the PU / PU-PDP0 configuration instructions (lifted from IDA output).
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <memory>
#include <ostream>
#include "engines/checkpoint.h"

#include "globals.h"  // _G.debug_flag, _G.PSUM_L1, ...

namespace pu {

// info@16 of every PU instruction: low u32 = 3, high u32 = 4 (instruction type / kind).
constexpr uint64_t kInstInfo = 0x400000003ULL;

// Extract `width` bits starting at bit `lo` of the 32-bit instruction word.
inline uint32_t bits(uint32_t raw, unsigned lo, unsigned width)
{
    return (raw >> lo) & ((1u << width) - 1u);
}

// 16-bit lane `i` (0 = lowest) of a packed shape register _G.shape_reg[n]
// (four 16-bit dimensions per 64-bit register).
inline uint32_t shape_word(uint64_t shape, unsigned i)
{
    return static_cast<uint32_t>((shape >> (16u * i)) & 0xFFFFu);
}

// Reference to a field at a raw byte offset inside another class's singleton object.
// TODO(layout): used where the target class layout is not yet reconstructed.
template <class T>
inline T& at(void* base, size_t byte_off)
{
    return *reinterpret_cast<T*>(static_cast<uint8_t*>(base) + byte_off);
}

// Queue of shared_ptr<...> held by Conv2D / PDP0 (16-byte elements). The element type is not
// reconstructed; shared_ptr<void> has the same layout and ownership semantics.
using OpaqueQueue = std::deque<std::shared_ptr<void>>;

// CheckPoint trace: one zero-padded 18-wide word per configuration instruction.
// The original writes to the object at CheckPoint offset 0 (stream kPu0); no std::hex is set in the
// code because the check-point streams are opened in hex mode.
inline void log_inst_word(uint64_t word)
{
    std::ostream& os = CheckPoint::GetCheckPoint()->Stream(CheckPoint::kPu0);
    os << std::setfill('0') << std::setw(18) << word << std::endl;
}

}  // namespace pu
