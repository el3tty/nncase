#pragma once
// Little-endian view onto a block of simulated memory (one GLB/DDR segment).
// Lifted from IDA/Hex-Rays output (MemAccessor1..MemAccessor9 in sources/).
//
// Usage in the ISA:   MemAccessor acc(g_GLB[addr >> 28]);  acc.MemAt<uint32_t>(addr & 0xFFFFFFF);
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "math/numeric_types.h"

class MemAccessor {
public:
    // @0x46f360 (MemAccessor5)
    MemAccessor(uint8_t * base);
    // @0x46f370 (MemAccessor6)
    MemAccessor();
    // @0x46f380 (MemAccessor7).  NOTE: not virtual; the pointer lives at offset 0 of the object.
    ~MemAccessor();

    // Typed little-endian load at byte offset `off` from the base pointer.
    //   uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/...  : plain load of sizeof(T) bytes
    //   FP16::fp16, BF16::bfloat16                            : raw 16-bit pattern
    //   FP24::fp24                                            : upper 24 bits of the 32-bit word at `off`
    // Folds the nine decompiled instantiations MemAt<uint16_t> (@0x429910), MemAt<uint8_t> (@0x429b70),
    // MemAt<signed char> (@0x4461a0) and MemAt<short> (@0x446410); the typeid() chains in those dumps are
    // dead branches of one generic template body.
    template <typename T>
    T MemAt(int off) const
    {
        if constexpr (std::is_same<T, FP24::fp24>::value) {
            // verified against asm @0x429b07 (dead branch of MemAt<ushort>): reads bytes off..off+3 and, for the fp24 typeid, keeps off+1..off+3 (word >> 8)
            uint32_t word;
            std::memcpy(&word, base_ + off, sizeof word);
            return FP24::fp24(word >> 8);
        } else {
            static_assert(std::is_trivially_copyable<T>::value, "MemAt<T> needs a trivially copyable T");
            T value;
            std::memcpy(&value, base_ + off, sizeof(T));
            return value;
        }
    }

    // 16-bit little-endian load (zero-extended pattern returned as int16_t).  @0x46f390 (MemAccessor8)
    int16_t MemAt(int off) const;
    // 24-bit little-endian load.  @0x46f3b0 (MemAccessor9)
    int64_t MemAt(uint32_t off) const;

private:
    uint8_t * base_;   // +0 : start of the memory block
};
