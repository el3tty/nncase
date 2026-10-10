#pragma once
// Little-endian view onto a block of simulated memory (one GLB/DDR segment).
//
// Usage in the ISA:   MemAccessor acc(_G.GLB[addr >> 28]);  acc.MemAt<uint32_t>(addr & 0xFFFFFFF);
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "math/numeric_types.h"

class MemAccessor {
public:
    MemAccessor(uint8_t * base);
    MemAccessor();
    // NOTE: not virtual; the pointer lives at offset 0 of the object.
    ~MemAccessor();

    // Typed little-endian load at byte offset `off` from the base pointer.
    //   uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/...  : plain load of sizeof(T) bytes
    //   FP16::fp16, BF16::bfloat16                            : raw 16-bit pattern
    //   FP24::fp24                                            : upper 24 bits of the 32-bit word at `off`
    // One generic template covers all instantiations (the typeid chains in the binary are dead branches).
    template <typename T>
    T MemAt(int off) const {
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

    // 16-bit little-endian load (zero-extended pattern returned as int16_t).
    int16_t MemAt(int off) const;
    // 24-bit little-endian load.
    int64_t MemAt(uint32_t off) const;

private:
    uint8_t * base_;   // start of the memory block
};
