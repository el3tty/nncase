#pragma once
// Lifted from IDA/Hex-Rays output (TileHelper: pure helper, no polymorphism).
#include <cstdint>

// Tiling helper. All members except the range result are stateless: the original
// functions never read `this`.
struct TileHelper {
    uint32_t size_;    // +0  number of elements in the range (end - begin)
    uint32_t end_;     // +4  exclusive end of the range
    uint32_t begin_;   // +8  first element of the range

    TileHelper();
    ~TileHelper();

    // Range of tile `tile_index` for tiles of `tile_size` elements, clamped to `total`.
    TileHelper GetRange(uint32_t tile_index, uint32_t tile_size, uint32_t total);
    // Element offset of (d0, d1, d2, d3) given strides; layout == 1 selects the nested (row-major) form.
    int64_t GetAddress(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                       uint32_t stride0, uint32_t stride1, uint32_t stride2, uint8_t layout);
    // Product of four dimensions.
    int64_t GetSize(uint32_t *dims);
    // Round `value` up to a multiple of `alignment` (via floating point division and ceil).
    int64_t Align(uint32_t value, uint32_t alignment);
    // bfloat16 bit pattern -> 8-bit-mantissa integer.
    int64_t BF16ToQint(uint16_t bf16_bits);
    // int32 -> bfloat16 bit pattern; `exp_bias` is subtracted from the exponent.
    int64_t Int32ToBF16(int value, signed char exp_bias);
};
