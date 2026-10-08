// Lifted from IDA/Hex-Rays output; names and types are inferred.
#include "engines/tilehelper.h"
#include "globals.h"
#include <cmath>
#include <cstring>

// @0x46f3d0 (TileHelper1)
TileHelper::TileHelper() {}

// @0x46f3e0 (TileHelper2)
TileHelper::~TileHelper() {}

// @0x46f3f0 (TileHelper3)
TileHelper TileHelper::GetRange(uint32_t tile_index, uint32_t tile_size, uint32_t total)
{
  TileHelper r;
  uint32_t first = tile_size * tile_index;
  uint32_t last = first + tile_size;
  if (last > total)
    last = total;
  r.begin_ = first;
  r.end_ = last;
  r.size_ = last - first;
  return r;
}

// @0x46f410 (TileHelper4)
int64_t TileHelper::GetAddress(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                               uint32_t stride0, uint32_t stride1, uint32_t stride2, uint8_t layout)
{
  uint32_t base = stride0 * d0;
  if (layout == 1)
    return stride2 * (stride1 * (d1 + base) + d2) + d3;
  return base + d3 + stride2 * d2 + stride1 * d1;
}

// @0x46f450 (TileHelper5)
int64_t TileHelper::GetSize(uint32_t *dims)
{
  return dims[3] * dims[2] * dims[1] * dims[0];
}

// @0x46f460 (TileHelper6)
int64_t TileHelper::Align(uint32_t value, uint32_t alignment)
{
  // The original uses vcvtsi2sd / vdivsd / vroundsd(ceil) / vcvttsd2si on the 64-bit registers.
  int64_t blocks = (int64_t)std::ceil((double)value / (double)alignment);
  return (uint32_t)(blocks * (int)alignment);
}

// @0x46f490 (TileHelper7)
int64_t TileHelper::BF16ToQint(uint16_t bf16_bits)
{
  uint32_t sign = bf16_bits >> 15;
  uint8_t mant = (uint8_t)(bf16_bits | 0x80);          // 7 mantissa bits + implicit one
  uint32_t exp = (uint8_t)(bf16_bits >> 7);            // biased exponent (low 8 bits)
  int16_t shift = (int16_t)(134 - exp);
  int32_t sign_mul = 1 - 2 * (int32_t)sign;
  if (shift <= 0)
    return (mant << (exp + 122)) * sign_mul;           // left shift by (exp - 134); count is masked like on x86
  if (shift <= 8) {
    uint8_t rounded = (uint8_t)((((int)mant >> shift) & 1) + (127 >> (8 - shift)));
    return sign_mul * ((int)(uint16_t)(mant + rounded) >> shift);
  }
  return 0;
}

// @0x46f520 (TileHelper8)
int64_t TileHelper::Int32ToBF16(int value, signed char exp_bias)
{
  if (value == 0)
    return 0;
  int mag = value < 0 ? -value : value;
  int lead = (int)norm_int(mag);                       // left-normalisation shift count
  uint32_t bits = (((uint32_t)(mag << (lead + 1))) >> 8) & 0x7FFFFF
                | ((uint32_t)(uint8_t)(-98 - exp_bias - (lead + 1)) << 23)
                | ((uint32_t)value & 0x80000000u);
  float as_float;
  std::memcpy(&as_float, &bits, sizeof(as_float));
  // verified against asm @0x46f520: vucomiss xmm1,xmm1 ; jp -> 0x7FC0, i.e. the NaN check of the inlined bfloat16 rounding
  if (std::isnan(as_float))
    return 0x7FC0;
  return (bits + ((bits >> 16) & 1) + 0x7FFF) >> 16;   // round to nearest even
}
