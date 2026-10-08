// Lifted from IDA/Hex-Rays output (fp161, fp162, FP161, FP162).
// NOTE: sources/fp161.cpp and sources/fp162.cpp were overwritten by FP161/FP162 (case-insensitive file
// system), so the body of round_to_fp16 is reconstructed (fp16_2exp_shift was re-lifted from the asm) from their
// semantics, not copied from a decompilation.
// Lifted from IDA/Hex-Rays output (fp241, fp242, fp243, FP241).
// NOTE: sources/fp241.cpp was overwritten by FP241 (case-insensitive file system), so
// round_to_fp24(float) is reconstructed, not copied from a decompilation.
// Lifted from IDA/Hex-Rays output (bfloat161..bfloat164).
// Lifted from IDA/Hex-Rays output (BF161, BF162).
// REDUCE_ELEMENT operators. Lifted from IDA/Hex-Rays output (REDUCE_ELEMENT1..5).
// fp24 values are 24-bit patterns; they are widened to float with `bits << 8`.
#include <cmath>
#include <cstring>
#include "math/numeric_types.h"
#include "globals.h"

// ---- fp16 ----
namespace {

inline uint32_t float_bits(float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; }
inline float bits_float(uint32_t u) { float f; std::memcpy(&f, &u, sizeof f); return f; }

// Half -> float using the exponent-rebias trick the original code uses.
float half_to_float(uint16_t h)
{
  uint32_t o = (uint32_t)(h & 0x7FFF) << 13;       // exponent + mantissa in float position
  const uint32_t exp = o & 0x0F800000;             // shifted 5-bit exponent field
  o += 0x38000000;                                  // rebias (127 - 15) << 23
  if (exp == 0x0F800000) {                          // inf / NaN
    o += 0x38000000;
  } else if (exp == 0) {                            // zero / subnormal
    o += 0x00800000;
    o = float_bits(bits_float(o) - bits_float(113u << 23));   // minus 2^-14 (.rodata 0x5220EC)
  }
  return bits_float(o | ((uint32_t)(h & 0x8000) << 16));
}

// Float -> half, round to nearest even.  Returns 16 significant bits.
uint16_t float_to_half(float f)
{
  const uint32_t u = float_bits(f);
  const uint32_t sign = (u >> 16) & 0x8000;
  uint32_t a = u & 0x7FFFFFFF;
  uint32_t h;
  if (a <= 0x477FFFFF) {                            // finite and representable range
    if (a > 0x387FFFFF) {                           // normal half
      h = (a + ((a >> 13) & 1) - 939520001u) >> 13; // 939520001 = 0x38000000 - 0xFFF
    } else {                                        // subnormal half: add 0.5f magic
      h = float_bits(bits_float(a) + 0.5f);         // low bits hold the mantissa
    }
  } else {
    h = (a < 0x7F800001) ? 0x7C00 : 0x7E00;         // overflow/inf -> inf, NaN -> qNaN
  }
  return (uint16_t)(sign | (h & 0xFFFF));
}

inline bool half_is_nan(uint16_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0; }

// |x| above the largest finite half value, i.e. +/-inf (.rodata 0x481410).
inline bool is_inf_half(float x) { return std::fabs(x) > 65504.0f && !std::isnan(x); }
}  // namespace

namespace FP16 {

float fp16::f_value() const
{
  return half_to_float(bits_);
}

// @0x4455b0 (fp161)
fp16 fp16::round_to_fp16(float value)
{
  return fp16(float_to_half(value));
}

// @0x445720 (fp162)
// verified against asm @0x445720: bit-level multiplication by 2^shift on the fp16 pattern (no float arithmetic).
// Inf/NaN pass through; subnormals are renormalised / shifted; exponent overflow gives the largest finite value.
fp16 fp16::fp16_2exp_shift(signed char shift) const
{
  auto clz32 = [](uint32_t v) -> int {            // @0x46b420 (0 -> 32)
    if (v == 0) return 32;
    int n = 0;
    while (!(v & 0x80000000u)) { ++n; v <<= 1; }
    return n;
  };
  uint32_t bx = bits_;
  const uint32_t frac = bx & 0x3FF;
  const int e = (bx >> 10) & 0x1F;
  const int clz = clz32(frac);
  if (((bx >> 8) & 0x7C) == 0x7C)
    return *this;                                  // inf / NaN
  const int s = static_cast<int8_t>(e + shift);    // low byte of (exp + shift), signed compares
  uint32_t eax;
  if (s > 0 && e == 0) {                           // subnormal scaled up (@0x4457a4)
    const int a = 0x20 - clz;
    const int ecx = clz - 0x15;
    if (a == 0 || s < ecx) {
      eax = (frac << (shift & 31)) & 0x3FF;
    } else {
      uint32_t esi = static_cast<uint32_t>(s - 10 + a) & 0x1F;
      eax = (frac << (ecx & 31)) & 0x3FF;
      bx &= ~0x7C00u;
      esi <<= 10;
      bx |= esi;
      if (static_cast<uint16_t>(esi) == 0x7C00) {  // exponent overflow
        bx = (bx & ~0x7C00u) | 0x7800u;
        eax = 0x3FF;
      }
    }
    return fp16(static_cast<uint16_t>((bx & 0xFC00u) | eax));
  }
  if (s <= 0 && e == 0) {                          // subnormal scaled down (@0x445838)
    eax = (frac >> ((-static_cast<int>(shift)) & 31)) & 0x3FF;
    return fp16(static_cast<uint16_t>((bx & 0xFC00u) | eax));
  }
  if (s <= 0) {                                    // normal -> subnormal (@0x445808)
    const int ecx = -static_cast<int>(shift) - e + 1;
    eax = ((0x400u + frac) >> (ecx & 31)) & 0x3FF;
    return fp16(static_cast<uint16_t>((bx & 0x8000u) | eax));
  }
  if (s <= 30) {                                   // normal result (@0x445860)
    bx = (bx & ~0x7C00u) | ((static_cast<uint32_t>(s) & 0x1F) << 10);
    return fp16(static_cast<uint16_t>(bx));
  }
  return fp16(static_cast<uint16_t>((bx & 0x8000u) + 0x7BFF));   // overflow: largest finite
}

// @0x445870 (FP161)
fp16 operator+(fp16 a, fp16 b)
{
  const float fa = a.f_value(), fb = b.f_value();
  uint16_t r = float_to_half(fa + fb);
  if (is_inf_half(fa) && is_inf_half(fb) && ((a.bits_ ^ b.bits_) & 0x8000))
    r = 0x7E00;                                     // inf - inf = NaN
  if (half_is_nan(r))
    r &= 0x7FFF;                                    // NaN results are always positive
  return fp16(r);
}

// @0x445a60 (FP162)
fp16 operator*(fp16 a, fp16 b)
{
  const float fa = a.f_value(), fb = b.f_value();
  uint16_t r = float_to_half(fa * fb);
  const bool a_zero = (a.bits_ & 0x7FFF) == 0;
  const bool b_zero = (b.bits_ & 0x7FFF) == 0;
  if ((is_inf_half(fa) && b_zero) || (a_zero && is_inf_half(fb)))
    r = 0x7E00;                                     // inf * 0 = NaN
  if (is_inf_half(fa) && is_inf_half(fb) && ((a.bits_ ^ b.bits_) & 0x8000))
    r = 0xFC00;                                     // inf * -inf = -inf
  if (half_is_nan(r))
    r &= 0x7FFF;
  return fp16(r);
}

} // namespace FP16

// ---- fp24 ----
namespace FP24 {

// @0x46b0e0 (fp243)
float fp24::f_value() const
{
  const uint32_t u = bits_ << 8;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// @0x4447c0 (fp241)
// verified against asm @0x4447c0: NaN -> 0x7FC000; otherwise round-to-nearest-even on the binary32 bit pattern:
// (bits + 0x7F + ((bits >> 8) & 1)) >> 8.  (It does not use the double overload.)
fp24 fp24::round_to_fp24(float value)
{
  uint32_t u;
  std::memcpy(&u, &value, sizeof u);
  if (std::isnan(value))
    return fp24(0x7FC000u);
  return fp24((u + 0x7Fu + ((u >> 8) & 1u)) >> 8);
}

// @0x444820 (fp242)
fp24 fp24::round_to_fp24(double value)
{
  uint64_t d;
  std::memcpy(&d, &value, sizeof d);
  const uint32_t sign = (uint32_t)(d >> 63) << 23;             // fp24 sign bit
  if (std::isnan(value))
    return fp24(sign | 0x7FC000u);
  if (std::isinf(value))
    return fp24(sign | 0x7F8000u);

  const uint64_t exp = (d >> 52) & 0x7FF;                       // biased binary64 exponent
  const uint64_t frac = d & 0xFFFFFFFFFFFFFULL;                 // 52-bit mantissa
  // fp24 exponent field = exp - (1023 - 127) = exp - 896.
  if (exp <= 0x370)                                             // too small even for a subnormal
    return fp24(sign);
  if (exp == 896) {                                             // exp field 0: first subnormal binade
    const uint64_t r = ((frac >> 38) & 1) + frac + 0x10001FFFFFFFFFULL;
    return fp24((uint32_t)((r >> 38) & 0x7FFF) | (uint32_t)(r >> 53) | sign);
  }
  if ((int)exp <= 895) {                                        // subnormal fp24: shift significand right
    const unsigned shift = (unsigned)(934 - exp);               // 39 .. 53
    const uint64_t sig = frac + (1ULL << 52);                   // add implicit one
    const uint64_t lsb = (sig >> shift) & 1;                    // round to nearest even
    const uint64_t r = (lsb + (1ULL << (shift - 1)) + frac + 0xFFFFFFFFFFFFFULL) >> shift;
    return fp24((uint32_t)r | sign);
  }
  if ((int)exp > 1150)                                          // overflow -> inf
    return fp24(sign | 0x7F8000u);

  // Normal number: keep the top 15 mantissa bits, round to nearest even.
  const uint64_t r = ((frac >> 37) & 1) + frac + 0x10000FFFFFFFFFULL;
  uint32_t exp24, man24;
  if (r & 0x20000000000000ULL) {                                // mantissa carry into exponent
    exp24 = (uint32_t)(exp - 895);
    if (exp24 > 0xFF)
      exp24 = 255;
    man24 = 0;
  } else {
    exp24 = (uint32_t)(exp - 896);
    man24 = (uint32_t)((r >> 37) & 0x7FFF);
  }
  return fp24((exp24 << 15) | man24 | sign);
}

// @0x4449e0 (FP241)
fp24 operator+(fp24 a, fp24 b)
{
  const double sum = (double)a.f_value() + (double)b.f_value();
  fp24 r = fp24::round_to_fp24(sum);
  if (std::isnan(r.f_value()))
    r.bits_ &= 0x7FFFFF;                                          // NaN results are always positive
  return r;
}

} // namespace FP24

// ---- bfloat16 ----
namespace BF16 {

// @0x444a50 (bfloat161)
bfloat16::bfloat16(int value)
{
  if (value == 0) {
    bits_ = 0;
    return;
  }
  const uint16_t sign = value <= 0 ? 1 : 0;                      // value != 0 here
  const uint32_t magnitude = value < 0 ? 0u - (uint32_t)value : (uint32_t)value;
  int msb = 0;                                                   // index of the highest set bit
  for (uint32_t t = magnitude >> 1; t; t >>= 1)
    ++msb;
  const uint32_t mantissa = msb > 7 ? magnitude >> (msb - 7)     // truncate low bits
                                    : magnitude << (7 - msb);
  bits_ = (uint16_t)(((msb + 127) << 7) | (sign << 15) | (mantissa & 0x7F));
}

// @0x444ac0 (bfloat162)
int64_t bfloat16::round_to_bfloat16(float value)
{
  if (std::isnan(value))
    return 0x7FC0;
  uint32_t u;
  std::memcpy(&u, &value, sizeof u);
  return (u + ((u >> 16) & 1) + 0x7FFF) >> 16;                   // round to nearest even
}

// @0x4673f0 (bfloat163)
float bfloat16::f_value() const
{
  const uint32_t u = (uint32_t)bits_ << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// @0x467400 (bfloat164)
FP24::fp24 bfloat16::fp24() const
{
  const uint32_t u = (uint32_t)bits_ << 16;
  if (std::isnan(f_value()))
    return FP24::fp24(0x7FC000);
  return FP24::fp24(u >> 8);
}

} // namespace BF16

// ---- bf16 ----
namespace BF16 {

// @0x444e20 (BF161)
int16_t AddTwoFp16Simp(bfloat16 a, bfloat16 b)
{
  const uint16_t x = a.bits_, y = b.bits_;
  if (a.f_value() != a.f_value() || b.f_value() != b.f_value())   // either operand is NaN
    return 0x7FC0;
  if (y == 0x7F80 || x == 0x7F80)                                 // +inf wins (even over -inf)
    return 0x7F80;
  if (x == 0xFF80 || y == 0xFF80)                                 // -inf
    return (int16_t)0xFF80;
  if (y == 0 && x == 0x8000)                                      // -0 + +0
    return (int16_t)0x8000;
  if ((x & 0x7FFF) == 0)                                          // x is +/-0 (or subnormal flush): result is y
    return (int16_t)y;
  if ((y & 0x7FFF) == 0)                                          // y is zero: result is x
    return (int16_t)x;

  // Order the operands: 'big' has the larger (or equal-to-x) exponent.
  uint8_t exp_x = (uint8_t)(x >> 7), exp_y = (uint8_t)(y >> 7);
  uint32_t big = y, small = x;
  uint8_t exp_big = exp_y, exp_small = exp_x;
  if (exp_x >= exp_y) {
    big = x;
    small = y;
    exp_big = exp_x;
    exp_small = exp_y;
  }
  const int diff = exp_big - exp_small;
  int32_t aligned = 0;                                            // smaller significand, aligned to the bigger one
  if (diff <= 8)
    aligned = ((uint8_t)(small | 0x80) >> diff) * (1 - 2 * (int)(small >> 15));
  // 8-bit significands (implicit one at bit 7), signed by the sign bit.
  const int32_t sum = ((uint8_t)big | 0x80) * (1 - 2 * (int)(big >> 15)) + aligned;
  if (sum == 0)
    return 0;
  const uint32_t magnitude = sum < 0 ? (uint32_t)-sum : (uint32_t)sum;
  const uint32_t sign = (uint32_t)sum >> 31 << 15;
  const uint8_t shift = (uint8_t)(norm_uint((int)magnitude) - 23); // left shift to bring the msb to bit 8
  const int exponent = exp_big + 1 - shift;
  if ((uint16_t)exponent == 255)                                  // overflow -> inf
    return (int16_t)(sign | 0x7F80);
  if (exponent & 0x8000)                                          // underflow -> signed zero
    return (int16_t)sign;
  return (int16_t)(((magnitude << shift >> 1) & 0x7F) | sign | ((exponent << 7) & 0x7F80));
}

// @0x444fb0 (BF162)
int64_t operator*(bfloat16 a, bfloat16 b)
{
  // Flush subnormals (exponent field == 0) to signed zero.
  bfloat16 fa = (a.bits_ & 0x7F80) ? a : bfloat16::from_bits(a.bits_ & 0x8000);
  bfloat16 fb = (b.bits_ & 0x7F80) ? b : bfloat16::from_bits(b.bits_ & 0x8000);
  const int64_t r = bfloat16::round_to_bfloat16(fb.f_value() * fa.f_value());
  if ((r & 0x7F80) == 0)                                          // flush subnormal result
    return r & 0x8000;
  return r;
}

} // namespace BF16

// ---- REDUCE_ELEMENT ----
namespace {
inline uint32_t rd24(const FP24::fp24 *p) { uint32_t v = 0; std::memcpy(&v, p, 3); return v; }
inline void wr24(FP24::fp24 *p, uint32_t v) { std::memcpy(p, &v, 4); }   // original stores a full dword
inline float f24(uint32_t v) { uint32_t u = v << 8; float f; std::memcpy(&f, &u, 4); return f; }
inline float bf2f(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline FP24::fp24 F(uint32_t v) { FP24::fp24 t{}; std::memcpy(&t, &v, sizeof(t) < 4 ? sizeof(t) : 4); return t; }
inline BF16::bfloat16 B(uint16_t v) { BF16::bfloat16 t(0); std::memcpy(&t, &v, sizeof(t) < 2 ? sizeof(t) : 2); return t; }
// fp24 -> bf16 (the operand is rounded to bfloat16 before min/max/mul).
inline uint16_t to_bf16(uint32_t v24) { return (uint16_t)BF16::bfloat16::round_to_bfloat16(f24(v24)); }
inline uint32_t to_fp24(float f) { FP24::fp24 d{}; return d.round_to_fp24(f).bits_; }
}  // namespace

// @0x444990 (REDUCE_ELEMENT1): out = lhs + rhs  (== FP24::operator+, NaN results lose their sign)
void REDUCE_ELEMENT::re_add(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *)
{
    uint32_t r = to_fp24(f24(rd24(lhs)) + f24(rd24(rhs)));
    if (std::isnan(f24(r)))
        r &= 0x7FFFFF;
    wr24(out, r);
}

// @0x444a30 (REDUCE_ELEMENT2): out = lhs - rhs, implemented as lhs + (-rhs)
void REDUCE_ELEMENT::re_sub(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *)
{
    wr24(out, FP24::operator+(F(rd24(lhs)), F(rd24(rhs) ^ 0x800000u)).bits_);
}

// @0x444b20 (REDUCE_ELEMENT3): out = min(lhs, rhs) compared as bfloat16; a NaN operand loses
void REDUCE_ELEMENT::re_min(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *)
{
    uint16_t b = to_bf16(rd24(rhs));
    uint16_t a = to_bf16(rd24(lhs));
    uint16_t sel = b;
    float fa = bf2f(a), fb = bf2f(b);
    if (!std::isnan(fa)) {
        if (std::isnan(fb) || fa <= fb) sel = a;
    }
    wr24(out, to_fp24(bf2f(sel)));
}

// @0x444ba0 (REDUCE_ELEMENT4): out = max(lhs, rhs) compared as bfloat16; a NaN operand loses
void REDUCE_ELEMENT::re_max(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *)
{
    uint16_t b = to_bf16(rd24(rhs));
    uint16_t a = to_bf16(rd24(lhs));
    uint16_t sel = a;
    float fa = bf2f(a), fb = bf2f(b);
    if (std::isnan(fa)) {
        sel = b;
    } else if (!std::isnan(fb) && fa <= fb) {
        sel = b;
    }
    wr24(out, to_fp24(bf2f(sel)));
}

// @0x4454b0 (REDUCE_ELEMENT5): out = fp24(bf16(lhs) * bf16(rhs))
void REDUCE_ELEMENT::re_mul(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *)
{
    uint16_t b = to_bf16(rd24(rhs));
    uint16_t a = to_bf16(rd24(lhs));
    uint16_t p = (uint16_t)BF16::operator*(B(a), B(b));
    wr24(out, to_fp24(bf2f(p)));
}
