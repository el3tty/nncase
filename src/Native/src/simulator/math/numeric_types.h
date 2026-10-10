#pragma once
// Half-precision (IEEE 754 binary16) helper type used by the K230 NPU C-model.
// 24-bit float (1 sign, 8 exponent, 15 mantissa bits) used by the reduce units of the K230 NPU C-model.
// bfloat16 helper type (1 sign, 8 exponent, 7 mantissa bits) of the K230 NPU C-model.
// REDUCE_ELEMENT: element-wise reduction operators on FP24 values used by MFU::ReduceFun and
#include <cstdint>
#include <cstddef>

// ---- fp16 ----
namespace FP16 {

// Raw 16-bit IEEE half: 1 sign, 5 exponent, 10 mantissa bits.
struct fp16 {
    uint16_t bits_;

    constexpr fp16() : bits_(0) {}
    constexpr fp16(uint16_t raw) : bits_(raw) {}

    // Exact conversion to single precision (binary16 -> binary32).
    float f_value() const;
    explicit operator float() const { return f_value(); }

    // Converts a float to binary16, round-to-nearest-even, overflow -> +/-inf.
    static fp16 round_to_fp16(float value);

    // Multiplies by 2^shift (used for the "exp shift" stage of the line-fit unit).
    fp16 fp16_2exp_shift(signed char shift) const;
};

fp16 operator+(fp16 a, fp16 b);
fp16 operator*(fp16 a, fp16 b);

} // namespace FP16

// ---- fp24 ----
namespace FP24 {

struct fp24 {
    uint32_t bits_;                                   // only the low 24 bits are significant

    constexpr fp24() : bits_(0) {}
    constexpr fp24(uint32_t raw) : bits_(raw) {}

    // The value as a binary32: the 24 bits are the upper bits of a float.
    float f_value() const;
    explicit operator float() const { return f_value(); }

    // Round a binary32 to fp24.
    static fp24 round_to_fp24(float value);
    // Round a binary64 to fp24 (round-to-nearest-even, inf on overflow, flush to zero on underflow).
    static fp24 round_to_fp24(double value);
};

fp24 operator+(fp24 a, fp24 b);

} // namespace FP24

// ---- bfloat16 ----
namespace BF16 {

struct bfloat16 {
    uint16_t bits_;

    constexpr bfloat16() : bits_(0) {}
    // Converts an integer (truncating the mantissa, no rounding).
    // Note: a non-explicit constructor taking int is kept because callers pass raw ints.
    bfloat16(int value);
    // Construct from a raw bit pattern.
    static constexpr bfloat16 from_bits(uint16_t raw) { bfloat16 b; b.bits_ = raw; return b; }

    // Round a float to bfloat16 (round-to-nearest-even; NaN -> 0x7FC0).
    static int64_t round_to_bfloat16(float value);
    // The value as a binary32 (bits << 16).
    float f_value() const;
    explicit operator float() const { return f_value(); }
    // Conversion to fp24 (bits << 8 of the float image; NaN -> 0x7FC000).
    FP24::fp24 fp24() const;
};

} // namespace BF16

// ---- bf16 ----
namespace BF16 {
// Simplified bfloat16 adder working on the 8-bit significands (no rounding).
int16_t AddTwoFp16Simp(BF16::bfloat16 lhs, BF16::bfloat16 rhs);
// Multiply; subnormal inputs/outputs are flushed to signed zero.
int64_t operator*(BF16::bfloat16 lhs, BF16::bfloat16 rhs);
} // namespace BF16

// ---- REDUCE_ELEMENT ----
struct REDUCE_ELEMENT {
    // All operators have the same shape: out = op(*lhs, *rhs).
    // In the binary the first operand arrived in `this`; a trailing 4th pointer is unused.
    typedef void (*Fn)(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *unused);

    static void re_add(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *unused = nullptr);
    static void re_sub(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *unused = nullptr);
    static void re_min(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *unused = nullptr);
    static void re_max(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *unused = nullptr);
    static void re_mul(FP24::fp24 *lhs, FP24::fp24 *rhs, FP24::fp24 *out, FP24::fp24 *unused = nullptr);
};
