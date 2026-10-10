// MFU: lifted from IDA/Hex-Rays output (MFU1..MFU15 in sources/).  The AVX instructions that the auto-converted
// results/ copy had stripped (vmulss/vaddss/vcomiss/vroundss/...) were recovered from sources/MFU*.cpp.
//
// Conventions used by the lifted code
//  * Tensors are addressed as   element(i, j, k, l) = l + w * (k + h * (j + c * i))   where (c, h, w) are the
//    three halfwords of a 64-bit "shape" register (MFU::GetCHW) and (i, j, k, l) run over MFU::dim_[0..3].
//  * All bfloat16 arithmetic Hex-Rays showed inline (flush-to-zero float multiply + round, the 8-bit-significand
//    adder with its inf/NaN/zero special cases) is BF16::operator* / BF16::AddTwoFp16Simp (see bf16.cpp).
//  * Hex-Rays lost the flags of every float compare; the asm shows `vcomiss`, which sets CF when "below or
//    unordered".  CfBelow() below reproduces that.
#include "engines/mfu.h"
#include "globals.h"
#include "math/numeric_types.h"
#include "engines/memaccessor.h"
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <iostream>


namespace {

using BF16::bfloat16;
using FP16::fp16;
using FP24::fp24;

// ---- small numeric helpers -------------------------------------------------------------------------------

// CF after `vcomiss a, b`: set when a < b or the operands are unordered.
inline bool CfBelow(float a, float b) { return !(a >= b); }

// vcvttss2si: truncation, 0x80000000 ("integer indefinite") for NaN / out of range.
inline int32_t TruncToInt32(float f)
{
  if (!(f > -2147483904.0f && f < 2147483648.0f))
    return INT32_MIN;
  return static_cast<int32_t>(f);
}

// verified against ELF .rodata and asm @0x438fd0 / 0x4396d0: .rodata 0x522060 = 127, 522070 = -128, 522080 = 255,
// 5220A0 = -127 (all four 32-bit lanes' first element; combined with vpminsd / vpmaxsd).
constexpr int32_t kInt8Max = 127;
constexpr int32_t kInt8MinActivate = -128;    // .rodata 0x522070
constexpr int32_t kInt8MinQuant = -127;       // .rodata 0x5220A0
constexpr int32_t kUint8Max = 255;            // .rodata 0x522080
// verified against asm @0x4394fe / 0x4397fa: .rodata 0x481410 = 0x7f7fffff (FLT_MAX); `ja` after vandps(abs) means "is +-infinity"
// (NaN is unordered and does not take it).  Only valid for an fp16 value (where it equals "> 65504") or a bf16 value
// (where finite values far above 65504 are NOT saturated by this test).
constexpr float kHalfInfLimit = 65504.0f;
constexpr float kFloatMax = 3.40282347e+38f;

inline bfloat16 B(uint16_t bits) { return bfloat16::from_bits(bits); }
inline float BfToFloat(uint16_t bits) { return B(bits).f_value(); }
inline uint16_t BfFtz(uint16_t v) { return (v & 0x7F80) ? v : static_cast<uint16_t>(v & 0x8000); }
inline uint16_t BfMul(uint16_t a, uint16_t b) { return static_cast<uint16_t>(BF16::operator*(B(a), B(b))); }
inline uint16_t BfAdd(uint16_t a, uint16_t b) { return static_cast<uint16_t>(BF16::AddTwoFp16Simp(B(a), B(b))); }
inline uint16_t BfNeg(uint16_t a) { return static_cast<uint16_t>(a ^ 0x8000); }
inline uint16_t BfRound(float f) { return static_cast<uint16_t>(bfloat16::round_to_bfloat16(f)); }

// Pointer to the start of the memory block a MemAccessor wraps (the pointer is its only member, at +0).
// TODO(layout): MemAccessor keeps `base_` private.
inline uint8_t * MemBase(const MemAccessor & acc) { return *reinterpret_cast<uint8_t * const *>(&acc); }

// GLB address -> host pointer: (addr >> 28) selects the segment, the low 28 bits are the offset.
inline uint8_t * GlbPtr(uint32_t addr) { return _G.GLB[addr >> 28] + (addr & 0xFFFFFFF); }

inline uint16_t Load16(const uint8_t * p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline void Store16(uint8_t * p, uint16_t v) { std::memcpy(p, &v, 2); }

// fp16 -> 16-bit integer, FP16::fp16::operator short().
// verified against asm @0x437960 (_ZNK4FP164fp16cvsEv): exponent field > 29 (|x| >= 32768, inf, NaN) gives +32767 / -32767
// (sign from bit 15, NOT -32768); exponent field <= 13 (|x| < 0.5) gives 0; otherwise the 11-bit significand is shifted
// by (exp - 25): left for exp >= 25, right with round-half-to-even (arithmetic shift of the signed significand) for exp <= 24.
inline int32_t Fp16ToInt16(fp16 v)
{
  const uint32_t b = v.bits_;
  const int32_t e = (b >> 10) & 0x1F;
  const int32_t s = (b >> 15) & 1;
  if (e > 29)
    return s ? -32767 : 32767;
  if (e <= 13)
    return 0;
  const int32_t m = (1 - 2 * s) * static_cast<int32_t>((b & 0x3FF) + 0x400);
  if (e >= 25)
    return m * (1 << (e - 25));
  const int32_t sh = 25 - e;
  return (m + (1 << (sh - 1)) - 1 + ((m >> sh) & 1)) >> sh;
}

}  // namespace

// ==================================================================================================================
// construction / destruction
// ==================================================================================================================

// Initial state of the singleton (@0x445da0, MFU15 zeroes the fields that are not static).
MFU::MFU() {
  Init();
}

void MFU::Reset() {
  for (std::ofstream & os : log_) {
    os.close();
    os = std::ofstream();
  }
  Init();
}

// Initializes the POD part of the object; the trace streams are left alone.
void MFU::Init() {
  reg0_ = 0;
  src_addr_ = 0;
  dst_addr_ = 0;
  aux_addr_ = 0;
  reduce_init_ = 0;
  std::fill(std::begin(dim_), std::end(dim_), 0);
  reduce_op_ = 0;
  reduce_mode_ = 0;
  shape_src_ = 0;
  shape_dst_ = 0;
  elem16_ = 0;
  trans_shape_src_ = 0;
  trans_shape_dst_ = 0;
  trans_elem16_ = 0;
  trans_type_ = 0;
  memset_value_ = 0;
  shape_grid_ = 0;
  interp_mode_ = 0;
  dequant_zero_ = 0;
  dequant_scale_ = 0;
  dequant_signed_ = 0;
  dequant_enable_ = 0;
  quant_zero_ = 0;
  quant_scale_ = 0;
  quant_signed_ = 0;
  quant_enable_ = 0;
  std::fill(std::begin(dim_sample_), std::end(dim_sample_), 0);
  busy_ = 0;
}

// @0x426e80 (MFU1): the original body only destroys the four trace streams and the (empty) containers.
MFU::~MFU() = default;

// @0x445da0 (MFU15)
void * MFU::GetMFU()
{
  static MFU instance;     // the original instance sits at 0x54BB00
  return &instance;
}

// verified against asm @0x437940 (GetCHW.constprop.0): w = shape & 0xFFFF, h = (shape >> 16) & 0xFFFF, c = (shape >> 32) & 0xFFFF.
void MFU::GetCHW(uint64_t shape, uint32_t & c, uint32_t & h, uint32_t & w)
{
  w = static_cast<uint16_t>(shape);
  h = static_cast<uint16_t>(shape >> 16);
  c = static_cast<uint16_t>(shape >> 32);
}

// ==================================================================================================================
// reduction buffer
// ==================================================================================================================

// @0x4382d0 (MFU2)
void MFU::ReduceInit(FP24::fp24 * buf, FP24::fp24 init, uint8_t op)
{
  switch (op) {
    case 0:    // max: most negative finite fp24
      for (int i = 0; i < 15; ++i) buf[i] = fp24(0xFF7FFF);
      break;
    case 1:    // min: most positive finite fp24
      for (int i = 0; i < 15; ++i) buf[i] = fp24(0x7F7FFF);
      break;
    case 2:    // add
    case 3:    // sub (accumulated as a sum)
      for (int i = 0; i < 15; ++i) buf[i] = fp24::round_to_fp24(0.0);
      break;
    case 4:    // mul
      for (int i = 0; i < 15; ++i) buf[i] = fp24::round_to_fp24(1.0);
      break;
    default:   // slots 0..14 are left alone
      break;
  }
  buf[15] = init;   // +60: the running accumulator
}

// ==================================================================================================================
// Activate
// ==================================================================================================================

namespace {

// L1Helper: (channel, row, col) view of an int32 tile.  Same raw layout as act0.cpp's L1View.
// TODO(layout): L1Helper is not lifted yet.
struct L1View {
  uint8_t * data_;        // +0
  int32_t   max_channels_; // +8   channel capacity of the buffer (24 for IF = PE rows, 32 for PSUM = PE lanes)
  int32_t   chan_stride_; // +12  bytes per channel
  int32_t   base_;        // +16  byte offset of element (0, 0, 0)
  int32_t   row_stride_;  // +20  per row index, in elements
  int32_t   col_stride_;  // +24  per column index, in elements
  int32_t   channels_;    // +28
  int32_t   height_;      // +32
  int32_t   width_;       // +36
};

// Matrix4<bfloat16>: rows (d2) x columns (d3) of bf16 parameters.  TODO(layout): Matrix4 is not lifted yet.
struct Bf16Matrix {
  uint8_t   owns_;        // +0
  uint16_t * data_;       // +8
  int32_t   d0_, d1_, d2_, d3_;   // +16..+28
};

// Matrix4::operator()(row, col): reports an out-of-range access but still performs it.
// verified against asm @0x437b70 (Matrix4<bfloat16>::operator()(iiii) constprop): see act0.cpp
uint16_t MatrixAt(const Bf16Matrix & m, int row, int col)
{
  if (m.d0_ <= 0 || m.d1_ <= 0 || row >= m.d2_ || col >= m.d3_)
    std::cout << "[Error: Matrix Exceed]" << std::endl;
  return m.data_[m.d3_ * row + col];
}

}  // namespace

// @0x438fd0 (MFU3)
// Per channel i the matrix row holds p[0..6]; every int32 element v of the tile is converted to bfloat16 and
//     x = bf16(v)
//     y = x <= knee ? p0 * x + p2 : p1 * x + p3            (bf16 multiply / bf16 add)
//     y = nearbyint(y)
//     r = y > lo ? (y < hi ? y : hi) : lo                   with lo = p4, hi = p5
// and r is truncated to an integer, clamped to the int8 / uint8 range and stored back in place.
void MFU::Activate(Matrix4<BF16::bfloat16> & params, L1Helper & psum, bool is_signed)
{
  const Bf16Matrix & m = reinterpret_cast<const Bf16Matrix &>(params);   // TODO(layout)
  const L1View & tile = reinterpret_cast<const L1View &>(psum);          // TODO(layout)

  for (int ch = 0; ch < tile.channels_; ++ch) {
    const uint16_t p0 = MatrixAt(m, ch, 0);
    const uint16_t p1 = BfFtz(MatrixAt(m, ch, 1));    // verified against asm @0x438fd0: column 1, flushed (subnormal -> signed zero)
    const uint16_t p2 = MatrixAt(m, ch, 2);
    const uint16_t p3 = MatrixAt(m, ch, 3);
    const uint16_t lo = MatrixAt(m, ch, 4);  // p[4]
    const uint16_t hi = MatrixAt(m, ch, 5);    // p[5]
    const uint16_t knee = MatrixAt(m, ch, 6);  // p[6]
    if (tile.height_ <= 0)
      continue;
    for (int row = 0; row < tile.height_; ++row) {
      for (int col = 0; col < tile.width_; ++col) {
        const int64_t word = static_cast<int64_t>(col * tile.col_stride_ + tile.row_stride_ * row)
                           + static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(tile.chan_stride_ * ch)) >> 2)
                           + static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(tile.base_)) >> 2);
        int32_t * elem = reinterpret_cast<int32_t *>(tile.data_ + 4 * static_cast<int64_t>(static_cast<int>(word)));

        const int32_t v = *elem;
        const uint16_t x = BfFtz(v == 0 ? uint16_t(0) : BF16::bfloat16(v).bits_);
        // vcomiss knee, x : the first segment applies when knee >= x
        uint16_t y;
        if (!CfBelow(BfToFloat(knee), BfToFloat(x)))
          y = BfAdd(BfMul(BfFtz(p0), x), p2);
        else
          y = BfAdd(BfMul(x, p1), p3);          // verified against asm @0x4393d0: x * ftz(p1) + p3
        const uint16_t yr = BfRound(std::nearbyint(BfToFloat(y)));

        // clamp stage (the compare chain of the decompilation reduces to this select)
        uint16_t out;
        if (CfBelow(BfToFloat(lo), BfToFloat(yr))) {          // lo < y
          out = hi;
          if (CfBelow(BfToFloat(yr), BfToFloat(hi)))           // y < hi
            out = yr;
        } else {
          out = lo;
        }

        int32_t q = TruncToInt32(BfToFloat(out));
        if (is_signed)
          q = std::max(std::min(q, kInt8Max), kInt8MinActivate);    // verified against asm @0x4392ea: [-128, 127]
        else
          q = std::max(std::min(q, kUint8Max), 0);
        *elem = q;
      }
    }
  }
}

// ==================================================================================================================
// fp16 (PDP1) quantisation helpers
// ==================================================================================================================

// @0x439480 (MFU4)
void MFU::quant_new(FP16::fp16 const & value, FP16::fp16 const & scale, FP16::fp16 const & zero_point,
                    uint8_t const & mode, signed char shift, short & out)
{
  if (std::isnan(value.f_value())) {
    out = 0;
    return;
  }
  fp16 t = value * scale;
  t = t.fp16_2exp_shift(shift);
  t = t + zero_point;

  int32_t q;
  // verified against asm @0x4394fe: `vucomiss |t|, FLT_MAX; ja` -> infinity only; NaN is unordered and uses operator short
  if (!(std::fabs(t.f_value()) > kFloatMax))
    q = Fp16ToInt16(t);
  else
    q = 0x7FFF * (1 - 2 * (t.bits_ >> 15));               // +/- infinity (asm @0x439558: (s << 15) - s)
  if (mode == 1) {                                       // uint8
    q = std::min(q, 255);
    out = static_cast<short>(std::max(q, 0));
  } else {
    int32_t r = std::max(q, -32767);                     // int16 (mode 3) is symmetric
    if (mode == 2) {                                     // int8
      q = std::min(q, 127);
      r = std::max(q, -127);
    }
    out = static_cast<short>(r);
  }
}

// @0x4395a0 (MFU5)
// verified against asm @0x4395a0: `cmpb $1, (mode); movswl q; jne skip; subl zero_point` -> the signed 16-bit zero point
// is subtracted ONLY when mode == 1 (uint8); for every other mode the raw q is converted.
void MFU::dequant_new(short const & q, FP16::fp16 const & scale, short const & zero_point,
                      uint8_t const & mode, signed char shift, FP16::fp16 & out)
{
  const int diff = static_cast<int>(q) - (mode == 1 ? static_cast<int>(zero_point) : 0);
  fp16 t = fp16::round_to_fp16(static_cast<float>(diff));
  t = t.fp16_2exp_shift(static_cast<signed char>(-static_cast<int>(shift)));
  out = t * scale;
}

// @0x439610 (MFU6)
void MFU::quant_new_MemSt(int index, MemAccessor dst, FP16::fp16 scale, FP16::fp16 zero_point,
                          uint8_t mode, uint8_t shift, FP16::fp16 & value)
{
  uint8_t * base = MemBase(dst);
  if (mode == 0) {                                        // raw fp16
    Store16(base + 2 * index, value.bits_);
    return;
  }
  short q = 0;
  quant_new(value, scale, zero_point, mode, static_cast<signed char>(shift), q);
  if (mode == 3)                                          // int16
    Store16(base + 2 * index, static_cast<uint16_t>(q));
  else
    base[index] = static_cast<uint8_t>(q);
}

// @0x43d970 (MFU14)
void MFU::dequant_new_MemAt(int index, MemAccessor src, FP16::fp16 scale, short zero_point,
                            uint8_t mode, uint8_t shift, FP16::fp16 & out)
{
  short q;
  if (mode == 1)
    q = src.MemAt<uint8_t>(index);
  else if (mode == 2)
    q = src.MemAt<signed char>(index);
  else if (mode == 3)
    q = src.MemAt<short>(2 * index);
  else {
    out = src.MemAt<FP16::fp16>(2 * index);               // not quantised: plain fp16
    return;
  }
  dequant_new(q, scale, zero_point, mode, static_cast<signed char>(shift), out);
}

// ==================================================================================================================
// bf16 (MeshNet / ReduceFun / Sample) quantisation helpers
// ==================================================================================================================

// @0x4396d0 (MFU7)
// All the fp24 conversions Hex-Rays showed as repeated round_to_fp24 calls are the bf16 -> fp24 widenings.
void MFU::quant(BF16::bfloat16 const & x, BF16::bfloat16 const & scale, uint32_t const & zero_point,
                uint8_t const & is_signed, uint8_t & out)
{
  int32_t v;
  if (std::isnan(x.f_value())) {
    v = 0;
  } else {
    const fp24 xs = fp24::round_to_fp24(x.f_value());
    const fp24 ss = fp24::round_to_fp24(scale.f_value());
    const fp24 zs = fp24::round_to_fp24(BfToFloat(static_cast<uint16_t>(zero_point)));
    const fp24 prod = fp24::round_to_fp24(xs.f_value() * ss.f_value());      // float multiply, then round
    const fp24 sum = prod + zs;
    const uint16_t rounded = BfRound(sum.f_value());
    const uint16_t q = BfRound(std::nearbyint(BfToFloat(rounded)));          // vroundss 0xC, back to bf16
    const float qf = BfToFloat(q);
    if (std::fabs(qf) > kFloatMax)                                           // verified against asm @0x4397fa: only +-inf gives +/-255
      v = (1 - 2 * (q >> 15)) * 255;
    else
      v = TruncToInt32(qf);                                                  // NaN -> INT_MIN
  }
  if (is_signed)
    v = std::max(std::min(v, kInt8Max), kInt8MinQuant);                      // verified against asm @0x439813: [-127, 127]
  else
    v = std::max(std::min(v, kUint8Max), 0);
  out = static_cast<uint8_t>(v);
}

// @0x4398a0 (MFU8)
void MFU::dequant(uint8_t const & q, BF16::bfloat16 const & scale, uint8_t const & zero_point,
                  uint8_t const & is_signed, BF16::bfloat16 & out)
{
  const int value = is_signed ? static_cast<int8_t>(q) : static_cast<int>(q);
  // verified against asm @0x4398b6 (movzbl): the zero point is read as an unsigned byte even for int8 data.
  const int diff = value - static_cast<int>(zero_point);
  const uint16_t bits = diff == 0 ? uint16_t(0) : BF16::bfloat16(diff).bits_;      // truncating int -> bf16
  out = B(static_cast<uint16_t>(BfMul(bits, scale.bits_)));
}

// ==================================================================================================================
// Memset / Memcpy / Trans
// ==================================================================================================================

// @0x439a00 (MFU9)
void MFU::Memset()
{
  const uint32_t dst = dst_addr_;
  if (elem16_ && (dst & 1)) {
    std::printf("when data type is fp16 or int16, memset addr_d must be align with 2 bytes, addr_dest = %d! \n",
                static_cast<int>(dst & 0xFFFFFFF));
    std::exit(1);
  }
  if (!dim_[0] || !dim_[1] || !dim_[2] || !dim_[3]) {
    std::puts("memset tensor shape size must not be 0! ");
    std::exit(1);
  }
  uint8_t * base = GlbPtr(dst);
  uint32_t c, h, w;
  GetCHW(shape_dst_, c, h, w);

  for (uint16_t i = 0; i < dim_[0]; ++i)
    for (uint16_t j = 0; j < dim_[1]; ++j)
      for (uint16_t k = 0; k < dim_[2]; ++k)
        for (uint16_t l = 0; l < dim_[3]; ++l) {
          const int index = static_cast<int>(w * (h * (c * i + j) + k) + l);
          if (elem16_)
            Store16(base + 2 * index, memset_value_);
          else
            base[index] = static_cast<uint8_t>(memset_value_);
        }
}

// @0x439c00 (MFU10)
void MFU::Memcpy()
{
  const uint8_t * src = GlbPtr(src_addr_);
  uint8_t * dst = GlbPtr(dst_addr_);
  uint32_t sc, sh, sw, dc, dh, dw;
  GetCHW(shape_src_, sc, sh, sw);
  GetCHW(shape_dst_, dc, dh, dw);

  for (uint16_t i = 0; i < dim_[0]; ++i)
    for (uint16_t j = 0; j < dim_[1]; ++j)
      for (uint16_t k = 0; k < dim_[2]; ++k)
        for (uint16_t l = 0; l < dim_[3]; ++l) {
          const int from = static_cast<int>(l + sw * (k + sh * (j + i * sc)));
          const int to = static_cast<int>(dw * (dh * (dc * i + j) + k) + l);
          dst[to] = src[from];
        }
}

namespace {
// Trans type -> which source loop index (0 = i, 1 = j, 2 = k, 3 = l) feeds destination dimension d0..d3.
// Types >= 24 behave like type 0.
const uint8_t kTransPerm[24][4] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
    {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
    {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0},
    {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};
}  // namespace

// @0x439e30 (MFU11)
void MFU::Trans()
{
  const uint32_t src_raw = src_addr_;
  const uint32_t dst_raw = dst_addr_;
  const uint32_t dst_off = dst_raw & 0xFFFFFFF;
  const uint8_t * src = GlbPtr(src_raw);
  uint8_t * dst = GlbPtr(dst_raw);

  uint32_t sc, sh, sw, dc, dh, dw;
  GetCHW(trans_shape_src_, sc, sh, sw);
  GetCHW(trans_shape_dst_, dc, dh, dw);

  // Alignment checks.  Types 0, 2, 6, 8, 12, 14 keep the innermost dimension in place and need no stride checks.
  const uint8_t type = trans_type_;
  const bool keeps_inner = ((uint8_t)(type - 6) & 0xFD) == 0 || (type & 0xFD) == 0 || (type & 0xFD) == 12;
  uint32_t stride_unit = 0;
  if (trans_elem16_) {
    if (src_raw & 1) {
      std::printf("when data type is fp16 or int16, transpose addr_s must be align with 2 bytes, addr_src = %d! \n",
                  static_cast<int>(src_raw & 0xFFFFFFF));
      std::exit(1);
    }
    if (dst_raw & 1) {
      std::printf("when data type is fp16 or int16, transpose addr_d must be align with 2 bytes, addr_dest = %d! \n",
                  static_cast<int>(dst_off));
      std::exit(1);
    }
    if (!keeps_inner) stride_unit = 16;
  } else if (!keeps_inner) {
    stride_unit = 32;
  }
  if (stride_unit) {
    if (sw % stride_unit) {
      std::printf("Transpose Src Stride_W must be align with 32 bytes, stride_w = %d! \n", static_cast<int>(sw));
      std::exit(1);
    }
    if (dw % stride_unit) {
      std::printf("Transpose Dst Stride_W must be align with 32 bytes, stride_w = %d! \n", static_cast<int>(dw));
      std::exit(1);
    }
    if (dst_raw & 0xF) {
      std::printf("when transtype is ***N/***C/***H, transpose addr_d must be align with 16 bytes, addr_dest = %d! \n",
                  static_cast<int>(dst_off));
      std::exit(1);
    }
  }

  const uint8_t * perm = kTransPerm[type < 24 ? type : 0];
  for (uint16_t i = 0; i < dim_[0]; ++i)
    for (uint16_t j = 0; j < dim_[1]; ++j)
      for (uint16_t k = 0; k < dim_[2]; ++k)
        for (uint16_t l = 0; l < dim_[3]; ++l) {
          const int from = static_cast<int>(l + sw * (k + sh * (j + i * sc)));
          const int idx[4] = {i, j, k, l};
          const int d0 = idx[perm[0]], d1 = idx[perm[1]], d2 = idx[perm[2]], d3 = idx[perm[3]];
          const int to = static_cast<int>(d3 + dw * (d2 + dh * (d1 + dc * d0)));
          if (trans_elem16_)
            Store16(dst + 2 * to, Load16(src + 2 * from));
          else
            dst[to] = src[from];
        }
}

// ==================================================================================================================
// ReduceFun
// ==================================================================================================================

namespace {

// Reduces the 8 freshly loaded elements buf[0..7] pairwise and folds the result into the accumulator buf[15].
// Slots 8..14 receive the intermediate tree levels.  The sub operation (3) sums the block with fp24 adds and
// subtracts the block sum from the accumulator.
void ReduceBlock(fp24 * buf, uint8_t op, REDUCE_ELEMENT::Fn fn)
{
  if (op == 3) {
    buf[8] = buf[0] + buf[1];
    buf[9] = buf[2] + buf[3];
    buf[10] = buf[4] + buf[5];
    buf[11] = buf[6] + buf[7];
    buf[12] = buf[8] + buf[9];
    buf[13] = buf[10] + buf[11];
    buf[14] = buf[12] + buf[13];
    buf[15] = buf[15] + fp24(buf[14].bits_ ^ 0x800000u);
  } else {
    fn(&buf[0], &buf[1], &buf[8], nullptr);
    fn(&buf[2], &buf[3], &buf[9], nullptr);
    fn(&buf[4], &buf[5], &buf[10], nullptr);
    fn(&buf[6], &buf[7], &buf[11], nullptr);
    fn(&buf[8], &buf[9], &buf[12], nullptr);
    fn(&buf[10], &buf[11], &buf[13], nullptr);
    fn(&buf[12], &buf[13], &buf[14], nullptr);
    fn(&buf[15], &buf[14], &buf[15], nullptr);
  }
}

}  // namespace

// @0x43aae0 (MFU12)
// Reduces the trailing `reduce_mode + 1` dimensions of a bf16 tensor (mode 3 reduces everything to one value).
// Elements are consumed in blocks of 8 through an fp24 tree (ReduceBlock); a block cut short by the end of a
// row is padded with the identity element that ReduceInit left in the unused slots.
void MFU::ReduceFun(uint64_t *)
{
  const uint8_t * src = GlbPtr(src_addr_);
  uint8_t * dst = GlbPtr(dst_addr_);
  uint32_t sc, sh, sw, dc, dh, dw;
  GetCHW(shape_src_, sc, sh, sw);
  GetCHW(shape_dst_, dc, dh, dw);

  REDUCE_ELEMENT::Fn fn;
  switch (reduce_op_) {
    case 1: fn = REDUCE_ELEMENT::re_min; break;
    case 2: fn = REDUCE_ELEMENT::re_add; break;
    case 3: fn = REDUCE_ELEMENT::re_sub; break;
    case 4: fn = REDUCE_ELEMENT::re_mul; break;
    default: fn = REDUCE_ELEMENT::re_max; break;
  }

  fp24 buf[16] = {};       // ..v240 : 8 inputs, 7 tree levels, accumulator
  uint32_t count = 0;      // elements loaded into the current row (v13)
  const uint8_t mode = reduce_mode_;

  const fp24 init_rounded = fp24::round_to_fp24(BfToFloat(reduce_init_));
  auto init_group = [&](fp24 init) { ReduceInit(buf, init, reduce_op_); count = 0; };

  // Converts the accumulator to bf16 (optionally quantises it) and stores it at element `index` of the output.
  auto store_result = [&](int index) {
    const uint16_t bits = BfRound(buf[15].f_value());
    if (!quant_enable_) {
      Store16(dst + 2 * index, bits);
    } else {
      uint8_t q = 0;
      quant(B(bits), B(quant_scale_), quant_zero_, quant_signed_, q);
      dst[index] = q;
    }
  };

  if (mode == 3)
    init_group(init_rounded);

  for (uint16_t i = 0; i < dim_[0]; ++i) {
    if (mode == 2)
      init_group(init_rounded);
    for (uint16_t j = 0; j < dim_[1]; ++j) {
      if (mode == 1)
        init_group(init_rounded);
      for (uint16_t k = 0; k < dim_[2]; ++k) {
        if (mode == 0)
          init_group(B(reduce_init_).fp24());      // NaN -> 0x7FC000 here, round_to_fp24 elsewhere
        for (uint16_t l = 0; l < dim_[3]; ++l) {
          const int from = static_cast<int>(l + sw * (k + sh * (j + sc * i)));
          BF16::bfloat16 element;
          if (dequant_enable_) {
            const uint8_t q = src[from];
            const uint8_t zero = static_cast<uint8_t>(dequant_zero_);
            dequant(q, B(dequant_scale_), zero, dequant_signed_, element);
          } else {
            element = B(Load16(src + 2 * from));
          }
          buf[count & 7] = element.fp24();
          if ((count & 7) == 7) {
            ReduceBlock(buf, reduce_op_, fn);
            ReduceInit(buf, buf[15], reduce_op_);
          }
          ++count;
        }
        if (count & 7) {                          // partial block at the end of a row
          ReduceBlock(buf, reduce_op_, fn);
          ReduceInit(buf, buf[15], reduce_op_);
        }
        count = 0;
        if (mode == 0)
          store_result(static_cast<int>(dw * (k + dh * (j + dc * i))));
      }
      if (mode == 1)
        store_result(static_cast<int>(dh * dw * (j + dc * i)));
    }
    if (mode == 2)
      store_result(static_cast<int>(i * dc * dh * dw));
  }
  if (mode == 3)
    store_result(0);
}

// ==================================================================================================================
// Sample
// ==================================================================================================================

// @0x43be70 (MFU13)
// Grid sampling.  For every (n, c, h, w) of dim_sample[] one record of the grid at aux_addr
//     { u16 x; bf16 wx; u16 y; bf16 wy; }    (8 bytes, indexed by n, h, w - shared by all channels c)
// selects the source pixel (x, y) of plane (n, c); with interp_mode != 0 the four neighbours are blended
// bilinearly in bf16, otherwise the pixel itself is copied.  Source and destination tensors are bf16, or 8-bit
// quantised when dequant_enable / quant_enable are set.
void MFU::Sample()
{
  const MemAccessor grid(GlbPtr(aux_addr_));
  const MemAccessor src(GlbPtr(src_addr_));
  uint8_t * dst = GlbPtr(dst_addr_);

  const uint32_t src_c = static_cast<uint16_t>(shape_src_ >> 32);     // 
  const uint32_t src_h = static_cast<uint16_t>(shape_src_ >> 16);     // 
  const uint32_t src_w = static_cast<uint16_t>(shape_src_);           // 
  uint32_t dst_c, dst_h, dst_w;
  GetCHW(shape_dst_, dst_c, dst_h, dst_w);
  const int grid_w = static_cast<uint16_t>(shape_grid_);              // 
  const int grid_h = static_cast<uint16_t>(shape_grid_ >> 16);
  const int grid_plane = grid_w * grid_h;                            // 

  const uint16_t count_n = dim_sample_[0], count_c = dim_sample_[1], count_h = dim_sample_[2], count_w = dim_sample_[3];

  int grid_base = 0;                                                  // first grid cell of image n
  for (uint32_t n = 0; n < count_n; ++n, grid_base += grid_plane) {
    for (uint32_t c = 0; c < count_c; ++c) {
      int grid_row = grid_base;                                       // 
      for (uint32_t h = 0; h < count_h; ++h, grid_row += grid_w) {
        for (uint32_t w = 0; w < count_w; ++w) {
          const int rec = 8 * (grid_row + static_cast<int>(w));
          const uint16_t x = grid.MemAt<uint16_t>(rec);
          const uint16_t wx = grid.MemAt<BF16::bfloat16>(rec + 2).bits_;
          const uint16_t y = grid.MemAt<uint16_t>(rec + 4);
          const uint16_t wy = grid.MemAt<BF16::bfloat16>(rec + 6).bits_;

          const int a00 = static_cast<int>(GetAddress(n, c, x, y, src_c, src_h, src_w, 1));
          const int a10 = static_cast<int>(GetAddress(n, c, x + 1, y, src_c, src_h, src_w, 1));
          const int a01 = static_cast<int>(GetAddress(n, c, x, y + 1, src_c, src_h, src_w, 1));
          const int a11 = static_cast<int>(GetAddress(n, c, x + 1, y + 1, src_c, src_h, src_w, 1));

          uint16_t p00, p10, p01, p11;
          if (dequant_enable_) {
            const uint8_t zero = static_cast<uint8_t>(dequant_zero_);
            BF16::bfloat16 v;
            dequant(src.MemAt<uint8_t>(a00), B(dequant_scale_), zero, dequant_signed_, v); p00 = v.bits_;
            dequant(src.MemAt<uint8_t>(a10), B(dequant_scale_), zero, dequant_signed_, v); p10 = v.bits_;
            dequant(src.MemAt<uint8_t>(a01), B(dequant_scale_), zero, dequant_signed_, v); p01 = v.bits_;
            dequant(src.MemAt<uint8_t>(a11), B(dequant_scale_), zero, dequant_signed_, v); p11 = v.bits_;
          } else {
            p00 = src.MemAt<BF16::bfloat16>(2 * a00).bits_;
            p10 = src.MemAt<BF16::bfloat16>(2 * a10).bits_;
            p01 = src.MemAt<BF16::bfloat16>(2 * a01).bits_;
            p11 = src.MemAt<BF16::bfloat16>(2 * a11).bits_;
          }

          uint16_t result;
          if (interp_mode_) {
            // bilinear weights: w11 = wx*wy, w01 = wy - w11, w10 = wx - w11, w00 = w11 + ((1 - wx) - wy)
            const uint16_t w11 = BfMul(wx, wy);
            const uint16_t t11 = BfMul(w11, p11);
            const uint16_t t01 = BfMul(BfAdd(wy, BfNeg(w11)), p01);
            const uint16_t t10 = BfMul(BfAdd(wx, BfNeg(w11)), p10);
            const uint16_t one_minus_wx = BfAdd(0x3F80, BfNeg(wx));
            const uint16_t w00 = BfAdd(w11, BfAdd(one_minus_wx, BfNeg(wy)));
            const uint16_t t00 = BfMul(w00, p00);
            result = BfAdd(t11, BfAdd(t01, BfAdd(t00, t10)));
          } else {
            result = p00;
          }

          const int out = static_cast<int>(GetAddress(n, c, h, w, dst_c, dst_h, dst_w, 1));
          if (quant_enable_) {
            uint8_t q = 0;
            quant(B(result), B(quant_scale_), quant_zero_, quant_signed_, q);
            dst[out] = q;
          } else {
            Store16(dst + 2 * out, result);
          }
        }
      }
    }
  }
}
