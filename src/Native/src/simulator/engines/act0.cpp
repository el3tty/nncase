// Lifted from IDA/Hex-Rays output (Act01..Act06.cpp). The AVX instructions that the auto-converted
// results/ copy had stripped (vmulss/vaddss/vcomiss/vroundss/...) were recovered from sources/Act0*.cpp.
//
// ACT0 post-processes the 32-bit PSUM of the PU array. Per channel `ch` it has seven fp16
// parameters p[0..6] (a Matrix4<fp16> row) and per element it does
//     x = fp16(psum * 2^-shift)                       (PsumToFp16, truncating)
//     y = x < p[6] ? p[0] * x + p[2] : p[1] * x + p[3] (LineFit, fp16 mul / add)
//     y = y <= p[4] ? p[4] : (y >= p[5] ? p[5] : y)     (clamp to [p4, p5])
// and finally stores y as fp16, or rounds it to an integer and saturates it to uint8 / int8 / int16.
// The fp16 multiply/add/convert sequences that Hex-Rays showed inline are FP16::operator* / operator+ /
// f_value() / round_to_fp16() (see fp16.cpp). The conditions of the float compares were lost by the
// decompiler; they were re-derived from the vcomiss / jb / jae / cmov sequences of the machine code.
#include "engines/act0.h"
#include "engines/dm.h"
#include "globals.h"
#include "engines/checkpoint.h"
#include "math/numeric_types.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// TODO(globals): not declared in globals.h (defined elsewhere in the original binary).
extern std::string debug_file;   // dump file path (COW std::string in the original)

using FP16::fp16;

namespace {

// ---- raw views of helper types that are not lifted yet (TODO(layout)) -------------------------------

// L1Helper: (channel, row, col) view into a buffer; strides in bytes except row/col which count elements.
struct L1View {
    uint8_t* data_;        // +0
    int32_t  max_channels_; // +8   channel capacity of the buffer (24 for IF = PE rows, 32 for PSUM = PE lanes)
    int32_t  chan_stride_; // +12  bytes per channel (4096 for PSUM buffers)
    int32_t  base_;        // +16  byte offset of element (0,0,0)
    int32_t  row_stride_;  // +20  per row index (counted in elements of the accessed type)
    int32_t  col_stride_;  // +24  per column index (same unit)
    int32_t  channels_;    // +28
    int32_t  height_;      // +32
    int32_t  width_;       // +36
};

// Tensor4DHelper as built from a DmStoreOf descriptor (see Act0::Compute).
struct Tensor4DView {
    uint8_t* data_;        // +0   DmStoreOf::dst_
    uint32_t full_[4];     // +8   DmStoreOf::full_shape (not read by any Act0 function)
    uint32_t tile_[4];     // +24  DmStoreOf::tile_shape_: [0] = channels, [1] = rows (H), [2] = columns (W)
};

// Matrix4<fp16> (dims d0 x d1 x d2 x d3, row-major); Act0 only uses it as rows (d2) of 7 columns (d3).
struct Fp16Matrix {
    uint8_t owns_;         // +0
    uint16_t* data_;       // +8
    int32_t d0_, d1_, d2_, d3_;  // +16..+28
};

// Matrix4::operator()(0, 0, row, col): reports an out-of-range access but still performs it.
fp16 MatrixAt(const Fp16Matrix& m, int row, int col)
{
  if (m.d0_ <= 0 || m.d1_ <= 0 || row >= m.d2_ || col >= m.d3_)
    std::cout << "[Error: Matrix Exceed]" << std::endl;
  return fp16(m.data_[m.d3_ * row + col]);
}

// ---- line-fit core ---------------------------------------------------------------------------------

// The seven fp16 parameters of one channel. Names are inferred from how Activate uses them.
struct ChannelParams {
    fp16 slope_lo_;    // p[0]  multiplier when x < knee
    fp16 slope_hi_;    // p[1]  multiplier otherwise
    fp16 offset_lo_;   // p[2]
    fp16 offset_hi_;   // p[3]
    fp16 lo_;          // p[4]  lower clamp of y
    fp16 hi_;          // p[5]  upper clamp of y
    fp16 knee_;        // p[6]  break point on x
};

ChannelParams ReadParams(const Fp16Matrix& m, int ch)
{
  ChannelParams p;
  p.slope_lo_ = MatrixAt(m, ch, 0);
  p.slope_hi_ = MatrixAt(m, ch, 1);
  p.offset_lo_ = MatrixAt(m, ch, 2);
  p.offset_hi_ = MatrixAt(m, ch, 3);
  p.lo_ = MatrixAt(m, ch, 4);
  p.hi_ = MatrixAt(m, ch, 5);
  p.knee_ = MatrixAt(m, ch, 6);
  return p;
}

// int32 PSUM -> fp16 with the scale 2^-shift. The mantissa is truncated, magnitudes beyond the fp16 range
// saturate to +/-65504 and tiny ones become subnormal / zero.
fp16 PsumToFp16(int32_t psum, uint32_t shift)
{
  if (psum == 0)
    return fp16(0);
  const uint16_t sign = psum < 0 ? 0x8000 : 0;
  const uint32_t mag = psum < 0 ? 0u - static_cast<uint32_t>(psum) : static_cast<uint32_t>(psum);
  int msb = 31;                                                // index of the highest set bit (mag != 0)
  while (!(mag >> msb))
    --msb;
  const int exponent = msb + 15 - static_cast<int>(shift);     // biased fp16 exponent
  uint32_t mant = msb > 10 ? mag >> (msb - 10) : mag << (10 - msb);   // 11 bits including the hidden one
  uint16_t bits;
  if (exponent > 30) {
    bits = 0x7BFF;                                             // saturate to the largest finite value
  } else if (exponent <= 0) {
    mant >>= (1 - exponent) & 31;                              // subnormal
    bits = static_cast<uint16_t>(mant & 0x3FF);
  } else {
    bits = static_cast<uint16_t>((exponent << 10) | (mant & 0x3FF));
  }
  return fp16(static_cast<uint16_t>(sign | bits));
}

// y = (x < knee) ? slope_lo * x + offset_lo : slope_hi * x + offset_hi, clamped to [lo, hi].
// verified against asm @0x4470c1 / 0x4474a0 / 0x447aa2 (and the Tensor4D overload @0x449b70 ...):
//   `vcomiss x, knee; jbe` -> the first segment is taken when knee > x (ordered), NaN takes the second;
//   `vcomiss y, p4; jb` not taken (p4 >= y, ordered) -> y = p4;
//   `vcomiss p5, y; jae` (y >= p5, ordered) -> y = p5; otherwise y is kept (also when NaN is involved).
// p[4] / p[5] are therefore the lower / upper clamp, not "min(y, p[4])" with an unused p[5].
fp16 LineFit(const ChannelParams& p, fp16 x)
{
  fp16 y;
  if (p.knee_.f_value() > x.f_value())
    y = p.slope_lo_ * x + p.offset_lo_;
  else
    y = p.slope_hi_ * x + p.offset_hi_;
  if (p.lo_.f_value() >= y.f_value())
    return p.lo_;
  if (y.f_value() >= p.hi_.f_value())
    return p.hi_;
  return y;
}

// vroundss(.., 0xC) with the default rounding mode, then back to fp16 (exact for the reachable values).
fp16 RoundToIntegral(fp16 y)
{
  return fp16::round_to_fp16(std::nearbyint(y.f_value()));
}

// Integral fp16 -> uint8. Magnitudes >= 2^15 / inf / NaN saturate (negative -> 0, positive -> 255).
uint8_t IntegralToUInt8(fp16 q)
{
  const int sign = q.bits_ >> 15;
  const int exponent = (q.bits_ >> 10) & 0x1F;
  const int mantissa = q.bits_ & 0x3FF;
  if (exponent > 29)
    return sign ? 0 : 255;
  if (exponent <= 13)
    return 0;
  const int value = (mantissa + 1024) * (1 - 2 * sign);   // signed 11-bit mantissa
  int result;
  if (((exponent + 7) & 0x1F) > 4) {                       // exponent 14..24: shift right, round half to even
    const int shift = 25 - exponent;
    result = static_cast<int16_t>((value + (1 << (shift - 1)) - 1 + ((value >> shift) & 1)) >> shift);
  } else {                                                 // exponent 25..29: shift left
    result = static_cast<int16_t>(value << (exponent - 25));
  }
  return static_cast<uint8_t>(std::clamp(result, 0, 255));
}

// Integral fp16 -> int, truncated; inf / NaN become +/-65504 first (asm @0x447cb7 / 0x447cc5: sign from bit 15); then
// clamped to [lo, hi] with vpminsd / vpmaxsd.
int32_t IntegralToSaturated(fp16 q, int32_t lo, int32_t hi)
{
  float f = q.f_value();
  if (std::isnan(f) || std::isinf(f))
    f = (q.bits_ & 0x8000) ? -65504.0f : 65504.0f;
  const int32_t v = static_cast<int32_t>(f);
  return std::max(std::min(v, hi), lo);
}

// verified against asm @0x447d13 / 0x44aabf / 0x44a8c8 and ELF .rodata: .rodata 0x522060 = 127, 5220A0 = -127,
// 5220B0 = 32767, 5220C0 = -32767 (symmetric limits).
constexpr int32_t kInt8Max = 127, kInt8Min = -127;
constexpr int32_t kInt16Max = 32767, kInt16Min = -32767;

[[noreturn]] void BadDestType()
{
  std::cerr << "Error act0 dest_datatype" << std::endl;
  std::exit(1);
}

// Element index (in 4-byte words) of PSUM element (ch, row, col) inside an L1 view.
int PsumWordIndex(const L1View& in, int ch, int row, int col)
{
  const int64_t sum = static_cast<int64_t>(in.row_stride_ * row)
                    + static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(in.base_)) >> 2)
                    + static_cast<int64_t>(in.col_stride_ * col)
                    + static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(ch * in.chan_stride_)) >> 2);
  return static_cast<int>(sum);
}

// Element index (in 2-byte words) of an fp16 element (ch, row, col) inside an L1 output view.
int HalfWordIndex(const L1View& out, int ch, int row, int col)
{
  const int64_t sum = static_cast<int64_t>(out.row_stride_ * row)
                    + static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(out.base_)) >> 1)
                    + static_cast<int64_t>(out.col_stride_ * col)
                    + static_cast<int64_t>(static_cast<uint64_t>(static_cast<int64_t>(ch * out.chan_stride_)) >> 1);
  return static_cast<int>(sum);
}

}  // namespace

// Act01.cpp  @0x446ac0 (Source 1)
// Snapshots the Act0 configuration (see Act0Src1ConfInstruction) into a new Act0Compute descriptor.
std::shared_ptr<Act0Compute> Act0::GetAct0Compute() const
{
  auto compute = std::make_shared<Act0Compute>();
  compute->batch_ = batch_;              // +0..+15: 16-byte copy of Act0 + 0x40000
  compute->channels_ = channels_;
  compute->height_ = height_;
  compute->width_ = width_;
  compute->shift_ = shift_;              // +16: 8-byte copy of Act0 + 0x40010 (shift, engine)
  compute->engine_ = engine_;
  compute->out_base_ = out_base_;        // +24 (overwritten by Act0ComputeInstruction)
  compute->out_route_ = out_route_;      // +28 (overwritten by Act0ComputeInstruction)
  compute->out_type_ = out_type_;        // +32 (overwritten by Act0ComputeInstruction)
  compute->per_channel_ = per_channel_;  // +36 (overwritten by Act0ComputeInstruction)
  // The second PSUM buffer belongs to PDP0 (engine != 0).
  compute->psum_ = const_cast<uint32_t *>(engine_ ? psum1_ : psum0_);   // +40: this + 0x20000 or this
  return compute;
}

// Act02.cpp  @0x446b80 (Source 2)
// Line-fit activation of the PSUM `psum` (channels x height x width) into the L1 view `out`.
void Act0::Activate(Matrix4<FP16::fp16> & params_, L1Helper & psum_, L1Helper & out_, uint32_t shift, ActOutputType type)
{
  const Fp16Matrix& params = reinterpret_cast<const Fp16Matrix&>(params_);  // TODO(layout)
  const L1View& in = reinterpret_cast<const L1View&>(psum_);                // TODO(layout)
  L1View& out = reinterpret_cast<L1View&>(out_);                            // TODO(layout)

  for (int row = 0; row < in.height_; ++row) {
    for (int col = 0; col < in.width_; ++col) {
      for (int ch = 0; ch < in.channels_; ++ch) {
        const ChannelParams p = ReadParams(params, ch);
        const int32_t psum = reinterpret_cast<const int32_t*>(in.data_)[PsumWordIndex(in, ch, row, col)];
        const fp16 y = LineFit(p, PsumToFp16(psum, shift));

        if (type == kActFp16) {
          reinterpret_cast<uint16_t*>(out.data_)[HalfWordIndex(out, ch, row, col)] = y.bits_;
          continue;
        }
        const fp16 q = RoundToIntegral(y);
        const int byte_offset = ch * out.chan_stride_ + out.col_stride_ * col + out.base_ + out.row_stride_ * row;
        if (type == kActInt8)
          out.data_[byte_offset] = static_cast<uint8_t>(IntegralToSaturated(q, kInt8Min, kInt8Max));
        else if (type == kActUInt8)
          out.data_[byte_offset] = IntegralToUInt8(q);
        else
          BadDestType();   // this overload has no int16 output
      }
    }
  }
}

// Act03.cpp  @0x449460 (Source 3)
// Gathers, for every (row, col) of `shape`, the channel values of the DM store tensor `out`.
// verified against asm @0x449460: the machine code really only fills a freshly allocated 0x80-byte (32 x int) buffer and
// frees it again - there is no print call.  The original stores lanes[ch] without a bound check (heap overflow for more
// than 32 channels); the bound check below is kept for safety.
void Act0::PrintActDmWriteCKP(L1Helper & shape_, Tensor4DHelper & out_, ActOutputType type)
{
  const L1View& shape = reinterpret_cast<const L1View&>(shape_);          // TODO(layout)
  const Tensor4DView& out = reinterpret_cast<const Tensor4DView&>(out_);  // TODO(layout)

  for (int row = 0; row < shape.height_; ++row) {
    for (int col = 0; col < shape.width_; ++col) {
      std::vector<int> lanes(32, 0);   // 0x80 bytes; the original overruns it for more than 32 channels
      for (int ch = 0; ch < shape.channels_; ++ch) {
        const int64_t index = col + static_cast<int64_t>(out.tile_[2]) * (row + static_cast<int64_t>(ch) * out.tile_[1]);
        int value;
        if (type == kActUInt8 || type == kActInt8)
          value = out.data_[index];
        else if (type == kActFp16 || type == kActInt16)
          value = reinterpret_cast<const int16_t*>(out.data_)[index];
        else
          BadDestType();
        if (ch < static_cast<int>(lanes.size()))
          lanes[ch] = value;
      }
    }
  }
}

// Act04.cpp  @0x449620 (Source 4)
// Same line-fit as the L1 overload, but the result goes to a DM store tensor (channel, row, column).
void Act0::Activate(Matrix4<FP16::fp16> & params_, L1Helper & psum_, Tensor4DHelper & out_, uint32_t shift, ActOutputType type)
{
  const Fp16Matrix& params = reinterpret_cast<const Fp16Matrix&>(params_);  // TODO(layout)
  const L1View& in = reinterpret_cast<const L1View&>(psum_);                // TODO(layout)
  Tensor4DView& out = reinterpret_cast<Tensor4DView&>(out_);                // TODO(layout)

  CheckPoint::GetCheckPoint();   // the original ensures the singleton exists; the result is unused

  for (int row = 0; row < in.height_; ++row) {
    for (int col = 0; col < in.width_; ++col) {
      for (int ch = 0; ch < in.channels_; ++ch) {
        const ChannelParams p = ReadParams(params, ch);
        const int32_t psum = reinterpret_cast<const int32_t*>(in.data_)[PsumWordIndex(in, ch, row, col)];
        const fp16 y = LineFit(p, PsumToFp16(psum, shift));

        // element index inside the (channel, H, W) tensor
        const int64_t index = col + static_cast<int64_t>(out.tile_[2]) * (row + static_cast<int64_t>(ch) * out.tile_[1]);
        if (type == kActFp16) {
          reinterpret_cast<uint16_t*>(out.data_)[index] = y.bits_;
          continue;
        }
        const fp16 q = RoundToIntegral(y);
        if (type == kActInt8)
          out.data_[index] = static_cast<uint8_t>(IntegralToSaturated(q, kInt8Min, kInt8Max));
        else if (type == kActInt16)
          reinterpret_cast<uint16_t*>(out.data_)[index] = static_cast<uint16_t>(IntegralToSaturated(q, kInt16Min, kInt16Max));
        else if (type == kActUInt8)
          out.data_[index] = IntegralToUInt8(q);
        else
          BadDestType();
      }
    }
  }

  if (_G.debug_flag && !_G.debug_tcu_sel) {
    // Dump the first byte of every output element, one hex byte per line.
    std::ofstream dump(debug_file.c_str(), std::ios::out);
    for (int ch = 0; ch < in.channels_; ++ch) {
      for (int row = 0; row < in.height_; ++row) {
        for (int col = 0; col < in.width_; ++col) {
          const int64_t index = col + static_cast<int64_t>(out.tile_[2]) * (row + static_cast<int64_t>(ch) * out.tile_[1]);
          dump << std::hex << std::setw(2) << std::setfill('0') << static_cast<uint64_t>(out.data_[index]) << std::endl;
        }
      }
    }
    _G.debug_flag = 0;
    dump.close();
  }
}

// Act05.cpp  @0x44c650 (Source 5)
// Runs the activation for one queued Act0Compute: builds the parameter matrix from the DmLoadAct0
// table, then activates the PSUM into the DM store tensor and / or into PSUM_L1 depending on out_route.
void Act0::Compute(std::shared_ptr<Act0Compute> compute, std::shared_ptr<DmLoadAct0> load, std::shared_ptr<DmStoreOf> store)
{
  const Act0Compute& c = *compute;
  const int channels = static_cast<int>(c.channels_);
  const ActOutputType type = static_cast<ActOutputType>(c.out_type_);

  // Matrix4<fp16> (1 x 1 x channels x 7). The per-channel flag selects one table row per channel, otherwise
  // row 0 is replicated. (The original also zero-initialises an unused array of 3072 empty vectors.)
  std::vector<uint16_t> table(static_cast<size_t>(7) * channels, 0);
  const int16_t* src = load->params_;
  for (int i = 0; i < channels; ++i) {
    const int src_row = c.per_channel_ ? 7 * i : 0;
    for (int k = 0; k < 7; ++k)
      table[7 * i + k] = static_cast<uint16_t>(src[src_row + k]);
  }
  Fp16Matrix params;
  params.owns_ = channels != 0;
  params.data_ = table.data();
  params.d0_ = 1;
  params.d1_ = 1;
  params.d2_ = channels;
  params.d3_ = 7;

  // L1Helper over the PSUM buffer to read.
  L1View in;
  in.data_ = reinterpret_cast<uint8_t*>(c.psum_);
  in.max_channels_ = 32;
  in.chan_stride_ = 4096;
  in.base_ = 0;
  in.row_stride_ = static_cast<int32_t>(c.width_);
  in.col_stride_ = 1;
  in.channels_ = channels;
  in.height_ = static_cast<int32_t>(c.height_);
  in.width_ = static_cast<int32_t>(c.width_);

  // L1Helper over PSUM_L1 for the L1 output route.
  L1View psum_l1;
  psum_l1.data_ = reinterpret_cast<uint8_t*>(_G.PSUM_L1);
  psum_l1.max_channels_ = 32;
  psum_l1.chan_stride_ = 4096;
  psum_l1.base_ = static_cast<int32_t>(c.out_base_);
  psum_l1.row_stride_ = static_cast<int32_t>(c.width_);
  psum_l1.col_stride_ = 1;
  psum_l1.channels_ = channels;
  psum_l1.height_ = static_cast<int32_t>(c.height_);
  psum_l1.width_ = static_cast<int32_t>(c.width_);

  auto& matrix = reinterpret_cast<Matrix4<FP16::fp16>&>(params);   // TODO(layout)
  auto& in_helper = reinterpret_cast<L1Helper&>(in);               // TODO(layout)

  if (c.out_route_ != 0) {
    // DM store tensor built from the DmStoreOf descriptor.
    Tensor4DView out;
    out.data_ = store->dst_;
    std::copy(store->full_shape_, store->full_shape_ + 4, out.full_);
    std::copy(store->tile_shape_, store->tile_shape_ + 4, out.tile_);
    Activate(matrix, in_helper, reinterpret_cast<Tensor4DHelper&>(out), c.shift_, type);   // TODO(layout)
    if (c.out_route_ != 1)
      Activate(matrix, in_helper, reinterpret_cast<L1Helper&>(psum_l1), c.shift_, type);   // TODO(layout)
  } else {
    Activate(matrix, in_helper, reinterpret_cast<L1Helper&>(psum_l1), c.shift_, type);     // TODO(layout)
  }
}

// Act06.cpp  @0x44d030 (Source 6)
// Writes the activation result of the DM store tensor to the dm_of checkpoint, one 32-byte line per record.
// `address` is the DM byte address of the tensor.
// verified against asm @0x44d030: elem_bytes = (type > 1) + 1, i.e. uint8 / int8 are one byte and fp16 / int16 two bytes;
// the address handed to PrintStoreOfCheckPoint (stack slot 0x58(%rsp)) is the address of the FIRST byte of the pending line:
// address + channel offset at the start of a channel (not aligned down), and the byte address of the element that opened
// the line after every line advance.
void Act0::PrintDmOfCKP(L1Helper & shape_, Tensor4DHelper & out_, ActOutputType type, int address)
{
  const L1View& shape = reinterpret_cast<const L1View&>(shape_);          // TODO(layout)
  const Tensor4DView& out = reinterpret_cast<const Tensor4DView&>(out_);  // TODO(layout)
  const int elem_bytes = (static_cast<unsigned>(type) > 1) ? 2 : 1;      // fp16 / int16 are two bytes wide

  CheckPoint* ckp = CheckPoint::GetCheckPoint();

  std::vector<uint8_t> line;   // bytes of the current 32-byte line, lowest address first
  uint32_t line_address = 0;   // address of the first byte of `line` (0x58(%rsp))
  for (int ch = 0; ch < shape.channels_; ++ch) {
    const int channel_offset = elem_bytes * static_cast<int>(out.tile_[2]) * static_cast<int>(out.tile_[1]) * ch;
    line_address = static_cast<uint32_t>(address + channel_offset);
    uint32_t line_end = (line_address & 0xFFFFFFE0u) + 32;                // end of the current line
    // Prints the pending line (if any), then starts a new line whose first byte is at `next_address`.
    auto advance = [&](uint32_t next_address) {
      if (!line.empty()) {
        CheckPoint::PrintStoreOfCheckPoint(ckp->Stream(CheckPoint::kDmOf), line, line_address);   // +6144
        line.clear();
      }
      line_address = next_address;
      line_end += 32;
    };
    for (int row = 0; row < shape.height_; ++row) {
      for (int col = 0; col < shape.width_; ++col) {
        const int64_t index = static_cast<int64_t>(out.tile_[2]) * (row + static_cast<int64_t>(out.tile_[1]) * ch) + col;
        if (elem_bytes == 2) {
          const uint32_t byte_address = static_cast<uint32_t>(address + 2 * index);
          if (!(line_end > byte_address))      // element starts in a later line (unsigned compare, `jbe`)
            advance(byte_address);
          const uint16_t value = reinterpret_cast<const uint16_t*>(out.data_)[index];
          line.push_back(static_cast<uint8_t>(value));            // low byte
          if (byte_address + 1 >= line_end)                       // the low byte was the last of the line
            advance(byte_address + 1);
          line.push_back(static_cast<uint8_t>(value >> 8));       // high byte
        } else {
          const uint32_t byte_address = static_cast<uint32_t>(index + address);
          if (!(line_end > byte_address))
            advance(byte_address);
          line.push_back(out.data_[index]);
        }
      }
    }
    if (!line.empty()) {
      CheckPoint::PrintStoreOfCheckPoint(ckp->Stream(CheckPoint::kDmOf), line, line_address);
      line.clear();
    }
  }
}
