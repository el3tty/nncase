// PDP1 (MFU pooling / reduction engine), lifted from IDA/Hex-Rays output (nncase K230 C-model).
// Source numbering follows work/clean/PDP1<N>.cpp ("Source N").
#include "engines/pdp1.h"
#include "globals.h"
#include "math/numeric_types.h"
#include "engines/memaccessor.h"
#include "engines/mfu.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace {

// Prints a configuration error and terminates, as the original does for every invalid setting.
[[noreturn]] void PdpFail(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  std::vprintf(fmt, args);
  va_end(args);
  std::exit(1);
}

[[noreturn]] void PdpFailMsg(const char* msg)
{
  std::puts(msg);
  std::exit(1);
}

// Byte-wide data types shared by the source and destination configuration: 1 = uint8, 2 = int8.
bool IsByteType(uint8_t dtype) { return static_cast<uint8_t>(dtype - 1) < 2u; }

}  // namespace

// PDP1 singleton constructor (inlined into PDP1::GetPDP1 in the dump, "lazy-init elided"):
// 16 empty window slots, everything else zero.
PDP1::PDP1() : reserved_8_{}, windows_(16), reserved_160_(0), cfg_{} {}

// PDP1::GetPDP1() is not in the dump; a function-local static is equivalent to the guarded static.
PDP1* PDP1::GetPDP1()
{
  static PDP1 pdp1;
  return &pdp1;
}

// PDP11.cpp  @0x426b20 (Source 1)
// Destroys the window vector and every window's element vector.
PDP1::~PDP1() = default;

// PDP12.cpp  @0x440470 (Source 2)
void PDP1::PdpDM(MemAccessor src, int dim1, int dim2, int out_row, int out_col)
{
  PdpWindow& window = windows_[0];
  const uint16_t shape_h = cfg_.shape_h_;
  const uint16_t shape_w = cfg_.shape_w_;
  // Top-left corner of the window in the source plane (may be negative inside the padding).
  const uint32_t row0 = static_cast<uint32_t>(out_row * cfg_.stride_h_ - cfg_.pad_top_);
  const int col0 = out_col * cfg_.stride_w_ - cfg_.pad_left_;

  window.data_.clear();
  window.acc_ = FP24::fp24(0);
  window.count_ = 0;

  if (cfg_.window_h_ == 0 || cfg_.window_w_ == 0)
    return;

  // Element offset of row `row0` of plane (dim1 * src_dim2 + dim2).
  uint32_t row_off = static_cast<uint32_t>(cfg_.src_pitch_) *
                     (row0 + static_cast<uint32_t>(cfg_.src_plane_rows_) *
                                 (static_cast<uint32_t>(dim1) * cfg_.src_dim2_ + static_cast<uint32_t>(dim2)));
  for (int r = 0; r < cfg_.window_h_; ++r) {
    const bool row_in_range = shape_h > row0 + static_cast<uint32_t>(r);  // unsigned: negative rows are out of range
    for (int c = 0; c < cfg_.window_w_; ++c) {
      FP16::fp16 value(0);
      if (shape_w > static_cast<uint32_t>(c + col0) && row_in_range) {
        MFU::dequant_new_MemAt(col0 + static_cast<int>(row_off) + c, src, FP16::fp16(cfg_.dequant_scale_),
                               static_cast<short>(cfg_.dequant_zero_), cfg_.src_dtype_, cfg_.dequant_flag_, value);
      } else {
        value = FP16::fp16(cfg_.pad_value_);  // padding
      }
      window.data_.push_back(value);
    }
    row_off += cfg_.src_pitch_;
  }
}

// PDP13.cpp  @0x4406b0 (Source 3)
void PDP1::PdpRedCompute()
{
  const Pdp1Config& c = cfg_;
  // The decompilation also builds and destroys a local array of 126 {vector, fp24, count} slots
  // that is never used; it is dropped.
  MemAccessor src(_G.GLB[c.src_addr_ >> 28] + (c.src_addr_ & 0xFFFFFFF));
  MemAccessor dst(_G.GLB[c.dst_addr_ >> 28] + (c.dst_addr_ & 0xFFFFFFF));

  // --- configuration checks (each failure prints a message and exits) -------------------------
  // Maximum window_w for 4 < window_h <= 8 (small_h) and window_h > 8 (large_h): halved for 16-bit sources.
  unsigned max_w_small_h = 64;
  unsigned max_w_large_h = 32;
  if (c.src_dtype_ == 0 || c.src_dtype_ == 3) {  // fp16 / int16
    if (c.src_addr_ & 1)
      PdpFail("when data type is fp16 or int16, pdp addr_s must be align with 2 bytes, addr_src = %d! \n",
              c.src_addr_ & 0xFFFFFFF);
    max_w_small_h = 32;
    max_w_large_h = 16;
  }
  if ((c.dst_dtype_ == 0 || c.dst_dtype_ == 3) && (c.dst_addr_ & 1))
    PdpFail("when data type is fp16 or int16, pdp addr_d must be align with 2 bytes, addr_dest = %d! \n",
            c.dst_addr_ & 0xFFFFFFF);
  if (c.src_pitch_ % (IsByteType(c.src_dtype_) ? 32 : 16))
    PdpFail("PDP Src Stride_W must be align with 32 bytes, stridew = %d! \n", c.src_pitch_);
  if (!c.enable_bw_ && c.dst_pitch_ % (IsByteType(c.dst_dtype_) ? 32 : 16))
    PdpFail("PDP Dst Stride_W must be align with 32 bytes, stridew = %d! \n", c.dst_pitch_);

  for (PdpWindow& w : windows_) {  // reset all 16 window slots
    w.data_.clear();
    w.acc_ = FP24::fp24(0);
    w.count_ = 0;
  }

  if (c.window_h_ > 16)
    PdpFailMsg("PDP WindowH cann't be bigger than 16! ");
  if (c.window_w_ > 64)
    PdpFailMsg("PDP WindowW cann't be bigger than 64! ");
  if (c.window_w_ * c.window_h_ > 256)
    PdpFailMsg("PDP Window Size cann't be bigger than 256! ");
  if (c.window_h_ > 4) {
    const unsigned max_w = (c.window_h_ <= 8) ? max_w_small_h : max_w_large_h;
    if (c.window_w_ > max_w)
      PdpFail("Pdp WindowW bytes size must be <= (32/2^)/2*32, WindowW = %d, max WindowW = %d! \n", c.window_w_, max_w);
  }
  if (c.window_h_ <= c.pad_top_ || c.window_h_ <= c.pad_bottom_)
    PdpFail("Pad top and bottom must be smaller than window_h, WindowH = %d, PaddingTop = %d, PaddingBottom = %d! \n",
            c.window_h_, c.pad_top_, c.pad_bottom_);
  if (c.window_w_ <= c.pad_left_ || c.window_w_ <= c.pad_right_)
    PdpFail("Pad left and right must be smaller than window_w, WindowW = %d, PaddingLeft = %d, PaddingRight = %d! \n",
            c.window_w_, c.pad_left_, c.pad_right_);

  const int need_shape_h = c.stride_h_ * (c.out_h_ - 1) + c.window_h_ - c.pad_top_ - c.pad_bottom_;
  const int16_t need_shape_w = static_cast<int16_t>(c.stride_w_ * (c.out_w_ - 1) + c.window_w_ - c.pad_left_ - c.pad_right_);
  if (c.shape_h_ != need_shape_h)
    PdpFail("Pdp input shape_h size doesn't match output shape_h needed size, shape_h = %d, need shape_h = %d! \n",
            c.shape_h_, need_shape_h);
  if (c.shape_w_ != need_shape_w)
    PdpFail("Pdp input shape_w size doesn't match output shape_w needed size, shape_w = %d, need shape_w = %d! \n",
            c.shape_w_, need_shape_w);
  if (c.enable_bw_ == 1 && (c.dst_plane_rows_ != 1 || c.dst_pitch_ != 1 || c.out_w_ != 1 || c.out_h_ != 1))
    PdpFail("when enable_bw=1, h and w and stride must be 1, h = %d, w = %d, sh = %d, sw = %d! \n", c.out_h_, c.out_w_,
            c.dst_plane_rows_, c.dst_pitch_);
  if (c.enable_h2c_ == 1 && c.pad_top_)
    PdpFailMsg("when enable_h2c=1, PaddingTop and PaddingBottom must be 0! ");

  // --- reduction -------------------------------------------------------------------------------
  const int dst_row_step = c.dst_pitch_;
  const int dst_plane_step = c.dst_pitch_ * c.dst_plane_rows_;
  const int dst_dim1_step = dst_plane_step * c.dst_dim2_;
  int dst_dim1_base = 0;
  for (int dim1 = 0; dim1 < c.dim1_count_; ++dim1, dst_dim1_base += dst_dim1_step) {
    int dst_plane_base = dst_dim1_base;
    for (int dim2 = 0; dim2 < c.dim2_count_; ++dim2, dst_plane_base += dst_plane_step) {
      int dst_row_base = dst_plane_base;
      for (int out_row = 0; out_row < c.out_h_; ++out_row, dst_row_base += dst_row_step) {
        for (int out_col = 0; out_col < c.out_w_; ++out_col) {
          FP16::fp16 result(0);  // stays 0 for pool_mode > 3
          PdpDM(src, dim1, dim2, out_row, out_col);

          PdpWindow& window = windows_[0];
          if (c.pool_mode_ == 0 || c.pool_mode_ == 1) {
            // Max (mode 1) / min (mode 0) over the non-NaN elements, starting from -/+65504.
            // verified against asm @0x440c59 (mode 0, vcomiss + jbe: update iff best > x) and @0x441504 (mode 1,
            // update iff x > best): strict compares on the fp16 values widened to float.
            FP16::fp16 best(c.pool_mode_ == 1 ? 0xFBFF : 0x7BFF);
            for (const FP16::fp16& x : window.data_) {
              const float xf = x.f_value();
              if (std::isnan(xf))
                continue;
              if (c.pool_mode_ == 1 ? xf > best.f_value() : xf < best.f_value())
                best = x;
            }
            window.acc_ = FP24::fp24::round_to_fp24(best.f_value());
          } else if (c.pool_mode_ == 2 || c.pool_mode_ == 3) {
            // Sum of the non-NaN elements in fp24.
            window.acc_ = FP24::fp24(0);
            window.count_ = 0;
            for (const FP16::fp16& x : window.data_) {
              const float xf = x.f_value();
              if (std::isnan(xf))
                continue;
              window.acc_ = window.acc_ + FP24::fp24::round_to_fp24(xf);
              ++window.count_;
            }
          }

          if (c.pool_mode_ <= 3) {
            result = FP16::fp16::round_to_fp16(window.acc_.f_value());
            if (c.pool_mode_ == 2) {
              // verified against asm @0x44121b: (float)(1.0 / (double)count) (vdivsd + vcvtsd2ss), rounded to fp16 and multiplied.
              result = result * FP16::fp16::round_to_fp16(static_cast<float>(1.0 / static_cast<double>(window.count_)));
            } else if (c.pool_mode_ == 3) {
              result = result * FP16::fp16(c.avg_scale_);
            }
          }

          MFU::quant_new_MemSt(dst_row_base + out_col, dst, FP16::fp16(c.quant_scale_), FP16::fp16(c.quant_zero_),
                               c.dst_dtype_, c.quant_flag_, result);
        }
      }
    }
  }
}
