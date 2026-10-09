// Lifted from IDA/Hex-Rays output (AI2D1..AI2D23).  See ai2d.h for the object layout.
#include "engines/ai2d.h"
#include "globals.h"
#include "engines/checkpoint.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace {

// Check point stream of the AI2D block.
// vcvttss2si: truncation, 0x80000000 ("integer indefinite") for NaN / out of range.
int32_t TruncToInt32(float f)
{
  if (!(f > -2147483904.0f && f < 2147483648.0f))
    return INT32_MIN;
  return static_cast<int32_t>(f);
}

std::ofstream &Ck(CheckPoint::StreamId id)
{
  return CheckPoint::GetCheckPoint()->Stream(id);
}

// The check point dumps print "<hex, zero filled, minimum width>"; like in the binary the hex/fill
// formatting flags stay set on the stream afterwards.
void PutHex(std::ostream &os, uint64_t value, int width)
{
  os << std::hex << std::setfill('0') << std::setw(width) << static_cast<uint64_t>(value);
}

// Address of a plane inside the source/destination memory: `in_ddr` selects DDR (flat address),
// otherwise bits [31:28] of the address select a GLB bank and the low 28 bits are the offset.
uint8_t *PlaneBase(const AI2D *a, bool in_ddr, uint32_t ptr, int64_t offset)
{
  if (in_ddr)
    return a->ddr_ + offset + ptr;
  return a->glb_[ptr >> 28] + offset + (ptr & 0xFFFFFFF);
}

}  // namespace

// @0x424ef0 (Source AI2D1)
AI2D::~AI2D()
{
}

// Singleton: the object is the global register block AI2D_Ai2dInst (see ai2d.h).
AI2D *AI2D::GetAI2D()
{
  return reinterpret_cast<AI2D *>(AI2D_Ai2dInst);
}

// @0x45d0c0 (Source AI2D2)
// Latches the registers: sign-extends the 12-bit YUV->RGB coefficients and copies M0..M5 into the
// float shadow registers used by cord_calc.
void AI2D::ai_2d_para_update()
{
  std::memcpy(m_, m_raw_, sizeof(m_));   // 16 bytes +184 -> +208 (M0..M3) and 8 bytes +200 -> +224 (M4, M5)
  for (int32_t &coef : yuv2rgb_coef_) {
    if (coef > 2047)
      coef -= 4096;
  }
}

// @0x45d220 (Source AI2D3)
// Dumps the configuration into the first AI2D check point stream.
void AI2D::ai_2d_para_print()
{
  std::ofstream &os = Ck(CheckPoint::kAi2d0);
  // verified against asm @0x45d220: every value goes through std::ostream::operator<<(int) (_ZNSolsEi), i.e. it is printed
  // as a SIGNED 32-bit integer (byte fields are zero-extended first); each print re-fetches the check point stream (+0x4000).
  auto line = [&os](const char *name, int32_t value) { os << name << ' ' << value << std::endl; };
  auto indexed = [&line](const char *prefix, int i, const char *suffix, int32_t value) {
    line((std::string(prefix) + std::to_string(i) + suffix).c_str(), value);
  };

  for (int i = 0; i < 4; ++i) indexed("src_ch", i, "_ptr", src_ch_ptr_[i]);
  line("src_x", src_x_);
  line("src_y", src_y_);
  for (int i = 0; i < 4; ++i) indexed("dst_ch", i, "_ptr", dst_ch_ptr_[i]);
  line("dst_x", dst_x_);
  line("dst_y", dst_y_);
  // The affine coefficients are printed as integers (cvttss2si of the float shadow registers).
  // verified against asm @0x45d74b: vcvttss2si of the float shadow registers (+0xd0..+0xe4) into the SIGNED int overload.
  for (int i = 0; i < 6; ++i) os << 'M' << i << ' ' << TruncToInt32(m_[i]) << std::endl;
  line("interpolation", interpolation_);
  line("cord_round", cord_round_);
  line("channel", channel_);
  line("dst_channel", dst_channel_);
  for (int i = 0; i < 4; ++i) indexed("src_ch", i, "_width_layout", src_width_layout_[i]);
  for (int i = 0; i < 4; ++i) indexed("dst_ch", i, "_width_layout", dst_width_layout_[i]);
  line("src_height_shape", src_height_shape_);
  line("src_width_shape", src_width_shape_);
  line("dst_height_shape", dst_height_shape_);
  line("dst_width_shape", dst_width_shape_);
  line("src_format", src_format_);
  line("dst_format", dst_format_);
  line("csc_en", csc_en_);
  line("src_ind", src_ind_);
  line("dst_ind", dst_ind_);
  line("shift", shift_);
  line("bound_ind", bound_ind_);
  line("bound_val", bound_val_);
  line("bound_smooth", bound_smooth_);
  line("pad_l", pad_l_);
  line("pad_r", pad_r_);
  line("pad_t", pad_t_);
  line("pad_b", pad_b_);
  line("pad_mod", pad_mod_);
  for (int i = 0; i < 4; ++i) indexed("const_pad_ch", i, "", const_pad_ch_[i]);
  for (int i = 0; i < 12; ++i) indexed("yuv2rgb_coef", i, "", yuv2rgb_coef_[i]);
  line("signed", is_signed_);
  line("cmd_id", cmd_id_);
  line("intr_mask", intr_mask_);
  line("ai2d_calc_enable", calc_enable_);
}

// @0x45fb80 (Source AI2D4)
// Dumps a DDR read of `len` bytes at `addr` as 16-byte rows (most significant byte first) into
// stream 4, plus one "<addr> <row count - 1>" request line per 256-byte block into the rx address stream.
void AI2D::PrintLoadDDRCheckPoint(uint32_t addr, uint32_t len)
{
  std::ofstream &data = Ck(CheckPoint::kAi2d4);                 // CheckPoint +18432
  std::ofstream &req = Ck(CheckPoint::kAi2dRxDdrRaddr);         // CheckPoint +17920

  uint32_t cur = addr + ddr_addr_offset_;
  const uint32_t end = cur + len;
  if (end <= cur)
    return;
  for (;;) {
    const uint32_t block_end = std::min<uint32_t>((cur & ~0xFFu) + 256, end);
    uint32_t rows_end = block_end;                    // rounded up to a whole 16-byte row
    if (rows_end & 0xF)
      rows_end = (rows_end & ~0xFu) + 16;
    uint32_t row = cur;
    uint32_t rows = static_cast<uint32_t>(-1);        // number of rows - 1
    if (rows_end > row) {
      for (;;) {
        int top = ~row & 0xF;                         // byte index of the last byte of this 16-byte row
        const int step = 16 - (row & 0xF);
        do {
          PutHex(data, ddr_[top + row], 2);
        } while (top-- != 0);
        for (uint32_t pad = row & 0xF; pad != 0; --pad)   // bytes below the start of an unaligned row
          PutHex(data, 0, 2);
        data << std::endl;
        ++rows;
        row += step;
        if (rows_end <= row)
          break;
      }
    }
    PutHex(req, cur, 8);
    req << ' ';
    PutHex(req, rows, 1);
    req << std::endl;
    if (end <= row)
      break;
    cur = row;
  }
}

// @0x45ffe0 (Source AI2D5)
// Dumps a GLB read of `len` bytes at `addr` (bank in bits [31:28]): per 16-byte row one line with the
// row index in the address stream and one line with the 16 data bytes (most significant first).
void AI2D::PrintLoadGlbCheckPoint(uint32_t addr, uint32_t len)
{
  std::ofstream &req = Ck(CheckPoint::kAi2d13);                 // CheckPoint +23040
  std::ofstream &data = Ck(CheckPoint::kAi2d14);                // CheckPoint +23552

  const uint32_t bank = addr >> 28;
  uint8_t line[16] = {};
  cur_glb_base_ = glb_[bank];
  uint32_t pos = (addr & 0xFFFFFFF) + glb_addr_offset_;
  if (static_cast<int32_t>(len) <= 0)
    return;
  uint32_t done = 0;
  do {
    const uint32_t row = (pos >> 4) + 2 * glb_start_[bank];
    const uint32_t offset = pos - glb_addr_offset_;      // byte offset inside the bank
    const uint32_t lane = pos & 0xF;
    const uint32_t n = std::min<uint32_t>(16 - lane, len - done);
    if (n != 0)
      std::memcpy(line + lane, cur_glb_base_ + offset, n);
    PutHex(req, row, 5);
    req << std::endl;
    for (int i = 15; i >= 0; --i)
      PutHex(data, line[i], 2);
    data << std::endl;
    std::memset(line, 0, sizeof(line));
    done += n;
    pos += n;
  } while (done < len);
}

// @0x460490 (Source AI2D6)
// Dumps a DDR write of `len` bytes at `addr`: one line "<byte mask> <16 bytes, msb first>" per 16-byte
// row into the data stream and one "<addr> <row count - 1>" line per 256-byte block into the waddr stream.
void AI2D::PrintStoreDDRCheckPoint(uint32_t addr, uint32_t len)
{
  std::ofstream &data = Ck(CheckPoint::kAi2d11);                // CheckPoint +22016
  std::ofstream &req = Ck(CheckPoint::kAi2dTxDdrWaddr);         // CheckPoint +21504

  uint32_t cur = addr + ddr_addr_offset_;
  const uint32_t end = cur + len;
  if (end <= cur)
    return;
  for (;;) {
    uint32_t pos = cur;
    uint32_t rows = static_cast<uint32_t>(-1);        // number of rows - 1
    const uint32_t block_end = std::min<uint32_t>((cur & ~0xFFu) + 256, end);
    uint32_t lane = cur & 0xF;
    if (block_end > cur) {
      for (;;) {
        uint8_t line[16] = {};
        uint32_t mask = 0;
        for (; lane < 16 && pos < block_end; ++lane, ++pos) {
          line[lane] = ddr_[pos];
          mask |= 1u << lane;
        }
        PutHex(data, mask, 4);
        data << ' ';
        for (int i = 15; i >= 0; --i)
          PutHex(data, line[i], 2);
        data << std::endl;
        ++rows;
        lane = 0;
        if (pos >= block_end)
          break;
      }
    }
    PutHex(req, cur, 8);
    req << ' ';
    PutHex(req, rows, 1);
    req << std::endl;
    if (end <= pos)
      break;
    cur = pos;
  }
}

// @0x460c10 (Source AI2D7)
// Dumps a GLB write of `len` bytes at `addr`: per 16-byte row "<row> <byte mask> <16 bytes, msb first>".
void AI2D::PrintWriteGlbCheckPoint(uint32_t addr, uint32_t len)
{
  std::ofstream &os = Ck(CheckPoint::kAi2dGlbWrite);            // CheckPoint +24064

  const uint32_t bank = addr >> 28;
  uint8_t line[16] = {};
  cur_glb_base_ = glb_[bank];
  uint32_t pos = (addr & 0xFFFFFFF) + glb_addr_offset_;
  if (static_cast<int32_t>(len) <= 0)
    return;
  uint32_t done = 0;
  do {
    const uint32_t row = (pos >> 4) + 2 * glb_start_[bank];
    const uint32_t offset = pos - glb_addr_offset_;
    const uint32_t lane = pos & 0xF;
    const uint32_t n = std::min<uint32_t>(16 - lane, len - done);
    uint32_t mask = 0;
    if (n != 0) {
      std::memcpy(line + lane, cur_glb_base_ + offset, n);
      mask = ((1u << n) - 1) << lane;
    }
    PutHex(os, row, 5);
    os << ' ';
    PutHex(os, mask, 4);
    os << ' ';
    for (int i = 15; i >= 0; --i)
      PutHex(os, line[i], 2);
    os << std::endl;
    std::memset(line, 0, sizeof(line));
    done += n;
    pos += n;
  } while (done < len);
}

// @0x4611e0 (Source AI2D8)
// Sets the per-plane scale / row divisor tables for a pixel format (0/1 NV12/NV21, 2 I420, 4 packed
// RGB888, 5 raw16).  Entries not mentioned keep their previous value.
void AI2D::slice_factor_cnt(int format)
{
  std::memset(plane_row_div_, 1, sizeof(plane_row_div_));
  if (format == 2) {
    plane_scale_[1] = 0.5f;
    plane_scale_[2] = 0.5f;
    plane_row_div_[1] = 2;
    plane_row_div_[2] = 2;
  } else if (format < 2) {
    if (static_cast<uint32_t>(format) <= 1)
      plane_row_div_[1] = 2;
  } else if (format == 4) {
    plane_scale_[0] = 3.0f;
  } else if (format == 5) {
    plane_scale_[0] = 2.0f;
    plane_scale_[1] = 2.0f;
  }
}

// @0x461270 (Source AI2D9)
// Dumps the source read requests of a plane: first the request line "<addr> <len> <rows>", then one
// Load check point per row (DDR or GLB depending on src_ind).
// verified against asm @0x461270: the row length is (int)((float)src_width_shape * scale) (movl 0x120(%rdi), unsigned->float),
// the byte offset of src_x is (int)((float)src_x * scale) (signed), and src_y * pitch / row_div uses a SIGNED idiv.
void AI2D::print_rx_data(uint32_t addr, int pitch, float scale, uint8_t row_div)
{
  const int32_t x_bytes = TruncToInt32(static_cast<float>(src_x_) * scale);
  uint32_t row_addr = addr + static_cast<uint32_t>(static_cast<int32_t>(static_cast<uint32_t>(pitch) * src_y_) / row_div) +
                      static_cast<uint32_t>(x_bytes);
  const uint32_t row_len = static_cast<uint32_t>(TruncToInt32(static_cast<float>(src_width_shape_) * scale));

  std::ofstream &os = Ck(CheckPoint::kAi2d2);                   // CheckPoint +17408
  PutHex(os, row_addr, 8);
  os << ' ';
  PutHex(os, row_len, 4);
  os << ' ';
  PutHex(os, src_height_shape_ / row_div, 4);
  os << std::endl;

  if (row_div <= src_height_shape_) {
    const uint32_t rows = src_height_shape_ / row_div;
    for (uint32_t i = 0; i < rows; ++i) {
      if (src_ind_)
        PrintLoadDDRCheckPoint(row_addr, row_len);
      else
        PrintLoadGlbCheckPoint(row_addr, row_len);
      row_addr += pitch;
    }
  }
}

// @0x461610 (Source AI2D10)
// Dumps the destination write request of a plane ("<addr> <len> 0001") and the matching Store/Write
// check point (DDR or GLB depending on dst_ind).
// verified against asm @0x461610: the row length is (float)pad_width (+0x130, NOT dst_width_shape) * scale; dst_y * pitch / row_div
// is a signed idiv, row * pitch / row_div an unsigned div.
void AI2D::print_tx_data(uint32_t addr, int pitch, float scale, uint8_t row_div, uint32_t row)
{
  const int32_t x_bytes = TruncToInt32(static_cast<float>(dst_x_) * scale);
  const uint32_t tx_addr = static_cast<uint32_t>(static_cast<int32_t>(static_cast<uint32_t>(pitch) * dst_y_) / row_div) + addr +
                           (row * static_cast<uint32_t>(pitch)) / row_div + static_cast<uint32_t>(x_bytes);
  const uint32_t row_len = static_cast<uint32_t>(TruncToInt32(static_cast<float>(pad_width_) * scale));

  std::ofstream &os = Ck(CheckPoint::kAi2dTxReq);               // CheckPoint +20992
  PutHex(os, tx_addr, 8);
  os << ' ';
  PutHex(os, row_len, 4);
  os << ' ';
  PutHex(os, 1, 4);
  os << std::endl;

  if (dst_ind_)
    PrintStoreDDRCheckPoint(tx_addr, row_len);
  else
    PrintWriteGlbCheckPoint(tx_addr, row_len);
}

// @0x461970 (Source AI2D11)
void AI2D::print_ddr_data()
{
  slice_factor_cnt(src_format_);
}

// @0x461980 (Source AI2D12)
// Loads the source rectangle (src_x, src_y, src_width_shape x src_height_shape) from DDR/GLB into the
// four plane buffers plane_in[], de-interleaving according to src_format.
void AI2D::slice_ld()
{
  const bool in_ddr = src_ind_ != 0;
  const uint32_t rows = src_height_shape_;
  const uint32_t cols = src_width_shape_;
  const int64_t x = src_x_;
  const uint32_t y = src_y_;
  const uint32_t stride0 = src_width_layout_[0];
  const uint32_t stride1 = src_width_layout_[1];
  const uint32_t stride2 = src_width_layout_[2];
  const uint32_t stride3 = src_width_layout_[3];

  switch (src_format_) {
    case 0:     // NV12: Y plane + interleaved U,V plane at half height
    case 1: {   // NV21: same, V first
      const uint8_t *luma = PlaneBase(this, in_ddr, src_ch_ptr_[0], x + stride0 * y);
      const uint8_t *chroma = PlaneBase(this, in_ddr, src_ch_ptr_[1], x + stride1 * (y / 2));
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
          const uint32_t uv = (j & ~1u) + stride1 * (i >> 1);
          plane_in_[0][i * cols + j] = luma[i * stride0 + j];
          plane_in_[1][i * cols + j] = chroma[src_format_ == 0 ? uv : uv + 1];
          plane_in_[2][i * cols + j] = chroma[src_format_ == 0 ? uv + 1 : uv];
        }
      }
      break;
    }
    case 2: {   // I420: Y, U and V planes, chroma at half width and height
      const int64_t cx = static_cast<int32_t>(src_x_) / 2;
      const uint32_t cy = y / 2;
      const uint8_t *luma = PlaneBase(this, in_ddr, src_ch_ptr_[0], x + stride0 * y);
      const uint8_t *u = PlaneBase(this, in_ddr, src_ch_ptr_[1], cy * stride1 + cx);
      const uint8_t *v = PlaneBase(this, in_ddr, src_ch_ptr_[2], cy * stride2 + cx);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
          plane_in_[0][i * cols + j] = luma[i * stride0 + j];
          plane_in_[1][i * cols + j] = u[(i >> 1) * stride1 + (j >> 1)];
          plane_in_[2][i * cols + j] = v[(i >> 1) * stride2 + (j >> 1)];
        }
      }
      break;
    }
    case 3: {   // planar, 1..4 planes (channel_cfg)
      const uint8_t *p0 = PlaneBase(this, in_ddr, src_ch_ptr_[0], x + stride0 * y);
      const uint8_t *p1 = PlaneBase(this, in_ddr, src_ch_ptr_[1], x + stride1 * y);
      const uint8_t *p2 = PlaneBase(this, in_ddr, src_ch_ptr_[2], x + stride2 * y);
      const uint8_t *p3 = PlaneBase(this, in_ddr, src_ch_ptr_[3], x + stride3 * y);
      const int32_t planes = static_cast<int32_t>(channel_cfg_);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
          plane_in_[0][i * cols + j] = p0[i * stride0 + j];
          if (planes > 1) plane_in_[1][i * cols + j] = p1[i * stride1 + j];
          if (planes > 2) plane_in_[2][i * cols + j] = p2[i * stride2 + j];
          if (planes > 3) plane_in_[3][i * cols + j] = p3[i * stride3 + j];
        }
      }
      break;
    }
    case 4: {   // packed RGB888
      const uint8_t *p = PlaneBase(this, in_ddr, src_ch_ptr_[0], 3 * x + stride0 * y);
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
          const uint32_t s = i * stride0 + 3 * j;
          plane_in_[0][i * cols + j] = p[s];
          plane_in_[1][i * cols + j] = p[s + 1];
          plane_in_[2][i * cols + j] = p[s + 2];
        }
      }
      break;
    }
    case 5: {   // raw16: two bytes per sample, two samples (planes 0/1 and 2/3) per pixel
      const uint8_t *pa = PlaneBase(this, in_ddr, src_ch_ptr_[0], 2 * x + stride0 * y);
      const uint8_t *pb = PlaneBase(this, in_ddr, src_ch_ptr_[1], 2 * x + stride1 * y);
      const bool second = static_cast<int32_t>(channel_cfg_) > 1;
      for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
          const uint32_t s = i * stride0 + 2 * j;
          plane_in_[0][i * cols + j] = pa[s];
          plane_in_[1][i * cols + j] = pa[s + 1];
          if (second) {
            // verified against asm @0x461a90: planes 2/3 come from the plane-1 pointer but are indexed with the row pitch of plane 0 (as in the binary).
            plane_in_[2][i * cols + j] = pb[s];
            plane_in_[3][i * cols + j] = pb[s + 1];
          }
        }
      }
      break;
    }
    default:
      break;
  }
}

// @0x4623a0 (Source AI2D13)
// Computes, for every output pixel (x, y), the source coordinate (M0*x + M1*y + M2, M3*x + M4*y + M5)
// in fixed point: integer parts into src_xi/src_yi (clamped to -1..4096) and, for bilinear
// interpolation, 8-bit fractions into src_xf/src_yf.
// verified against asm @0x4623a0 / 0x4672d0: fp_to_int32 is only vroundss $0xC + vcvttss2si (no scaling), so the
// fixed-point format is whatever the host put into M0..M5: the sum is shifted >> 10 for the integer part and >> 2 (low
// byte) for the fraction.  Float maths as in the binary: x*M0, x*M3 per column; y*M1+M2, y*M4+M5 per row (vmulss, vaddss);
// round offset 512/2 (cord_round 0) or 1024/4 (cord_round 1) for nearest/bilinear, 0 otherwise.
void AI2D::cord_calc()
{
  const uint32_t w = dst_width_shape_;
  const uint32_t h = dst_height_shape_;
  std::vector<int32_t> x_from_col(w), y_from_col(w);

  int round_offset;
  if (cord_round_ == 0)
    round_offset = interpolation_ == 0 ? 512 : 2;
  else if (cord_round_ == 1)
    round_offset = interpolation_ == 0 ? 1024 : 4;
  else
    round_offset = 0;

  for (uint32_t x = 0; x < w; ++x) {
    x_from_col[x] = fp_to_int32(static_cast<float>(x) * m_[0]);
    y_from_col[x] = fp_to_int32(static_cast<float>(x) * m_[3]);
  }

  for (uint32_t y = 0; y < h; ++y) {
    const int32_t x_from_row = fp_to_int32(static_cast<float>(y) * m_[1] + m_[2]);
    const int32_t y_from_row = fp_to_int32(static_cast<float>(y) * m_[4] + m_[5]);
    for (uint32_t x = 0; x < w; ++x) {
      const int32_t fx = round_offset + x_from_col[x] + x_from_row;
      const int32_t fy = round_offset + y_from_col[x] + y_from_row;
      const int32_t xi = fx >> 10;
      const int32_t yi = fy >> 10;
      if (interpolation_) {
        uint8_t &frac_x = src_xf_[x][y];
        uint8_t &frac_y = src_yf_[x][y];
        frac_x = static_cast<uint8_t>(fx >> 2);
        frac_y = static_cast<uint8_t>(fy >> 2);
        // bound_smooth: 0 = zero the fraction for coordinates < 0, 1 = for coordinates < -1.
        if (bound_smooth_ == 0) {
          if (xi < 0) frac_x = 0;
          if (yi < 0) frac_y = 0;
        } else if (bound_smooth_ == 1) {
          if (xi < -1) frac_x = 0;
          if (yi < -1) frac_y = 0;
        }
      }
      src_xi_[x][y] = static_cast<uint16_t>(xy_climp(xi, -1, 4096));
      src_yi_[x][y] = static_cast<uint16_t>(xy_climp(yi, -1, 4096));
    }
  }
}

// @0x4626a0 (Source AI2D14)
// Resamples the loaded planes: for every output pixel and plane fetches the 2x2 neighbourhood at
// (src_xi, src_yi) (out-of-range samples are replaced by the edge pixel when bound_ind is set, otherwise
// by the constant bound_val), then writes nearest (interpolation == 0) or bilinear results to interp_planes.
void AI2D::inter_calc()
{
  const int32_t sw = static_cast<int32_t>(src_width_shape_);
  const int32_t sh = static_cast<int32_t>(src_height_shape_);

  // Number of planes to process.  Signed data is biased by +0x80 first.
  bool bias[4] = {false, false, false, false};
  if (src_format_ == 3) {
    channel_ = channel_cfg_;
    bias[0] = bias[1] = bias[2] = bias[3] = is_signed_ != 0;
  } else if (src_format_ == 5) {
    channel_ = 2 * channel_cfg_;
    bias[1] = bias[3] = is_signed_ != 0;       // only the high bytes of the 16-bit samples
  } else {
    channel_ = 3;
    bias[0] = bias[1] = bias[2] = bias[3] = is_signed_ != 0;
  }
  for (int k = 0; k < 4; ++k) {
    if (!bias[k])
      continue;
    for (uint32_t i = 0; i < src_height_shape_; ++i)
      for (uint32_t j = 0; j < src_width_shape_; ++j)
        plane_in_[k][i * src_width_shape_ + j] += 0x80;
  }

  for (int k = 0; k < 4; ++k)
    std::memset(padded_planes_[k], 0, 0x400000);

  const uint32_t w = dst_width_shape_;
  const uint32_t h = dst_height_shape_;
  int32_t low_part = 0;   // low byte accumulator of 16-bit (src_format 5) samples, carried from even to odd plane
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      for (uint32_t k = 0; k < channel_; ++k) {
        const int32_t xi = static_cast<int16_t>(src_xi_[x][y]);
        const int32_t yi = static_cast<int16_t>(src_yi_[x][y]);
        const uint8_t *src = plane_in_[k];

        // Value of a sample outside the loaded rectangle when bound_ind == 0.
        uint8_t border;
        if (src_format_ == 5) {
          if ((k & ~2u) == 1)
            border = static_cast<uint8_t>(((bound_val_ >> 8) & 0xFF) + (is_signed_ << 7));
          else
            border = static_cast<uint8_t>(bound_val_);
        } else {
          border = static_cast<uint8_t>(bound_val_ + (is_signed_ << 7));
        }
        auto fetch = [&](int32_t sx, int32_t sy) -> uint8_t {
          if (sx >= 0 && sx < sw && sy >= 0 && sy < sh)
            return src[sy * sw + sx];
          if (bound_ind_) {    // replicate the edge pixel
            const int32_t cx = sx < 0 ? 0 : (sx >= sw ? sw - 1 : sx);
            const int32_t cy = sy < 0 ? 0 : (sy >= sh ? sh - 1 : sy);
            return src[cx + sw * cy];
          }
          return border;
        };

        const uint8_t p00 = fetch(xi, yi);
        const uint8_t p10 = fetch(xi + 1, yi);
        const uint8_t p01 = fetch(xi, yi + 1);
        const uint8_t p11 = fetch(xi + 1, yi + 1);

        const size_t out = x + static_cast<size_t>(y) * w;
        padded_planes_[k][out] = p00;      // scratch; slice_padding clears padded_planes again
        if (interpolation_) {
          const int32_t frac_y = src_yf_[x][y];
          const int32_t frac_x = src_xf_[x][y];
          const int32_t v = (p11 + p00 - p10 - p01) * frac_x * frac_y +
                            (((p00 << 8) - frac_y * (p00 - p01) - (p00 - p10) * frac_x) << 8);
          if (src_format_ == 5) {
            if ((k & ~2u) != 0) {
              // odd plane: combine with the low part of the previous plane into a 16-bit sample
              const uint32_t sum = static_cast<uint32_t>((v << 8) + low_part);
              const int64_t s16 = xy_climp(sum, 0, 0xFFFFFFFFLL) >> 16;
              interp_planes_[k - 1][out] = static_cast<uint8_t>(s16);
              interp_planes_[k][out] = static_cast<uint8_t>(s16 >> 8);
            } else {
              low_part = v + 0x8000;
            }
          } else {
            interp_planes_[k][out] =
                static_cast<uint8_t>(static_cast<uint32_t>(xy_climp(static_cast<uint32_t>(v + 0x8000), 0, 0xFFFFFF)) >> 16);
          }
        } else {
          interp_planes_[k][out] = p00;
        }
      }
    }
  }
}

// @0x4630e0 (Source AI2D15)
// In-place YUV444 -> RGB888 on interp_planes[0..2] using the 3x4 coefficient matrix (12-bit signed,
// scaled by 256; the offsets are added after the shift).
// verified against asm @0x4630e0: signed imull, arithmetic sarl $8 on (sum + 128), then + offset, then xy_climp(.., 0, 255).
void AI2D::YUV444toRGB888()
{
  const uint32_t count = dst_width_shape_ * dst_height_shape_;
  const int32_t *c = yuv2rgb_coef_;
  for (uint32_t i = 0; i < count; ++i) {
    const int32_t a = interp_planes_[0][i];
    const int32_t b = interp_planes_[1][i];
    const int32_t d = interp_planes_[2][i];
    const int64_t r0 = xy_climp(c[3] + ((b * c[1] + a * c[0] + d * c[2] + 128) >> 8), 0, 255);
    const int64_t r1 = xy_climp(c[7] + ((b * c[5] + a * c[4] + d * c[6] + 128) >> 8), 0, 255);
    const int64_t r2 = xy_climp(c[11] + ((b * c[9] + c[8] * a + d * c[10] + 128) >> 8), 0, 255);
    interp_planes_[0][i] = static_cast<uint8_t>(r0);
    interp_planes_[1][i] = static_cast<uint8_t>(r1);
    interp_planes_[2][i] = static_cast<uint8_t>(r2);
  }
}

// @0x463270 (Source AI2D16)
// Applies the raw16 shift (rounding right shift, or left shift for shift <= 0) with saturation to the
// 16-bit range (signed data is re-centred by 0x8000) and, for 8-bit destinations, to 8 bits.
int64_t AI2D::raw16_data_shift(int value)
{
  int64_t result;
  if (is_signed_) {
    const int v = value - 0x8000;
    if (shift_ <= 0)
      result = xy_climp(v << -shift_, -32768, 0x7FFF);
    else
      result = xy_climp((v + (1 << (shift_ - 1))) >> shift_, -32768, 0x7FFF);
  } else if (shift_ <= 0) {
    result = xy_climp(value << -shift_, 0, 0xFFFF);
  } else {
    result = xy_climp((value + (1 << (shift_ - 1))) >> shift_, 0, 0xFFFF);
  }
  if (dst_format_ == 3) {
    if (is_signed_)
      return xy_climp(static_cast<int>(result), -128, 127);
    return xy_climp(static_cast<int>(result), 0, 255);
  }
  return result;
}

// @0x463360 (Source AI2D17)
// Pads the resampled image (interp_planes, dst_width x dst_height) into padded_planes
// (pad_width x pad_height) according to pad_mod: 0 constant, 1 replicate edge, 2 mirror.
// For raw16 sources the 16-bit samples are first shifted/saturated (raw16_data_shift); for signed
// 8-bit data the samples are biased by 0x80.
void AI2D::slice_padding()
{
  uint8_t **in = interp_planes_;
  uint8_t **out = padded_planes_;
  const uint32_t rows = dst_height_shape_;
  const uint32_t cols = dst_width_shape_;

  if (src_format_ == 5) {
    // raw16: plane 0/1 = low/high byte of sample 0, plane 2/3 = low/high byte of sample 1
    for (uint32_t y = 0; y < rows; ++y) {
      for (uint32_t x = 0; x < cols; ++x) {
        const uint32_t i = x + y * cols;
        const int16_t s0 = static_cast<int16_t>(raw16_data_shift(in[0][i] + (in[1][i] << 8)));
        in[0][i] = static_cast<uint8_t>(s0);
        if (dst_format_ != 3)
          in[1][i] = static_cast<uint8_t>(s0 >> 8);
        if (static_cast<int32_t>(channel_cfg_) > 1) {
          const int16_t s1 = static_cast<int16_t>(raw16_data_shift(in[2][i] + (in[3][i] << 8)));
          if (dst_format_ == 3) {
            in[1][i] = static_cast<uint8_t>(s1);
          } else {
            in[2][i] = static_cast<uint8_t>(s1);
            in[3][i] = static_cast<uint8_t>(s1 >> 8);
          }
        }
      }
    }
  } else if (is_signed_) {
    for (uint32_t y = 0; y < rows; ++y) {
      for (uint32_t x = 0; x < cols; ++x) {
        const uint32_t i = x + y * cols;
        for (int p = 0; p < 4; ++p)
          in[p][i] = static_cast<uint8_t>(in[p][i] + 0x80);
      }
    }
  }

  for (int p = 0; p < 4; ++p)
    std::memset(out[p], 0, 0x400000);

  const uint32_t out_rows = pad_height_;
  const uint32_t out_cols = pad_width_;
  switch (pad_mod_) {
    case 0:   // constant border
      for (uint32_t y = 0; y < out_rows; ++y) {
        for (uint32_t x = 0; x < out_cols; ++x) {
          const uint32_t o = x + y * out_cols;
          if (pad_t_ > y || pad_t_ + rows <= y || pad_l_ > x || pad_l_ + cols <= x) {
            for (int p = 0; p < 4; ++p)
              out[p][o] = const_pad_ch_[p];
          } else {
            const uint32_t i = x + cols * (y - pad_t_) - pad_l_;
            for (int p = 0; p < 4; ++p)
              out[p][o] = in[p][i];
          }
        }
      }
      break;
    case 1:   // replicate the edge pixels
    case 2:   // mirror
      for (uint32_t y = 0; y < out_rows; ++y) {
        for (uint32_t x = 0; x < out_cols; ++x) {
          int sy, sx;
          if (pad_mod_ == 1) {
            sy = static_cast<int>(xy_climp(static_cast<int>(y - pad_t_), 0, rows - 1));
            sx = static_cast<int>(xy_climp(static_cast<int>(x - pad_l_), 0, cols - 1));
          } else {
            sy = static_cast<int>(mirror_climp(static_cast<int>(y - pad_t_), 0, rows - 1));
            sx = static_cast<int>(mirror_climp(static_cast<int>(x - pad_l_), 0, cols - 1));
          }
          const uint32_t i = static_cast<uint32_t>(sx + sy * cols);
          const uint32_t o = x + y * out_cols;
          for (int p = 0; p < 4; ++p)
            out[p][o] = in[p][i];
        }
      }
      break;
    default:
      printf("pad_mod=%d, config error,it must be 0/1/2 ", pad_mod_);
      break;
  }
}

// @0x463b00 (Source AI2D18)
// Converts YUV->RGB if requested, pads the image and writes it to the destination planes in DDR or
// GLB (dst_ind) in the destination format, then finishes with slice_factor_cnt.
// verified against asm @0x463b00: dst_x is a signed 32-bit value in every format (movslq 0xb0(%rbp), or the sign-correcting
// shr $31 / sar for x/2); packed RGB888 and raw16 compute 3*x / 2*x in 32 bits and sign-extend it (cltq / movslq).
void AI2D::slice_store()
{
  if (csc_en_ == 1)
    YUV444toRGB888();
  pad_height_ = pad_b_ + dst_height_shape_ + pad_t_;
  pad_width_ = pad_r_ + dst_width_shape_ + pad_l_;
  slice_padding();

  const bool in_ddr = dst_ind_ != 0;
  const uint32_t rows = pad_height_;
  const uint32_t cols = pad_width_;
  uint8_t *const *pad = padded_planes_;
  const int64_t x0 = dst_x_;
  const uint32_t y0 = dst_y_;
  const uint32_t pitch0 = dst_width_layout_[0], pitch1 = dst_width_layout_[1];
  const uint32_t pitch2 = dst_width_layout_[2], pitch3 = dst_width_layout_[3];

  switch (dst_format_) {
    case 0:     // NV12: Y plane + interleaved UV plane at half height
    case 1: {   // NV21: same, VU order
      uint8_t *luma = PlaneBase(this, in_ddr, dst_ch_ptr_[0], x0 + y0 * pitch0);
      uint8_t *chroma = PlaneBase(this, in_ddr, dst_ch_ptr_[1], x0 + pitch1 * (y0 / 2));
      for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
          luma[x + y * pitch0] = pad[0][x + y * cols];
          if (((x | y) & 1) == 0) {
            const uint32_t c = (x & ~1u) + (y >> 1) * pitch1;
            chroma[c + (dst_format_ == 0 ? 0 : 1)] = pad[1][x + y * cols];
            chroma[c + (dst_format_ == 0 ? 1 : 0)] = pad[2][x + y * cols];
          }
        }
      }
      break;
    }
    case 2: {   // I420: Y, U, V planes; chroma at half width and height
      uint8_t *luma = PlaneBase(this, in_ddr, dst_ch_ptr_[0], x0 + y0 * pitch0);
      uint8_t *u = PlaneBase(this, in_ddr, dst_ch_ptr_[1], (y0 / 2) * pitch1 + x0 / 2);
      uint8_t *v = PlaneBase(this, in_ddr, dst_ch_ptr_[2], x0 / 2 + pitch2 * (y0 / 2));
      for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
          luma[x + y * pitch0] = pad[0][x + y * cols];
          if (((x | y) & 1) == 0) {
            u[(x >> 1) + (y >> 1) * pitch1] = pad[1][x + y * cols];
            v[(y >> 1) * pitch2 + (x >> 1)] = pad[2][x + y * cols];
          }
        }
      }
      break;
    }
    case 3: {   // planar, dst_channel planes
      uint8_t *p0 = PlaneBase(this, in_ddr, dst_ch_ptr_[0], x0 + y0 * pitch0);
      uint8_t *p1 = PlaneBase(this, in_ddr, dst_ch_ptr_[1], x0 + y0 * pitch1);
      uint8_t *p2 = PlaneBase(this, in_ddr, dst_ch_ptr_[2], x0 + y0 * pitch2);
      uint8_t *p3 = PlaneBase(this, in_ddr, dst_ch_ptr_[3], x0 + pitch3 * y0);
      const int32_t planes = static_cast<int32_t>(dst_channel_);
      for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
          p0[x + y * pitch0] = pad[0][x + y * cols];
          if (planes > 1) p1[x + y * pitch1] = pad[1][x + y * cols];
          if (planes > 2) p2[x + y * pitch2] = pad[2][x + y * cols];
          if (planes > 3) p3[x + y * pitch3] = pad[3][x + y * cols];
        }
      }
      break;
    }
    case 4: {   // packed RGB888
      uint8_t *p = PlaneBase(this, in_ddr, dst_ch_ptr_[0], 3 * x0 + y0 * pitch0);
      for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
          const uint32_t o = 3 * x + y * pitch0;
          p[o] = pad[0][x + y * cols];
          p[o + 1] = pad[1][x + y * cols];
          p[o + 2] = pad[2][x + y * cols];
        }
      }
      break;
    }
    case 5: {   // raw16: little-endian 16-bit samples; planes 0/1 = sample 0, planes 2/3 = sample 1
      const int64_t xb = static_cast<int64_t>(static_cast<int32_t>(2u * static_cast<uint32_t>(dst_x_)));   // movslq of leal (%rax,%rax)
      uint8_t *s0 = PlaneBase(this, in_ddr, dst_ch_ptr_[0], xb + y0 * pitch0);
      uint8_t *s1 = PlaneBase(this, in_ddr, dst_ch_ptr_[1], xb + pitch1 * y0);
      const int32_t planes = static_cast<int32_t>(dst_channel_);
      for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
          const uint32_t o0 = 2 * x + y * pitch0;
          s0[o0] = pad[0][x + y * cols];
          s0[o0 + 1] = pad[1][x + y * cols];
          if (planes > 1) {
            const uint32_t o1 = 2 * x + y * pitch1;
            s1[o1] = pad[2][x + y * cols];
            s1[o1 + 1] = pad[3][x + y * cols];
          }
        }
      }
      break;
    }
    default:
      break;
  }
  slice_factor_cnt(dst_format_);
}

// @0x464640 (Source AI2D19)
// Allocates the working buffers: 4 interpolated planes and 4 padded planes of 4 MiB each (zero
// filled) plus the per-output-pixel coordinate tables src_xi/src_yi (uint16) and src_xf/src_yf
// (uint8), each [dst_width][dst_height] and zero filled.
// verified against asm @0x464640: the 4 MiB plane fill loops store a zeroed ymm (vpxor + vmovdqu / vextractf128, 32 bytes per
// iteration), i.e. zero fill; the coordinate tables are memset to 0 as well.
void AI2D::mem_init()
{
  constexpr size_t kPlaneBytes = 0x400000;
  const uint32_t cols = dst_width_shape_;
  const uint32_t rows = dst_height_shape_;

  interp_planes_ = new uint8_t *[4];
  for (int p = 0; p < 4; ++p)
    interp_planes_[p] = new uint8_t[kPlaneBytes]();
  padded_planes_ = new uint8_t *[4];
  for (int p = 0; p < 4; ++p)
    padded_planes_[p] = new uint8_t[kPlaneBytes]();

  src_xi_ = new uint16_t *[cols];
  for (uint32_t x = 0; x < cols; ++x)
    src_xi_[x] = new uint16_t[rows]();
  src_yi_ = new uint16_t *[cols];
  for (uint32_t x = 0; x < cols; ++x)
    src_yi_[x] = new uint16_t[rows]();
  src_xf_ = new uint8_t *[cols];
  for (uint32_t x = 0; x < cols; ++x)
    src_xf_[x] = new uint8_t[rows]();
  src_yf_ = new uint8_t *[cols];
  for (uint32_t x = 0; x < cols; ++x)
    src_yf_[x] = new uint8_t[rows]();
}

// @0x464970 (Source AI2D20)
// Releases everything allocated by mem_init.
void AI2D::mem_delete()
{
  for (uint32_t x = 0; x < dst_width_shape_; ++x) {
    delete[] src_xi_[x];
    delete[] src_yi_[x];
    delete[] src_xf_[x];
    delete[] src_yf_[x];
  }
  delete[] src_xi_;
  delete[] src_yi_;
  delete[] src_xf_;
  delete[] src_yf_;
  for (int p = 0; p < 4; ++p) {
    delete[] interp_planes_[p];
    delete[] padded_planes_[p];
  }
  delete[] interp_planes_;
  delete[] padded_planes_;
}

// @0x464ac0 (Source AI2D21)
// Runs one AI2D job: latches the memory bases, then load -> coordinate calc -> resample -> store.
void AI2D::ai2d_proc()
{
  ddr_ = g_DDR;
  // The decompiler showed an overlap check plus a vectorised copy; both paths copy the 16 GLB bases.
  for (int bank = 0; bank < 16; ++bank)
    glb_[bank] = g_GLB[bank];
  ai_2d_para_update();
  ai_2d_para_print();
  mem_init();
  print_ddr_data();
  slice_ld();
  cord_calc();
  inter_calc();
  slice_store();
  mem_delete();
}

namespace {

uint32_t *U32(int32_t &field) { return reinterpret_cast<uint32_t *>(&field); }

// Registers settable by name in the text parameter file read by para_parser_back.
// verified against asm @0x465430: the binary's key list is exactly the keys below (38 strcmp/memcmp
// literals); dst_ch1/dst_ch2_width_layout, channel_cfg, csc_en, ... are not recognised.
std::vector<std::pair<std::string, uint32_t *>> NamedParams(AI2D &a)
{
  std::vector<std::pair<std::string, uint32_t *>> keys;
  for (int i = 0; i < 4; ++i)
    keys.emplace_back("src_ch" + std::to_string(i) + "_ptr", &a.src_ch_ptr_[i]);
  keys.emplace_back("src_x", U32(a.src_x_));
  keys.emplace_back("src_y", &a.src_y_);
  for (int i = 0; i < 4; ++i)
    keys.emplace_back("dst_ch" + std::to_string(i) + "_ptr", &a.dst_ch_ptr_[i]);
  keys.emplace_back("dst_x", U32(a.dst_x_));
  keys.emplace_back("dst_y", &a.dst_y_);
  for (int i = 0; i < 6; ++i)
    keys.emplace_back("M" + std::to_string(i), &a.m_raw_[i]);
  keys.emplace_back("interpolation", &a.interpolation_);
  keys.emplace_back("cord_round", &a.cord_round_);
  keys.emplace_back("channel", &a.channel_);
  for (int i = 0; i < 4; ++i)
    keys.emplace_back("src_ch" + std::to_string(i) + "_width_layout", &a.src_width_layout_[i]);
  keys.emplace_back("dst_ch0_width_layout", &a.dst_width_layout_[0]);
  keys.emplace_back("dst_ch3_width_layout", &a.dst_width_layout_[3]);
  keys.emplace_back("src_height_shape", &a.src_height_shape_);
  keys.emplace_back("src_width_shape", &a.src_width_shape_);
  keys.emplace_back("dst_height_shape", &a.dst_height_shape_);
  keys.emplace_back("dst_width_shape", &a.dst_width_shape_);
  keys.emplace_back("src_format", &a.src_format_);
  keys.emplace_back("dst_format", &a.dst_format_);
  keys.emplace_back("src_ind", &a.src_ind_);
  keys.emplace_back("dst_ind", &a.dst_ind_);
  keys.emplace_back("shift", U32(a.shift_));
  keys.emplace_back("bound_ind", &a.bound_ind_);
  keys.emplace_back("bound_val", &a.bound_val_);
  keys.emplace_back("pad_l", &a.pad_l_);
  keys.emplace_back("pad_r", &a.pad_r_);
  keys.emplace_back("pad_t", &a.pad_t_);
  keys.emplace_back("pad_b", &a.pad_b_);
  keys.emplace_back("pad_mod", &a.pad_mod_);
  keys.emplace_back("const_pad", &a.const_pad_);
  return keys;
}

}  // namespace

// @0x465430 (Source AI2D22)
// Reads a text parameter file with lines "<name> <decimal value>" and sets the named registers.
void AI2D::para_parser_back(const char *path)
{
  std::ifstream file(path, std::ios::in);
  const auto keys = NamedParams(*this);
  std::string line;
  for (;;) {
    std::getline(file, line);
    if (file.rdstate() & (std::ios::failbit | std::ios::badbit))   // verified against asm @0x4655c0: testb $0x5 (bad|fail); a last line without '\n' is still parsed
      break;
    std::vector<std::string> tokens;
    Stringsplit(line, ' ', tokens);
    if (tokens.size() < 2)
      continue;   // verified against asm @0x465aa9: the binary has no size check (reads tokens[1] unconditionally, UB for short lines); skipping is a safe superset
    for (const auto &entry : keys) {
      if (entry.first == tokens[0]) {
        *entry.second = static_cast<uint32_t>(strtol(tokens[1].c_str(), nullptr, 10));
        break;
      }
    }
  }
  // Same tail as ai_2d_para_update (inlined in the binary): sign-extend the coefficients, latch M4/M5.
  ai_2d_para_update();
}

// @0x466460 (Source AI2D23)
// Reads a register dump: every line is "<8 chars> <32 hex digits>" = tag + four 32-bit register words
// (w0..w3, 8 hex digits each). Line n (0-based) is decoded like the n-th Extrw register group.
void AI2D::para_parser(const char *path)
{
  std::ifstream file(path, std::ios::in);
  std::string line;
  int line_no = 0;
  for (;;) {
    std::getline(file, line);
    if (file.rdstate() & (std::ios::eofbit | std::ios::badbit))   // last line without '\n' is dropped
      break;
    // line.substr(0, 8) is the tag; it is not used.
    const std::string words = line.substr(9, 0x20);   // throws std::out_of_range on short lines, as the binary
    const uint32_t w0 = static_cast<uint32_t>(strtol(words.substr(0, 8).c_str(), nullptr, 16));
    const uint32_t w1 = static_cast<uint32_t>(strtol(words.substr(8, 8).c_str(), nullptr, 16));
    const uint32_t w2 = static_cast<uint32_t>(strtol(words.substr(16, 8).c_str(), nullptr, 16));
    const uint32_t w3 = static_cast<uint32_t>(strtol(words.substr(24, 8).c_str(), nullptr, 16));

    switch (line_no) {
      case 0: case 1: case 3:
        break;   // groups without register fields here (pointers are not read from the dump)
      case 2:   // row pitches
        dst_width_layout_[3] = w0 >> 16;
        dst_width_layout_[2] = w0 & 0xFFFF;
        dst_width_layout_[1] = w1 >> 16;
        dst_width_layout_[0] = w1 & 0xFFFF;
        src_width_layout_[2] = w2 & 0xFFFF;
        src_width_layout_[3] = w2 >> 16;
        src_width_layout_[0] = w3 & 0xFFFF;
        src_width_layout_[1] = w3 >> 16;
        break;
      case 4:   // formats / modes
        bound_smooth_ = (w0 >> 16) & 1;
        bound_val_ = w0 & 0xFFFF;
        dst_format_ = w1 >> 28;
        src_format_ = (w1 >> 24) & 0xF;
        bound_ind_ = (w1 >> 20) & 0xF;
        {
          int32_t field = static_cast<uint8_t>(w1 >> 12);
          if (((w1 >> 12) & 0xF0) != 0)
            field -= 32;
          shift_ = field;
        }
        pad_mod_ = (w1 >> 10) & 3;
        interpolation_ = (w1 >> 8) & 3;
        cord_round_ = (w1 & 0xFF) >> 6;
        dst_channel_ = (w1 >> 3) & 7;
        channel_cfg_ = channel_ = w1 & 7;
        break;
      case 5:   // YUV->RGB coefficients 0..7
        yuv2rgb_coef_[7] = (w0 >> 12) & 0xFFF;
        yuv2rgb_coef_[6] = w0 & 0xFFF;
        yuv2rgb_coef_[5] = (w1 >> 12) & 0xFFF;
        yuv2rgb_coef_[4] = w1 & 0xFFF;
        yuv2rgb_coef_[2] = w2 & 0xFFF;
        yuv2rgb_coef_[3] = (w2 >> 12) & 0xFFF;
        yuv2rgb_coef_[0] = w3 & 0xFFF;
        yuv2rgb_coef_[1] = (w3 >> 12) & 0xFFF;
        break;
      case 6:   // coefficients 8..11, constant padding, indirection flags
        const_pad_ch_[1] = static_cast<uint8_t>(w2);          // +369 (16-bit store: bytes 1 and 2)
        const_pad_ch_[2] = static_cast<uint8_t>(w2 >> 8);
        yuv2rgb_coef_[9] = (w1 >> 12) & 0xFFF;
        yuv2rgb_coef_[8] = w1 & 0xFFF;
        is_signed_ = (w2 & 0x8000000) != 0;
        cmd_id_ = (w2 & 0x4000000) != 0;
        dst_ind_ = (w2 >> 25) & 1;
        src_ind_ = (w2 >> 24) & 1;
        const_pad_ch_[3] = static_cast<uint8_t>(w2 >> 16);
        const_pad_ch_[0] = static_cast<uint8_t>(w3 >> 24);
        yuv2rgb_coef_[10] = w3 & 0xFFF;
        yuv2rgb_coef_[11] = (w3 >> 12) & 0xFFF;
        break;
      case 7:   // shapes and padding
        csc_en_ = w0 >> 31;
        intr_mask_ = (w0 >> 30) & 1;
        dst_height_shape_ = (w0 >> 16) & 0x1FFF;
        dst_width_shape_ = w0 & 0x1FFF;
        src_height_shape_ = (w1 >> 16) & 0x1FFF;
        src_width_shape_ = w1 & 0x1FFF;
        pad_l_ = w2 & 0x3FF;
        pad_r_ = (w2 >> 16) & 0x3FF;
        pad_t_ = w3 & 0x3FF;
        pad_b_ = (w3 >> 16) & 0x3FF;
        break;
      case 8:   // transform and window
        m_raw_[5] = w2;
        m_raw_[2] = w3;
        calc_enable_ = w0 >> 31;
        dst_y_ = (w0 >> 16) & 0x1FFF;
        dst_x_ = w0 & 0x1FFF;
        src_y_ = (w1 >> 16) & 0x1FFF;
        src_x_ = w1 & 0x1FFF;
        break;
      default:
        printf("offset not support now");
        exit(0);
    }
    ++line_no;
  }
  // Same tail as ai_2d_para_update (inlined in the binary).
  ai_2d_para_update();
}
