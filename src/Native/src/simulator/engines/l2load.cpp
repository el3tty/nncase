// Lifted from IDA/Hex-Rays output (L2Load1..L2Load5).
#include "engines/l2load.h"
#include "globals.h"
#include "engines/checkpoint.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

extern uint8_t debug_flag;       // set by L2LoadW / DMLoadW when the next weight load should be dumped
extern std::string debug_file;   // declared in globals (path of the dump file)

namespace {

// IEEE binary32 -> binary16, round to nearest even, NaN keeps its sign (quiet), inf/overflow -> inf,
// tiny values flush to signed zero (exponent below 2^-26).
uint16_t Fp32ToFp16(float value)
{
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  const uint32_t sign = (bits >> 31) << 15;
  if (std::isnan(value))
    return (uint16_t)(sign | 0x7E00);
  // verified against asm @0x434e3d: vandps abs, then `vucomiss .rodata 0x481410 (= FLT_MAX 0x7f7fffff); ja` -> +/-inf (i.e. |x| is infinite).
  if (std::isinf(value))
    return (uint16_t)(sign | 0x7C00);

  const uint32_t exp = (bits >> 23) & 0xFF;
  const uint32_t mant = bits & 0x7FFFFF;
  if (exp <= 0x65)                                  // below half of the smallest subnormal
    return (uint16_t)sign;
  if (exp == 0x70) {                                // [2^-15, 2^-14): subnormal, may carry into the exponent
    const uint32_t r = 0x7FFFFF + mant + (1u << 13) + (((0x800000 + mant) >> 14) & 1);
    return (uint16_t)(((r >> 14) & 0x3FF) | (((r >> 24) & 1) << 10) | sign);
  }
  if (exp <= 0x6F) {                                // [2^-25, 2^-15): subnormal
    const uint32_t shift = 126 - exp;
    const uint32_t r = 0x7FFFFF + mant + (1u << (125 - exp)) + (((0x800000 + mant) >> shift) & 1);
    return (uint16_t)(((r >> shift) & 0x83FF) | sign);
  }
  if (exp <= 0x8E) {                                // normal half range
    const uint32_t r = 0x800FFF + mant + ((mant >> 13) & 1);
    uint32_t half_exp, half_mant;
    if (r & 0x1000000) {                            // mantissa rounded up into the next binade
      half_exp = exp - 111;
      half_mant = 0;
    } else {
      half_exp = exp - 112;
      half_mant = (r >> 13) & 0x3FF;
    }
    return (uint16_t)(sign | half_mant | (half_exp << 10));
  }
  return (uint16_t)(sign | 0x7C00);                 // too large
}

// One beat of the load_glb_write check file: "<line:5> <byte mask:8> <32 bytes, MSB first>".
void PrintGlbWriteBeat(std::ofstream & os, uint32_t line, uint32_t byte_mask, const uint8_t * beat)
{
  if (line > 0x1FFFF) {
    std::cerr << "[Error] glb addr exceed 4M" << std::endl;
    exit(-1);
  }
  os << std::setw(5) << std::setfill('0') << line << " "
     << std::setw(8) << std::setfill('0') << byte_mask << " ";
  for (int i = 31; i >= 0; --i)
    os << std::setw(2) << std::setfill('0') << (unsigned)beat[i];
  os << std::endl;
}

}  // namespace

// L2Load1.cpp  @0x434c00
int64_t L2Load::Load()
{
  if (!(row_len_ * row_count_ * plane_count_ * batch_count_))
    return plane_count_;
  if ((int)batch_count_ <= 0)
    return plane_count_;

  // Strides in elements.  Row stride = pitch, plane stride = pitch * dim1, tensor stride = plane * dim0.
  const int dst_plane_stride = (int)(dst_pitch_ * dst_dim1_);
  const int dst_batch_stride = dst_plane_stride * (int)dst_dim0_;
  const int src_plane_stride = (int)(src_pitch_ * src_dim1_);
  const int src_batch_stride = src_plane_stride * (int)src_dim0_;
  const int len = (int)row_len_;
  const uint16_t mode = Mode();

  int dst_batch = 0, src_batch = 0;
  for (int b = 0; b < (int)batch_count_; ++b) {
    int dst_plane = dst_batch, src_plane = src_batch;
    for (int p = 0; p < (int)plane_count_; ++p) {
      int dst_row = dst_plane, src_row = src_plane;
      for (int r = 0; r < (int)row_count_; ++r) {
        if (mode == 513) {
          // float32 (DDR) -> float16 (GLB)
          const uint8_t * s = ddr_ptr_ + 4LL * src_row;
          uint8_t * d = glb_ptr_ + 2LL * dst_row;
          for (int i = 0; i < len; ++i) {
            float f;
            std::memcpy(&f, s + 4 * (size_t)i, sizeof f);
            const uint16_t h = Fp32ToFp16(f);
            std::memcpy(d + 2 * (size_t)i, &h, sizeof h);
          }
        } else if (mode0_ == 0) {
          std::memcpy(glb_ptr_ + dst_row, ddr_ptr_ + src_row, (size_t)len);                  // 8-bit elements
        } else {
          std::memcpy(glb_ptr_ + 2LL * dst_row, ddr_ptr_ + 2LL * src_row, 2 * (size_t)len);  // 16-bit elements
        }
        dst_row += (int)dst_pitch_;
        src_row += (int)src_pitch_;
      }
      dst_plane += dst_plane_stride;
      src_plane += src_plane_stride;
    }
    dst_batch += dst_batch_stride;
    src_batch += src_batch_stride;
  }
  return plane_count_;
}

// L2Load2.cpp  @0x434fe0
// rel_offset: byte offset relative to ddr_offset, num_bytes: number of bytes.  The data is dumped in 256-byte
// chunks of 16-byte lines (MSB first, bytes below an unaligned start are written as 00).
// Returns 0 (the original returned a stream reference left in a register).
int64_t L2Load::PrintLoadDDRCheckPoint(uint32_t rel_offset, uint32_t num_bytes)
{
  CheckPoint * cp = CheckPoint::GetCheckPoint();
  std::ofstream & rdata = cp->Stream(CheckPoint::kLoadDdrRdata);
  std::ofstream & raddr = cp->Stream(CheckPoint::kLoadDdrRaddr);

  uint32_t start = rel_offset + ddr_offset_;
  const uint32_t end = start + num_bytes;
  while (end > start) {
    uint32_t chunk_end = (start & ~0xFFu) + 256;
    if (chunk_end > end)
      chunk_end = end;
    if (chunk_end & 0xF)
      chunk_end = (chunk_end & ~0xFu) + 16;       // lines are always complete

    uint32_t cur = start;
    uint32_t line = (uint32_t)-1;
    while (chunk_end > cur) {
      const uint32_t lane = cur & 0xF;
      for (int k = 15 - (int)lane; k >= 0; --k)   // bytes cur+15-lane .. cur, highest address first
        rdata << std::setw(2) << std::setfill('0') << (unsigned)g_DDR[cur + k];
      for (uint32_t i = 0; i < lane; ++i)         // lanes below the start
        rdata << std::setw(2) << std::setfill('0') << 0;
      rdata << std::endl;
      ++line;
      cur += 16 - lane;
    }
    raddr << std::setw(8) << std::setfill('0') << start << " "
          << std::setw(2) << std::setfill('0') << line << std::endl;
    if (end <= cur)
      break;
    start = cur;
  }
  return 0;
}

// L2Load3.cpp  @0x435410
// Dumps what LoadW writes into the GLB as 32-byte beats (check file load_glb_write.chk).
//   rel_offset: byte offset from glb_mmu_addr, num_bytes: number of bytes, row_len_m1: elements per row group minus one,
//   elem_bytes: element size in bytes.  Row groups are placed at a pitch of 24 elements; with
//   row_len_m1 == 23 the data is contiguous.
// UNCERTAIN (open in asm too): no direct call to @0x435410 exists in the binary; row_len_m1/elem_bytes meanings inferred from LoadW's index formula.
// Returns 0 (the original returned a stream reference left in a register).
int64_t L2Load::PrintWriteGlbCheckPoint(uint32_t rel_offset, uint32_t num_bytes, uint32_t row_len_m1, int elem_bytes)
{
  std::ofstream & os = CheckPoint::GetCheckPoint()->Stream(CheckPoint::kLoadGlbWrite);
  const int total = (int)num_bytes;
  if (total <= 0)
    return 0;

  uint8_t beat[32] = {0};
  uint32_t mask = 0;
  uint32_t addr = glb_mmu_addr_ + rel_offset;               // address in "MMU" space
  int consumed = 0;
  const uint32_t group_bytes = (uint32_t)elem_bytes * (row_len_m1 + 1);
  const uint32_t group_pitch = 24u * (uint32_t)elem_bytes;
  const bool contiguous = (row_len_m1 == 23);

  for (;;) {
    const int remaining = total - consumed;
    const uint32_t lane = addr & 0x1F;             // byte position inside the 32-byte beat
    const uint32_t line = addr >> 5;
    const uint32_t src_off = addr - glb_mmu_addr_;  // offset into glb_ptr

    if (contiguous) {
      uint32_t n = remaining;
      if (32 - lane <= n)
        n = 32 - lane;
      for (uint32_t k = 0; k < n; ++k) {
        beat[lane + k] = glb_ptr_[src_off + k];
        mask |= 1u << (lane + k);
      }
      PrintGlbWriteBeat(os, line, mask, beat);
      consumed += (int)n;
      addr += n;
      std::memset(beat, 0, sizeof beat);
      if (consumed >= total)
        return 0;
      mask = 0;
      continue;
    }

    uint32_t chunk = remaining;
    if (group_bytes <= chunk)
      chunk = group_bytes;
    if (chunk + lane > 32) {                       // the group straddles two beats
      const uint32_t head = 32 - lane;
      for (uint32_t k = 0; k < head; ++k) {
        beat[lane + k] = glb_ptr_[src_off + k];
        mask |= 1u << (lane + k);
      }
      PrintGlbWriteBeat(os, line, mask, beat);
      std::memset(beat, 0, sizeof beat);
      mask = 0;
      for (uint32_t k = 0; k < chunk + lane - 32; ++k) {
        beat[k] = glb_ptr_[src_off + head + k];
        mask |= 1u << k;
      }
      consumed += (int)chunk;
      addr += group_pitch;
      if ((addr >> 5) != line + 1 || consumed >= total) {
        PrintGlbWriteBeat(os, line + 1, mask, beat);
        std::memset(beat, 0, sizeof beat);
        if (consumed >= total)
          return 0;
        mask = 0;
      }
    } else {                                       // the group fits into the current beat
      for (uint32_t k = 0; k < chunk; ++k) {
        beat[lane + k] = glb_ptr_[src_off + k];
        mask |= 1u << (lane + k);
      }
      consumed += (int)chunk;
      addr += group_pitch;
      if ((addr >> 5) != line || consumed >= total) {
        PrintGlbWriteBeat(os, line, mask, beat);
        std::memset(beat, 0, sizeof beat);
        if (consumed >= total)
          return 0;
        mask = 0;
      }
    }
  }
}

// L2Load4.cpp  @0x4361f0
// Compressed block format: for every 16 output bytes a 16-bit mask (bit k set => byte k present),
// followed by the present bytes.  Missing bytes are zero.
// verified against asm @0x4361f0 (memset + 16-byte mask loop).  The original returns whatever is left in rax (dst when
// dst_len <= 0, otherwise the last mask / last copied byte); no caller uses it, the output pointer is returned here.
uint8_t* L2Load::Decompress(uint8_t * dst, uint8_t * src, int dst_len)
{
  uint8_t * const out = dst;
  std::memset(dst, 0, dst_len > 0 ? (size_t)dst_len : 0);
  if (dst_len > 0) {
    uint8_t * cur = dst;
    uint8_t * const dst_end = dst + 16 * (((unsigned)(dst_len - 1) >> 4) + 1);
    const uint8_t * in = src;
    do {
      const uint16_t mask = (uint16_t)(in[0] | (in[1] << 8));
      in += 2;
      for (int k = 0; k < 16; ++k) {
        if (mask & (1u << k))
          cur[k] = *in++;
      }
      cur += 16;
    } while (cur != dst_end);
  }
  return out;
}

// L2Load5.cpp  @0x436340
void L2Load::LoadW()
{
  // Size of the staging buffer: weight_count elements of the source format.
  int elem_bytes = 1 << w_mode1_;
  if (w_mode1_ == 5)
    elem_bytes = 2;
  int byte_count = (int)weight_count_ * elem_bytes;
  std::vector<uint8_t> staging(byte_count > 0 ? (size_t)byte_count : 0, 0);
  uint8_t * const buf = staging.data();

  if (compressed_) {
    Decompress(buf, ddr_ptr_, byte_count);
  } else if (w_mode1_ == 3) {
    // packed 4-bit values -> one byte each, value in the high nibble
    byte_count = (int)(weight_count_ + 1) >> 1;
    for (int64_t i = 0; i < (int64_t)(int)weight_count_; ++i) {
      const uint8_t packed = ddr_ptr_[(int)i >> 1];
      buf[i] = (i & 1) ? (uint8_t)(packed & 0xF0) : (uint8_t)(packed << 4);
    }
  } else if (w_mode1_ == 4) {
    // packed 6-bit values (4 values in 3 bytes) -> one byte each, value in the upper 6 bits
    const uint32_t n = weight_count_;
    byte_count = (int)((6 * n + 7) >> 3);
    const uint32_t groups = n >> 2;
    for (uint32_t g = 0; g < groups; ++g) {
      uint32_t w;
      std::memcpy(&w, ddr_ptr_ + 3 * g, sizeof w);
      buf[4 * g + 0] = (uint8_t)(4 * w);
      buf[4 * g + 1] = (uint8_t)(4 * (w >> 6));
      buf[4 * g + 2] = (uint8_t)(4 * (w >> 12));
      buf[4 * g + 3] = (uint8_t)(4 * (w >> 18));
    }
    const uint32_t rest = n & 3;
    if (rest) {
      uint32_t w;
      std::memcpy(&w, ddr_ptr_ + 3 * groups, sizeof w);
      buf[4 * groups] = (uint8_t)(4 * w);
      if (rest != 1) {
        buf[4 * groups + 1] = (uint8_t)(4 * (w >> 6));
        if (rest != 2)
          buf[4 * groups + 2] = (uint8_t)(4 * (w >> 12));
      }
    }
  } else {
    std::memcpy(buf, ddr_ptr_, (size_t)byte_count);
  }

  if (debug_flag) {
    // Hex dump of the staging buffer, one byte per line.
    // verified against asm @0x436340: byte_count (r13d) is overwritten with the packed size for the 4/6-bit formats at
    // @0x4363d2, so the dump length is the packed size (the buffer itself is expanded).
    std::ofstream dump(debug_file.c_str(), std::ios::out);
    dump << std::hex;
    for (int i = 0; i < byte_count; ++i)
      dump << std::setw(2) << std::setfill('0') << (unsigned)buf[i] << std::endl;
    debug_flag = 0;
  }

  // Scatter into the GLB: groups of (row_len_m1 + 1) elements at a pitch of 24 elements.
  const uint16_t mode = WMode();
  const unsigned group = row_len_m1_ + 1;
  for (int i = 0; i < (int)weight_count_; ++i) {
    const int64_t d = (int)((unsigned)i % group) + 24LL * (int)((unsigned)i / group);
    if (mode == 513) {
      float f;
      std::memcpy(&f, buf + 4 * (size_t)i, sizeof f);
      const uint16_t h = Fp32ToFp16(f);
      std::memcpy(glb_ptr_ + 2 * d, &h, sizeof h);
    } else if (w_mode0_) {
      std::memcpy(glb_ptr_ + 2 * d, buf + 2 * (size_t)i, 2);
    } else {
      glb_ptr_[d] = buf[i];
    }
  }
}
