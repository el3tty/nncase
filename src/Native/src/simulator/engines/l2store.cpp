// Lifted from IDA/Hex-Rays output (L2Store1..L2Store3).
#include "engines/l2store.h"
#include "globals.h"
#include "engines/checkpoint.h"
#include <cstring>
#include <fstream>
#include <iomanip>

// The check-point streams are opened in hexadecimal mode (see CheckPoint), so integers are written as hex.

namespace {

// IEEE binary16 -> binary32 (exact; subnormals are normalised, inf/NaN keep their mantissa).
uint32_t Fp16ToFp32(uint16_t h)
{
  const uint32_t mant = h & 0x3FF;
  const uint32_t sign = (uint32_t)(h >> 15) << 31;
  const uint32_t mant32 = (mant << 13) | sign;      // mantissa already shifted, sign in place
  const uint32_t exp = h & 0x7C00;
  if (exp == 0) {
    if (mant == 0)
      return mant32;                                // +-0
    const int lz = norm_uint((int32_t)mant) - 8;    // subnormal: shift the leading one to bit 23
    return (mant32 & 0x80000000u) | ((mant << lz) & 0x7FFFFF) | ((uint32_t)(uint8_t)(126 - lz) << 23);
  }
  if (exp == 0x7C00)
    return mant32 | 0x7F800000u;                    // inf / NaN
  return mant32 | ((uint32_t)(uint8_t)(((h >> 10) & 0x1F) + 112) << 23);
}

}  // namespace

// L2Store1.cpp  @0x436b30
int64_t L2Store::Store()
{
  if (!(row_len_ * row_count_ * plane_count_ * batch_count_))
    return batch_count_;
  if ((int)batch_count_ <= 0)
    return batch_count_;

  // Strides in elements: row stride = pitch, plane stride = pitch * dim1, tensor stride = plane * dim0.
  const int ddr_plane_stride = (int)(ddr_pitch_ * ddr_dim1_);
  const int ddr_batch_stride = ddr_plane_stride * (int)ddr_dim0_;
  const int glb_plane_stride = (int)(glb_pitch_ * glb_dim1_);
  const int glb_batch_stride = glb_plane_stride * (int)glb_dim0_;
  const int len = (int)row_len_;
  const uint16_t mode = Mode();

  int ddr_batch = 0, glb_batch = 0;
  for (int b = 0; b < (int)batch_count_; ++b) {
    int ddr_plane = ddr_batch, glb_plane = glb_batch;
    for (int p = 0; p < (int)plane_count_; ++p) {
      int ddr_row = ddr_plane, glb_row = glb_plane;
      for (int r = 0; r < (int)row_count_; ++r) {
        if (mode == 513) {
          // float16 (GLB) -> float32 (DDR)
          const uint8_t * s = glb_ptr_ + 2LL * glb_row;
          uint8_t * d = ddr_ptr_ + 4LL * ddr_row;
          for (int i = 0; i < len; ++i) {
            uint16_t h;
            std::memcpy(&h, s + 2 * (size_t)i, sizeof h);
            const uint32_t f = Fp16ToFp32(h);
            std::memcpy(d + 4 * (size_t)i, &f, sizeof f);
          }
        } else if (mode0_ == 0) {
          std::memcpy(ddr_ptr_ + ddr_row, glb_ptr_ + glb_row, (size_t)len);                  // 8-bit elements
        } else {
          std::memcpy(ddr_ptr_ + 2LL * ddr_row, glb_ptr_ + 2LL * glb_row, 2 * (size_t)len);  // 16-bit elements
        }
        ddr_row += (int)ddr_pitch_;
        glb_row += (int)glb_pitch_;
      }
      ddr_plane += ddr_plane_stride;
      glb_plane += glb_plane_stride;
    }
    ddr_batch += ddr_batch_stride;
    glb_batch += glb_batch_stride;
  }
  return batch_count_;
}

// L2Store2.cpp  @0x436e00
// rel_offset: byte offset relative to ddr_offset, num_bytes: number of bytes.  The DDR range is walked in 256-byte
// windows of 16-byte lines.  store_ddr_wdata gets one line per 16-byte group: "<16-bit byte-enable mask>
// <16 bytes, highest address first>"; store_ddr_waddr gets "<window start address> <last line index>".
// Returns 0 (the original returned a stream reference left in a register).
int64_t L2Store::PrintStoreDDRCheckPoint(uint32_t rel_offset, uint32_t num_bytes)
{
  CheckPoint * cp = CheckPoint::GetCheckPoint();
  std::ofstream & wdata = cp->Stream(CheckPoint::kStoreDdrWdata);
  std::ofstream & waddr = cp->Stream(CheckPoint::kStoreDdrWaddr);

  uint32_t start = rel_offset + ddr_offset_;
  const uint32_t end = start + num_bytes;
  // verified against asm @0x436e80: the 16-byte line buffer is zeroed (vpxor/vmovdqu) at the top of every line, so
  // disabled lanes print 00.
  uint8_t line_buf[16] = {0};

  while (end > start) {
    uint32_t window_end = (start & ~0xFFu) + 256;
    if (window_end > end)
      window_end = end;

    uint32_t cur = start;
    uint32_t lane = start & 0xF;
    uint32_t line = (uint32_t)-1;
    while (cur < window_end) {
      std::memset(line_buf, 0, sizeof line_buf);
      uint32_t mask = 0;
      for (; lane < 16 && cur < window_end; ++lane, ++cur) {
        line_buf[lane] = g_DDR[cur];
        mask |= 1u << lane;
      }
      wdata << std::setw(4) << std::setfill('0') << mask << " ";
      for (int i = 15; i >= 0; --i)
        wdata << std::setw(2) << std::setfill('0') << (unsigned)line_buf[i];
      wdata << std::endl;
      ++line;
      lane = 0;
    }
    waddr << std::setw(8) << std::setfill('0') << start << " "
          << std::setw(2) << std::setfill('0') << line << std::endl;
    if (end <= cur)
      break;
    start = cur;
  }
  return 0;
}

// L2Store3.cpp  @0x437560
// Dumps what Store reads from the GLB: store_glb_raddr "<32-byte line:5> <byte mask:8>" and
// store_glb_rdata "<32 bytes, highest first>" per 32-byte beat.  rel_offset: byte offset from glb_mmu_addr,
// num_bytes: number of bytes.  Returns 0 (the original returned the loop counter).
int64_t L2Store::PrintLoadGlbCheckPoint(uint32_t rel_offset, uint32_t num_bytes)
{
  CheckPoint * cp = CheckPoint::GetCheckPoint();
  std::ofstream & raddr = cp->Stream(CheckPoint::kStoreGlbRaddr);
  std::ofstream & rdata = cp->Stream(CheckPoint::kStoreGlbRdata);

  uint32_t addr = glb_mmu_addr_ + rel_offset;
  int done = 0;
  const int total = (int)num_bytes;
  // verified against asm @0x437584: the 32-byte beat buffer is zeroed once (two vmovdqu) before the loop and keeps its
  // contents between beats.
  uint8_t beat[32] = {0};

  while (done < total) {
    uint32_t lane = addr & 0x1F;
    uint32_t n = 32 - lane;
    if ((uint32_t)(total - done) <= n)
      n = (uint32_t)(total - done);
    uint32_t mask = 0;
    for (uint32_t k = 0; k < n; ++k) {
      beat[lane + k] = glb_ptr_[addr - glb_mmu_addr_ + k];
      mask |= 1u << (lane + k);
    }
    raddr << std::setw(5) << std::setfill('0') << (addr >> 5) << " "
          << std::setw(8) << std::setfill('0') << mask << std::endl;
    for (int i = 31; i >= 0; --i)
      rdata << std::setw(2) << std::setfill('0') << (unsigned)beat[i];
    rdata << std::endl;
    done += (int)n;
    addr += n;
  }
  return 0;
}
