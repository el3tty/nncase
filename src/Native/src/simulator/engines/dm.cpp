#include "engines/dm.h"
#include "globals.h"
#include "engines/checkpoint.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// TODO(globals): not declared in globals.h (defined elsewhere in the original binary).
extern std::string debug_file;     // path of the dump file (a COW std::string at 0x5cc7a8 in the original)

namespace {

// L1Helper: (channel, row, col) view into a byte buffer; strides in bytes (same layout as in pdp0.cpp).
// TODO(layout): replace by the real type once its owner defines L1Helper.
struct L1View {
    uint8_t* data_;
    int32_t  max_channels_; // channel capacity of the buffer (24 for IF = PE rows, 32 for PSUM = PE lanes)
    int32_t  chan_stride_;
    int32_t  base_;
    int32_t  row_stride_;
    int32_t  col_stride_;
    int32_t  channels_;
    int32_t  height_;
    int32_t  width_;
};

}  // namespace

// The original constructs the singleton behind a guard variable; the object is the Dm_dm global.
Dm* Dm::GetDm() {
  return reinterpret_cast<Dm*>(_G.Dm_dm);
}

// Dumps the input feature of the DM / PU interface (one vector of per-channel bytes per pixel) to the
// dm_if and Pu1 (pu_if_in) checkpoint streams (verified @0x44d790).
void Dm::PrintDmIfAndPuIfInCkp(L1Helper & helper, L2DataType dataType) {
  const L1View& l1 = reinterpret_cast<const L1View&>(helper);
  CheckPoint* ckp = CheckPoint::GetCheckPoint();
  std::vector<int> lanes(24, 0);   // 0x60 bytes in the original

  // verified against asm @0x44d8a1..0x44d96d: both PrintCheckPoint calls use edx = 2 (width) and ecx = 1 (reverse);
  // first stream +0x1600 (dm_if), then +0x200 (pu_if_in).
  constexpr int kWidth = 2;
  constexpr bool kReverse = true;
  const int lane_count = std::min<int>(l1.channels_, static_cast<int>(lanes.size()));  // the original overruns beyond 24

  auto emit = [&](int c, int i, int plane) {
    for (int k = 0; k < lane_count; ++k)
      lanes[k] = l1.data_[k * l1.chan_stride_ + plane + i * l1.col_stride_ + l1.base_ + c * l1.row_stride_];
    CheckPoint::PrintCheckPoint(ckp->Stream(CheckPoint::kDmIf), lanes, kWidth, kReverse);
    CheckPoint::PrintCheckPoint(ckp->Stream(CheckPoint::kPu1), lanes, kWidth, kReverse);
  };

  for (int c = 0; c < l1.height_; ++c) {
    for (int i = 0; i < l1.width_; ++i) {
      if (dataType == 2) {
        for (int plane = 0; plane != 1024; plane += 512)   // low and high byte plane of 16-bit data
          emit(c, i, plane);
      } else {
        emit(c, i, 0);
      }
    }
  }
}

// Snapshots the DmLoadW operands and copies line_bytes * lines bytes of weights out of the GLB.
std::shared_ptr<DmLoadW> Dm::GetLoadW() const {
  auto load = std::make_shared<DmLoadW>();
  const size_t bytes = static_cast<size_t>(loadw_len_ * loadw_lines_);
  load->weights_ = static_cast<uint8_t*>(std::malloc(loadw_len_ * loadw_lines_));
  load->line_bytes_ = loadw_len_;
  load->deq_mode_ = loadw_deq_mode_;
  std::memcpy(load->weights_, loadw_src0_, bytes);
  load->zero_points_ = loadw_src1_;           // (pointer only, not copied)
  load->use_pdp0_ = loadw_use_pdp0_;
  return load;
}

std::shared_ptr<DmStoreOf> Dm::GetStoreOf() const {
  auto store = std::make_shared<DmStoreOf>();
  std::copy(of_shape_, of_shape_ + 4, store->tile_shape_);
  store->mode_ = of_mode_;
  store->dst_ = of_dst_;
  std::copy(of_full_shape_, of_full_shape_ + 4, store->full_shape_);
  store->mmu_addr_ = of_mmu_addr_;
  store->use_pdp0_ = of_use_pdp0_;
  return store;
}

std::shared_ptr<DmLoadAct0> Dm::GetLoadAct0() const {
  auto load = std::make_shared<DmLoadAct0>();
  load->params_ = loadact0_src_;
  load->use_pdp0_ = loadact0_use_pdp0_;
  load->load_flag_ = loadact0_flag_;
  return load;
}

std::shared_ptr<DmLoadL1> Dm::GetDmLoadL1() const {
  auto load = std::make_shared<DmLoadL1>();
  load->if_snapshot_ = static_cast<uint8_t*>(std::malloc(sizeof(_G.IF_L1)));
  std::copy(l1_shape_, l1_shape_ + 4, load->shape_);
  load->mode_ = l1_mode_;
  std::copy(l1_dims_, l1_dims_ + 4, load->dims_);
  load->src_ = l1_src_;
  load->layout_ = l1_layout_;
  std::memcpy(load->if_snapshot_, _G.IF_L1, sizeof(_G.IF_L1));
  return load;
}

// Copies a GLB region into the IF_L1 buffer (cleared first). IF_L1 is organised as lines of 1024
// bytes; for 16-bit data (mode == 2) the low byte goes to IF_L1[i] and the high byte to IF_L1[i + 512].
void Dm::LoadL1(std::shared_ptr<DmLoadL1> load) {
  const DmLoadL1& l = *load;
  const int n_y = static_cast<int>(l.dims_[1]);   // number of 1024-byte lines per group
  const int n_z = static_cast<int>(l.dims_[2]);
  const int n_x = static_cast<int>(l.dims_[3]);   // elements per row copy
  const int pitch1 = static_cast<int>(l.shape_[1]);   // source pitch factor
  const int pitch2 = static_cast<int>(l.shape_[2]);   // source pitch
  const bool wide = (l.mode_ == 2);
  const uint8_t* src = l.src_;
  int dump_lines;   // number of "lines" printed by the debug dump

  std::memset(_G.IF_L1, 0, sizeof(_G.IF_L1));

  auto store_pair = [&](int dst, uint16_t value) {   // 16-bit element -> two byte planes
    _G.IF_L1[dst] = static_cast<uint8_t>(value);
    _G.IF_L1[dst + 512] = static_cast<uint8_t>(value >> 8);
  };

  if (l.layout_ <= 1) {
    // Contiguous layout: IF_L1[z * n_x + x + 1024 * y] = src[z * pitch2 + x + y * pitch1 * pitch2]
    const int64_t plane_stride = static_cast<int64_t>(pitch1 * pitch2);
    dump_lines = n_z;
    if (n_z > 0 && n_x > 0 && n_y > 0) {
      for (int z = 0; z < n_z; ++z) {
        for (int x = 0; x < n_x; ++x) {
          for (int y = 0; y < n_y; ++y) {
            const int dst = z * n_x + x + 1024 * y;
            const int64_t idx = static_cast<int64_t>(z) * pitch2 + x + y * plane_stride;
            if (wide) {
              uint16_t value;
              std::memcpy(&value, src + 2 * idx, sizeof(value));
              store_pair(dst, value);
            } else {
              _G.IF_L1[dst] = src[idx];
            }
          }
        }
      }
    }
  } else {
    // Strided layout: low16 = number of groups, high16 = step between consecutive z lines.
    const int count = static_cast<int>(l.layout_ & 0xFFFF);
    const int step = static_cast<int>(l.layout_ >> 16);
    dump_lines = count + step * (n_z - 1);
    if (count != 0 && n_y > 0 && n_z > 0) {
      const int y_stride = pitch2 * pitch1;
      const int z_stride = pitch2 * step;
      for (int a = 0; a < count; ++a) {
        for (int y = 0; y < n_y; ++y) {
          int dst = (a * n_y + y) << 10;
          int64_t idx = a * pitch2 + y * y_stride;
          for (int z = 0; z < n_z; ++z) {
            if (!wide) {
              std::memcpy(&_G.IF_L1[dst], src + idx, static_cast<size_t>(n_x));
            } else {
              for (int t = 0; t < n_x; ++t) {
                uint16_t value;
                std::memcpy(&value, src + 2 * (idx + t), sizeof(value));
                store_pair(dst + t, value);
              }
            }
            idx += z_stride;
            dst += n_x;
          }
        }
      }
    }
  }

  if (_G.debug_flag) {
    // One hex byte per line.
    std::ofstream dump(debug_file.c_str(), std::ios::out);
    for (int line = 0; line < dump_lines; ++line) {
      for (int x = line; x != n_x + line; ++x) {
        int idx = x;
        for (int y = 0; y < n_y; ++y) {
          dump << std::hex << std::setw(2) << std::setfill('0') << static_cast<uint64_t>(_G.IF_L1[idx]) << std::endl;
          idx += n_x;
        }
      }
    }
    _G.debug_flag = 0;
    dump.close();
  }
}
