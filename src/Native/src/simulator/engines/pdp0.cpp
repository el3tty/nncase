// PDP0 pooling / depthwise-convolution engine, lifted from IDA/Hex-Rays output (nncase K230 C-model).
// Source numbering follows work/clean/PDP0<N>.cpp ("Source N").
#include "engines/pdp0.h"
#include "globals.h"
#include "engines/act0.h"
#include "engines/checkpoint.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// Partial-sum L1 buffer read by the pooling / depthwise engines (declared in pu_common.h as well).

namespace {

// ---------------------------------------------------------------------------------------------
// Raw views of helper types that are not lifted yet (L1Helper, WeightsHelper, Matrix4<uint8_t>,
// DmLoadW). TODO(layout): replace by the real types once their owners define them.
// ---------------------------------------------------------------------------------------------

// L1Helper: a (channel, row, col) view into a byte buffer. Strides are in bytes.
struct L1View {
    uint8_t* data_;        // +0
    int32_t  max_channels_; // +8   channel capacity of the buffer (24 for IF = PE rows, 32 for PSUM = PE lanes)
    int32_t  chan_stride_; // +12
    int32_t  base_;        // +16  byte offset of element (0,0,0)
    int32_t  row_stride_;  // +20
    int32_t  col_stride_;  // +24
    int32_t  channels_;    // +28
    int32_t  height_;      // +32
    int32_t  width_;       // +36
};

// WeightsHelper as consumed by ComputeDW.
struct WeightsView {
    uint8_t  type_;        // +0   0: 24-byte channel blocks, otherwise 32-byte blocks
    uint8_t* data_;        // +8
    int32_t  kernel_h_;    // +16
    int32_t  kernel_w_;    // +20
    int32_t  line_bytes_;  // +24  bytes per weight line (DmLoadW::line_bytes_; not read by ComputeDW)
};

// Matrix4<uint8_t>: per-channel weight zero points (only the first channel row is used).
struct MatrixView {
    uint8_t  owns_;        // +0
    uint8_t* data_;        // +8
    int32_t  dim0_, dim1_, dim2_, dim3_;  // +16 .. +28
    int32_t  count_;        // +32  (= dim3)
};

// DmLoadW descriptor as consumed by Compute() (created by DmLoadWInstruction).
struct DmLoadWView {
    int32_t  line_bytes_;  // +0
    int32_t  signed_flag_; // +4   ==1: weights are unsigned
    uint8_t* weights_;     // +8
    uint8_t* zero_points_; // +16
};

// Trace channels of the CheckPoint singleton used by the PDP0 engine.
std::ofstream& CkptStream(CheckPoint::StreamId id) { return CheckPoint::GetCheckPoint()->Stream(id); }
constexpr CheckPoint::StreamId kCkptPdp0In  = CheckPoint::kPdp0Unk19;   // +9728:  input pixels
constexpr CheckPoint::StreamId kCkptPdp0W   = CheckPoint::kPdp0PeWeIn;  // +10240: weights
constexpr CheckPoint::StreamId kCkptPdp0Out = CheckPoint::kPdp0PeOut;   // +10752: products / pooled values

}  // namespace

// PDP01.cpp  @0x42b5d0 (Source 1)
// The original constructs the singleton in place (five empty deques, one default shared_ptr) under a
// guard variable and registers PDP0::~PDP0 with atexit; a function-local static is equivalent.
PDP0* PDP0::GetPDP0()
{
  static PDP0 pdp0;
  return &pdp0;
}

void PDP0::Reset() {
  std::memset(&regs_, 0, sizeof(regs_));
  weight_queue_.clear();
  store_of_queue_.clear();
  load_act0_queue_.clear();
  act0_queue_.clear();
  cur_act0_.reset();
  compute_queue_.clear();
}

// PDP02.cpp  @0x42bd40 (Source 2): PDP0::~PDP0() = default (header). The decompiled body only
// destroys the five std::deque<std::shared_ptr<...>> members and cur_act0 (shared_ptr refcount code).

// PDP03.cpp  @0x46b6b0 (Source 3)
void PDP0::ComputePDP0(L1Helper& in_l1, L1Helper& out_l1, bool is_signed, std::shared_ptr<Pdp0Compute> cfg)
{
  const L1View& in = reinterpret_cast<const L1View&>(in_l1);
  L1View& out = reinterpret_cast<L1View&>(out_l1);
  const Pdp0Compute& c = *cfg;

  // verified against asm @0x46b6b0 (pooling: mode 1 = running minimum, mode 2 = running maximum).
  for (int oy = 0; oy < out.height_; ++oy) {
    for (int ox = 0; ox < out.width_; ++ox) {
      // Initial value of the 32-lane accumulator: -256 for mode 2, +256 otherwise (broadcast into a 0x80-byte buffer).
      const int init_value = (c.mode_ == 2) ? -256 : 256;
      std::vector<int32_t> pooled(32, init_value);   // per-channel result (r15 in the asm)

      const int x0 = c.stride_x_ * ox - c.pad_left_;
      const int y0 = c.stride_y_ * oy - c.pad_top_;
      if (static_cast<int>(c.kernel_h_) > 0 && static_cast<int>(c.kernel_w_) > 0) {
        for (int ky = 0; ky < static_cast<int>(c.kernel_h_); ++ky) {
          const int y = y0 + ky;
          for (int kx = 0; kx < static_cast<int>(c.kernel_w_); ++kx) {
            const int x = x0 + kx;
            std::vector<int> window_vals(32, 0);  // traced to the "in"  checkpoint (width 3)
            std::vector<int> window_acc(32, 0);   // traced to the "out" checkpoint (width 5)
            for (int ch = 0; ch < out.channels_; ++ch) {
              // Out-of-bounds taps use pad_value (c+0x1c).
              int sample = static_cast<int>(c.pad_value_);
              if (x >= 0 && x < in.width_ && y >= 0 && y < in.height_) {
                const uint8_t raw = in.data_[ch * in.chan_stride_ + x * in.col_stride_ + in.base_ + y * in.row_stride_];
                if (is_signed)
                  sample = static_cast<int8_t>(raw);
                else
                  sample = static_cast<int>(raw) - static_cast<int>(c.in_zero_point_);   // vpsubd with c+0x30
              }
              window_vals[ch] = sample;
              if (c.mode_ == 1)
                pooled[ch] = std::min(sample, pooled[ch]);        // vpminsd
              else if (c.mode_ == 2)
                pooled[ch] = std::max(sample, pooled[ch]);        // vpmaxsd
              window_acc[ch] = pooled[ch];
            }
            CheckPoint::PrintCheckPoint(CkptStream(kCkptPdp0In), window_vals, 3, true);
            CheckPoint::PrintCheckPoint(CkptStream(kCkptPdp0Out), window_acc, 5, true);
          }
        }
      }
      for (int ch = 0; ch < out.channels_; ++ch) {
        const int idx = oy * out.row_stride_ + static_cast<int>(static_cast<uint32_t>(out.base_) >> 2) +
                        ox * out.col_stride_ + static_cast<int>(static_cast<uint32_t>(ch * out.chan_stride_) >> 2);
        reinterpret_cast<int32_t*>(out.data_)[idx] = pooled[ch];
      }
    }
  }
}

// PDP04.cpp  @0x46bb50 (Source 4)
void PDP0::ComputeDW(L1Helper& in_l1, WeightsHelper& weights_h, L1Helper& out_l1, bool in_signed, bool weights_signed,
                     Matrix4<uint8_t>& zero_points_m, std::shared_ptr<Pdp0Compute> cfg)
{
  const L1View& in = reinterpret_cast<const L1View&>(in_l1);
  L1View& out = reinterpret_cast<L1View&>(out_l1);
  const WeightsView& w = reinterpret_cast<const WeightsView&>(weights_h);
  const MatrixView& wzp = reinterpret_cast<const MatrixView&>(zero_points_m);
  const Pdp0Compute& c = *cfg;
  const int block = (w.type_ == 0) ? 24 : 32;  // channels per weight block

  for (int oy = 0; oy < out.height_; ++oy) {
    for (int ox = 0; ox < out.width_; ++ox) {
      const int x0 = c.stride_x_ * ox - c.pad_left_;
      const int y0 = c.stride_y_ * oy - c.pad_top_;
      std::vector<int32_t> acc(out.channels_, 0);  // per-channel accumulators

      for (int ky = 0; ky < static_cast<int>(c.kernel_h_); ++ky) {
        const int y = y0 + ky;
        for (int kx = 0; kx < static_cast<int>(c.kernel_w_); ++kx) {
          const int x = x0 + kx;
          std::vector<int> pixels(32, 0);    // traced to the 9728 checkpoint (width 3)
          std::vector<int> weights(32, 0);   // traced to the 10240 checkpoint (width 3)
          std::vector<int> products(32, 0);  // traced to the 10752 checkpoint (width 5)
          for (int ch = 0; ch < out.channels_; ++ch) {
            // Per-channel weight zero point (only needed for unsigned weights).
            int wzp_value = 0;
            if (!weights_signed) {
              if (wzp.dim0_ <= 0 || wzp.dim1_ <= 0 || wzp.dim2_ <= 0 || wzp.dim3_ <= ch)
                std::cout << "[Error: Matrix Exceed]" << std::endl;  // the original keeps going and still reads wzp.data[ch]
              wzp_value = wzp.data_[ch];
            }

            // Input pixel; out-of-bounds taps read the configured padding value.
            int pixel = static_cast<int>(c.pad_value_);
            if (x >= 0 && x < in.width_ && y >= 0 && y < in.height_) {
              pixel = static_cast<int8_t>(in.data_[ch * in.chan_stride_ + x * in.col_stride_ + in.base_ + y * in.row_stride_]);
              if (!in_signed)
                pixel = static_cast<uint8_t>(pixel) - static_cast<int>(c.in_zero_point_);
            }

            // Weight for (tap, channel): blocks of `block` channels, kernel_w x kernel_h taps per block.
            const int widx = block * (kx + ky * w.kernel_w_ + w.kernel_w_ * w.kernel_h_ * (ch / block)) + ch % block;
            const int weight = weights_signed ? static_cast<int8_t>(w.data_[widx])
                                              : static_cast<uint8_t>(w.data_[widx]) - wzp_value;

            acc[ch] += pixel * weight;
            pixels[ch] = pixel;
            weights[ch] = weight;
            products[ch] = pixel * weight;
          }
          CheckPoint::PrintCheckPoint(CkptStream(kCkptPdp0In), pixels, 3, true);
          CheckPoint::PrintCheckPoint(CkptStream(kCkptPdp0W), weights, 3, true);
          CheckPoint::PrintCheckPoint(CkptStream(kCkptPdp0Out), products, 5, true);
        }
      }

      for (int ch = 0; ch < out.channels_; ++ch) {
        const int idx = oy * out.row_stride_ + static_cast<int>(static_cast<uint32_t>(out.base_) >> 2) +
                        ox * out.col_stride_ + static_cast<int>(static_cast<uint32_t>(ch * out.chan_stride_) >> 2);
        reinterpret_cast<int32_t*>(out.data_)[idx] = acc[ch];
      }
    }
  }
}

// PDP05.cpp  @0x46c2d0 (Source 5)
// The decompilation shows only stores for words 0, 1-2, 7 and 24 of the 100-byte descriptor; every
// field the consumers read (strides, padding, kernel, output shape) lives in the words in between,
// so the whole register block is copied.
// verified against asm @0x46c2d0: the stores are vectorised copies of words 0, 1-2, 3-6, 7, 8-11, 12-15, 16-19, 20-23,
// 24, i.e. the whole 100-byte register block is copied.
std::shared_ptr<Pdp0Compute> PDP0::GetPdp0Compute() const
{
  return std::make_shared<Pdp0Compute>(regs_);
}

// PDP06.cpp  @0x46c3a0 (Source 6)
void PDP0::Activate()
{
  // Pop one entry from each queue (the original copies the front shared_ptr, then pop_front()s).
  std::shared_ptr<Act0Compute> act0 = act0_queue_.front();
  act0_queue_.pop_front();
  std::shared_ptr<DmLoadAct0> load = load_act0_queue_.front();
  load_act0_queue_.pop_front();
  std::shared_ptr<DmStoreOf> store = store_of_queue_.front();
  store_of_queue_.pop_front();
  Act0::Compute(act0, load, store);
}

// PDP07.cpp  @0x46ce50 (Source 7)
void PDP0::Compute()
{
  // Oldest queued configuration snapshot.
  std::shared_ptr<Pdp0Compute> cfg = compute_queue_.front();
  compute_queue_.pop_front();
  const Pdp0Compute& c = *cfg;

  // Input: the partial-sum L1 buffer, (in_h - pads) x (in_w - pads) x in_c.
  L1View input{};
  input.data_ = reinterpret_cast<uint8_t*>(_G.PSUM_L1);
  input.max_channels_ = 32;
  input.chan_stride_ = 0x1000;
  input.base_ = static_cast<int32_t>(c.psum_offset_);
  input.col_stride_ = 1;
  input.channels_ = static_cast<int32_t>(c.in_c_);
  input.height_ = static_cast<int32_t>(c.in_h_ - c.pad_top_ - c.pad_bottom_);
  input.width_ = static_cast<int32_t>(c.in_w_ - c.pad_left_ - c.pad_right_);
  input.row_stride_ = input.width_;

  // verified against asm @0x46cf5b: the output buffer is the second PSUM buffer of the Act0 singleton (Act0 @0x58c760
  // + 0x20000), zeroed with memset(.., 0, 0x20000).
  uint8_t * const psum1 = reinterpret_cast<uint8_t *>(_G.Act0_act0) + 0x20000;
  std::memset(psum1, 0, 0x20000);
  L1View output{};
  output.data_ = psum1;
  output.max_channels_ = 32;
  output.chan_stride_ = 0x1000;
  output.base_ = 0;
  output.col_stride_ = 1;
  output.channels_ = static_cast<int32_t>(c.out_c_);
  output.height_ = static_cast<int32_t>(c.out_h_);
  output.width_ = static_cast<int32_t>(c.out_w_);
  output.row_stride_ = output.width_;

  const bool is_signed = c.unsigned_flag_ != 1;
  if (c.mode_ == 0) {
    // Depthwise convolution: consume the oldest weight-load descriptor.
    std::shared_ptr<DmLoadW> load_w = weight_queue_.front();
    weight_queue_.pop_front();
    const DmLoadWView& lw = *reinterpret_cast<const DmLoadWView*>(load_w.get());  // TODO(layout)

    WeightsView weights{};
    weights.type_ = 1;
    weights.data_ = lw.weights_;
    weights.kernel_h_ = static_cast<int32_t>(c.kernel_h_);
    weights.kernel_w_ = static_cast<int32_t>(c.kernel_w_);
    weights.line_bytes_ = lw.line_bytes_;

    MatrixView zero_points{};
    zero_points.owns_ = 0;
    zero_points.data_ = lw.zero_points_;
    zero_points.dim0_ = zero_points.dim1_ = zero_points.dim2_ = 1;
    zero_points.dim3_ = zero_points.count_ = static_cast<int32_t>(c.out_c_);

    ComputeDW(reinterpret_cast<L1Helper&>(input), reinterpret_cast<WeightsHelper&>(weights),
              reinterpret_cast<L1Helper&>(output), is_signed, lw.signed_flag_ != 1,
              reinterpret_cast<Matrix4<uint8_t>&>(zero_points), cfg);
  } else {
    ComputePDP0(reinterpret_cast<L1Helper&>(input), reinterpret_cast<L1Helper&>(output), is_signed, cfg);
  }

  Activate();
}
