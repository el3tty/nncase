// Lifted from IDA/Hex-Rays output (Conv2D1..Conv2D5.cpp). The AVX instructions that the auto-converted
// results/ copy had stripped (16/32-byte block copies of the configuration, the IF/PSUM tensor setup) were
// recovered from sources/Conv2D*.cpp. All inlined libstdc++ code (shared_ptr reference counting, std::deque
// pop_front / node release, make_shared) is expressed with the standard containers.
#include "engines/conv2d.h"
#include "globals.h"
#include "engines/act0.h"
#include "engines/dm.h"
#include "engines/tcu.h"
#include <cstring>
#include <memory>

// TODO(globals): not declared in globals.h (defined elsewhere in the original binary).

namespace {


}  // namespace

Conv2D::Conv2D()
{
    // @0x468f10 (Source 4): the only non-zero initial value; the rest of the object is zero / empty queues.
    cfg_.if_flag_ = 1;
}

// @0x469500 (Source 5): destroys the six queues (the 1700 decompiled lines are the inlined deque / shared_ptr releases).
Conv2D::~Conv2D() = default;

// @0x467450 (Source 1)
std::shared_ptr<PuCompute> Conv2D::GetPuCompute() const
{
    // The original allocates a shared control block and copies the configuration bytes [0, 158) with 16-byte moves.
    return std::make_shared<PuCompute>(cfg_);
}

// Legacy shim: the generated header declared `Conv2D* GetPuCompute(int64_t*)` because the sret slot (a
// shared_ptr<PuCompute> in the caller) arrives in the 'this' register. See conv2d.h.
Conv2D* Conv2D::GetPuCompute(int64_t* conv)
{
    ::new (static_cast<void*>(this)) std::shared_ptr<PuCompute>(reinterpret_cast<const Conv2D*>(conv)->GetPuCompute());
    return this;
}

// @0x467590 (Source 2)
// Pops the next ACT0 job: Act0Compute descriptor + activation parameters (+ the DM store when the result goes to
// the DM).
void Conv2D::Activate()
{
    std::shared_ptr<Act0Compute> compute = act0_queue_.front();
    act0_queue_.pop_front();
    std::shared_ptr<DmLoadAct0> load = act0_param_queue_.front();
    act0_param_queue_.pop_front();

    std::shared_ptr<DmStoreOf> store;
    if (compute->out_route_ != 0) {   // +28: 0 = _G.PSUM_L1 only, no DM store involved
        store = store_queue_.front();
        store_queue_.pop_front();
    }
    Act0::Compute(compute, load, store);
}

// @0x4681e0 (Source 3)
void Conv2D::Compute()
{
    std::shared_ptr<DmLoadW> load_w = weight_queue_.front();
    weight_queue_.pop_front();
    std::shared_ptr<PuCompute> pu_job = compute_queue_.front();
    compute_queue_.pop_front();
    const PuCompute& pu = *pu_job;

    if (pu.clear_psum_)
        std::memset(_G.PSUM_L1, 0, 0x20000);

    // Unpadded extent of the IF tile (the padded size minus both borders).
    const int if_height = static_cast<int>(pu.in_height_) - static_cast<int>(pu.pad_top_) - static_cast<int>(pu.pad_bottom_);
    const int if_width = static_cast<int>(pu.in_width_) - static_cast<int>(pu.pad_left_) - static_cast<int>(pu.pad_right_);
    const int if_area = if_width * if_height;

    // Fetch the next IF tile (24 KiB) from the DmLoadL1 queue.
    if (pu.if_flag_ && if_area > 0) {
        std::shared_ptr<DmLoadL1> tile = if_queue_.front();
        if_queue_.pop_front();
        std::memcpy(_G.if_l1_buffer, tile->if_snapshot_, sizeof(_G.if_l1_buffer));
    }
    if (if_area > 0 || pu.flag_142_)
        cfg_.if_flag_ = pu.flag_142_;   // sticky copy: tells the next job whether to fetch an IF tile

    // ---- build the TCU job (TCU::ComputeInfo, 224 bytes on the original stack) ----
    TCU::ComputeInfo info;
    info.ifmap_.data_ = _G.if_l1_buffer;
    info.ifmap_.max_channels_ = 24;
    info.ifmap_.chan_stride_ = 1024;
    info.ifmap_.base_ = static_cast<int32_t>(pu.if_base_);
    // verified against asm @0x4683d7: row_stride = pu+0x10 (if_shape_d0, vpinsrd $1 into the base/row_stride qword).
    info.ifmap_.row_stride_ = static_cast<int32_t>(pu.if_shape_d0_);
    info.ifmap_.col_stride_ = 1;
    info.ifmap_.channels_ = static_cast<int32_t>(pu.in_channels_);
    info.ifmap_.height_ = if_height;
    info.ifmap_.width_ = if_width;
    info.wide_rows_ = 0;
    info.weights_ = load_w->weights_;
    info.kh_ = static_cast<int32_t>(pu.kernel_h_);
    info.kw_ = static_cast<int32_t>(pu.kernel_w_);
    info.weight_line_bytes_ = static_cast<int32_t>(load_w->line_bytes_);
    info.psum_.data_ = reinterpret_cast<uint8_t*>(_G.PSUM_L1);
    info.psum_.max_channels_ = 32;
    info.psum_.chan_stride_ = 4096;
    // verified against asm @0x46858c / @0x4686ab: the PSUM tensor and the window steps are filled from the snapshot
    // BEFORE TCU::ComputeConv (they are overwritten again after it for mode 1).
    info.psum_.base_ = static_cast<int32_t>(pu.psum_base_);          // +0x88
    info.psum_.row_stride_ = static_cast<int32_t>(pu.of1_d0_);       // +0x70
    info.psum_.channels_ = static_cast<int32_t>(pu.of2_d2_);         // +0x7c
    info.psum_.height_ = static_cast<int32_t>(pu.of2_d1_);           // +0x80
    info.psum_.width_ = static_cast<int32_t>(pu.of2_d0_);            // +0x84
    if (pu.param_148_ == 0) {
        info.psum_.col_stride_ = 1;
        info.stride_x_ = static_cast<int32_t>(pu.fetch_imm17_);     // qword at pu+0: (imm17, imm22)
        info.stride_y_ = static_cast<int32_t>(pu.fetch_imm22_);
    } else {
        info.psum_.col_stride_ = static_cast<int32_t>(pu.fetch_imm17_);
        info.stride_x_ = 1;                                        // .rodata 0x522270 = 0x0000000100000001
        info.stride_y_ = 1;
    }
    info.pad_bottom_ = static_cast<int32_t>(pu.pad_bottom_);
    info.pad_top_ = static_cast<int32_t>(pu.pad_top_);
    info.pad_left_ = static_cast<int32_t>(pu.pad_left_);
    info.pad_right_ = static_cast<int32_t>(pu.pad_right_);
    info.pad_value_ = pu.pad_value_;
    info.groups_ = static_cast<int32_t>(pu.groups_);
    info.weights_signed_ = load_w->deq_mode_ != 1;
    info.owns_zero_points_ = 0;
    info.weight_zero_points_ = load_w->zero_points_;
    info.weight_dim_[0] = 1;
    info.weight_dim_[1] = 1;
    info.weight_dim_[2] = 1;
    info.weight_dim_[3] = static_cast<int32_t>(pu.of2_d2_);
    info.psum_channels_ = static_cast<int32_t>(pu.of2_d2_);
    // IF bytes are int8 unless the deq mode asks for zero-point subtraction.
    bool if_signed = false;
    if (pu.if_deq_mode_ != 1) {
        if_signed = true;
        if (pu.if_deq_mode_ == 3)
            if_signed = pu.shift_mode_ != 2;
    }
    info.if_signed_ = if_signed;
    info.if_zero_points_ = pu.if_zero_points_;
    info.shift_mode_ = static_cast<int32_t>(pu.shift_mode_);
    info.accumulate_ = pu.accumulate_;
    info.mode_ = static_cast<int32_t>(pu.mode_);

    // The 24 x 32 PE array is ~72 KiB: heap instead of the original stack frame.
    auto tcu = std::make_unique<TCU>();
    tcu->ComputeConv(info);

    if (pu.mode_ == 1) {
        // Hand the PSUM to ACT0: copy one 4 KiB plane per output channel into the Act0 PSUM buffer, then Activate().
        std::shared_ptr<Act0Compute> act = act0_queue_.front();   // peeked, popped by Activate()
        const int channels = static_cast<int>(act->channels_);    // +4
        const int height = static_cast<int>(act->height_);        // +8
        const int width = static_cast<int>(act->width_);          // +12
        const int base = static_cast<int32_t>(cfg_.psum_base_);    // live Conv2D+136, not the snapshot
        info.psum_.col_stride_ = 1;
        info.psum_.channels_ = channels;
        info.psum_.height_ = height;
        info.psum_.width_ = width;
        info.psum_.row_stride_ = width;
        info.psum_.base_ = base;

        uint8_t* dst = reinterpret_cast<uint8_t*>(act->psum_);    // +40
        const size_t bytes = 4 * static_cast<size_t>(height * width);
        for (int ch = 0; ch < channels; ++ch) {
            const int32_t word = (base >> 2) + static_cast<int32_t>(static_cast<uint64_t>(info.psum_.chan_stride_ * ch) >> 2);
            std::memcpy(dst + (static_cast<size_t>(ch) << 12),
                        reinterpret_cast<const uint8_t*>(_G.PSUM_L1) + 4 * static_cast<int64_t>(word), bytes);
        }
        Activate();
    }
    // The original also deletes info.weight_zero_points (1 byte) when owns_zero_points is set; it never is.
}

// @0x468f10 (Source 4)
Conv2D* Conv2D::GetConv2D()
{
    static Conv2D conv2d;   // guarded lazy init; the original registers ~Conv2D with __cxa_atexit
    return &conv2d;
}
