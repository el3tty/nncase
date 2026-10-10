// Lifted from IDA/Hex-Rays output (TCU1..TCU4.cpp). The AVX instructions that the auto-converted
// results/ copy had stripped were recovered from sources/TCU*.cpp (they only initialise the PE
// bookkeeping words and compute the minimum of two ints).
//
// Dataflow of one convolution (TCU::ComputeConv):
//   for each kernel row ky:            FillWeight(ky) loads weight[ky][0..kw) into every PE (double buffered)
//     for each output row oy:          the per-PE input queues are cleared, then
//       for each output column ox:     FillIf() pushes the new IF samples of the sliding window into the PE queues,
//                                      PECalculate() multiplies the kw queued samples with the kw weights, sums
//                                      over the PE rows (input channels) and writes/accumulates the 32 lanes into PSUM_L1.
#include "engines/tcu.h"
#include "globals.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// TODO(globals): not declared in globals.h (defined elsewhere in the original binary).
extern std::string debug_file;   // dump file path

namespace {

void report_matrix_exceed()
{
    std::cout << "[Error: Matrix Exceed]" << std::endl;
}

}  // namespace

// @0x46d810 (Source 1)
// Loads the weights of kernel row `kernel_row` into the inactive weight buffer of every PE and swaps buffers.
void TCU::FillWeight(ComputeInfo& info, int kernel_row)
{
    const int block_rows = info.wide_rows_ ? 32 : 24;   // rows of one weight block

    for (int kx = 0; kx < info.kw_; ++kx) {
        for (int group_col = 0; group_col < cols_; group_col += cols_per_group_) {
            const int first_row = group_col / cols_per_group_ * rows_per_group_;
            for (int col = group_col; col < group_col + cols_per_group_; ++col) {
                // The original also filled a 96-byte scratch buffer with the values here; it was never read.
                for (int r = 0; r < rows_per_group_; ++r) {
                    int value;
                    if (!info.weights_signed_) {
                        // weights are uint8 with a per-output-channel zero point
                        if (info.weight_dim_[0] <= 0 || info.weight_dim_[1] <= 0 || info.weight_dim_[2] <= 0 ||
                            col < 0 || col >= info.weight_dim_[3])
                            report_matrix_exceed();
                    }
                    const int index = block_rows * info.kw_ * info.kh_ * (r / block_rows) +
                                      info.weight_line_bytes_ * col +
                                      (kx + info.kw_ * kernel_row) * block_rows + r % block_rows;
                    if (info.weights_signed_)
                        value = static_cast<int8_t>(info.weights_[index]);
                    else
                        value = static_cast<int>(info.weights_[index]) - static_cast<int>(info.weight_zero_points_[col]);
                    // verified against asm: the bogus self-call of the decompilation is the compiler's duplicated
                    // (.cold / signed-case) copy of this same loop, lifted as one.
                    pe_[first_row + r][col].weight_[weight_sel_][kx] = value;
                }
            }
        }
    }
    weight_prev_ = weight_sel_;
    weight_sel_ = (weight_sel_ + 1) % 2;
}

// @0x46dbe0 (Source 2)
// One output pixel: for every PE column, sum over kernel taps and PE rows of weight * queued IF sample.
void TCU::PECalculate(ComputeInfo& info, int out_y, int out_x, bool accumulate, bool last_kernel_row)
{
    (void)last_kernel_row;   // passed by ComputeConv but never read by the decompiled body

    const int taps = pe_[0][0].taps_;
    std::vector<std::vector<int32_t>> tap_sum(taps, std::vector<int32_t>(kCols, 0));   // [tap][lane]

    for (int tap = 0; tap < taps; ++tap) {
        // The original also stored pe[row][0].ifmap[tap] and pe[row][0].weight[prev][tap] of every row into two
        // scratch arrays that were never read; dropped.
        for (int row0 = 0; row0 < rows_; row0 += rows_per_group_) {
            int col = cols_per_group_ * (row0 / rows_per_group_);
            for (int c = 0; c < cols_per_group_; ++c, ++col) {
                for (int r = 0; r < rows_per_group_; ++r) {
                    PE& p = pe_[row0 + r][col];
                    // NOTE: reads the raw queue storage; the queue holds >= taps samples in normal operation.
                    const int32_t product = p.weight_[weight_prev_][tap] * p.ifmap_.data()[tap];
                    if (tap == taps - 1) {
                        // Slide the window: drop min(pop_count, taps) samples from the front.
                        const int pops = std::min(p.pop_count_, p.taps_);
                        for (int i = 0; i < pops; ++i)
                            if (!p.ifmap_.empty())
                                p.ifmap_.erase(p.ifmap_.begin());
                    }
                    tap_sum[tap][col] += product;
                }
            }
        }
    }

    int32_t* psum = reinterpret_cast<int32_t*>(info.psum_.data_);
    const L1Tensor& t = info.psum_;
    auto lane_ptr = [&](int lane) {
        const int64_t off = static_cast<int64_t>(static_cast<int64_t>(t.chan_stride_) * lane) / 4 +   // (uint64)(stride * lane) >> 2
                            (static_cast<int64_t>(t.base_) >> 2) +
                            static_cast<int64_t>(t.col_stride_) * out_x + static_cast<int64_t>(t.row_stride_) * out_y;
        return psum + static_cast<int32_t>(off);
    };

    for (int lane = 0; lane < kCols; ++lane) {
        int32_t sum = 0;
        for (int tap = 0; tap < taps; ++tap)
            sum += tap_sum[tap][lane];
        if (info.shift_mode_ == 1)
            sum *= 16;
        else if (info.shift_mode_ != 0)
            sum >>= 4;

        int32_t* dst = lane_ptr(lane);
        if (accumulate) {
            int64_t total = static_cast<int64_t>(*dst) + sum;
            if (total >= 0x80000000LL) {
                std::cout << "max int: " << total << std::endl;
                total = 0x7FFFFFFF;
            } else if (total <= -0x80000001LL) {
                std::cout << "min int: " << total << std::endl;
                total = static_cast<int32_t>(0x80000000u);
            }
            *dst = static_cast<int32_t>(total);
        } else {
            *dst = sum;
        }
    }
}

// @0x46e5b0 (Source 3)
// Pushes the IF samples at (y, x .. x+count-1) into the input queue of every PE (all columns of a group get the
// same sample). Positions outside the tensor feed info.pad_value.
void TCU::FillIf(ComputeInfo& info, int y, int x, int count)
{
    for (int xi = x; xi < x + count; ++xi) {
        const bool inside = y >= 0 && xi >= 0 && y < info.ifmap_.height_ && xi < info.ifmap_.width_;
        const int32_t pad = info.pad_value_;
        for (int row0 = 0; row0 < rows_; row0 += rows_per_group_) {
            const int first_col = cols_per_group_ * (row0 / rows_per_group_);
            for (int r = 0; r < rows_per_group_; ++r) {
                const int row = row0 + r;
                int32_t value = pad;
                if (inside) {
                    const L1Tensor& t = info.ifmap_;
                    int v = static_cast<int8_t>(t.data_[t.base_ + t.row_stride_ * y + t.col_stride_ * xi + t.chan_stride_ * row]);
                    if (!info.if_signed_)
                        v = static_cast<uint8_t>(v) - static_cast<int>(info.if_zero_points_[row]);
                    value = v;
                }
                for (int c = 0; c < cols_per_group_; ++c)
                    pe_[row][first_col + c].ifmap_.push_back(value);
            }
        }
    }
    // verified against asm: JUMPOUT(0x40ABDE) / JUMPOUT(0x40AC3B) land in the .cold copy of the inside-tensor loop body
    // (movsbl (%r13,%rdi) ... = the same IF load as above); nothing is pushed when cols_per_group <= 0.
}

// @0x46e950 (Source 4)
void TCU::ComputeConv(ComputeInfo& info)
{
    rows_ = info.ifmap_.channels_;
    cols_ = info.psum_.channels_;
    cols_per_group_ = cols_ / info.groups_;
    rows_per_group_ = rows_ / info.groups_;

    if (rows_ > 0 && cols_ > 0) {
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c) {
                pe_[r][c].pop_count_ = info.stride_x_;
                // verified against asm @0x46e9cc: pop_count = info+0x70 (stride_x), taps = r11d = info+0x3c (kw).
                pe_[r][c].taps_ = info.kw_;
            }
    }

    if (info.kh_ <= 0)
        return;

    FillWeight(info, 0);
    int ky = 0;
    int y_first = -info.pad_top_;
    while (true) {
        const bool last_ky = (info.kh_ - 1 == ky);
        int y = y_first;
        for (int oy = 0; oy < info.psum_.height_; ++oy) {
            // Reset the sliding window of every PE.
            for (int r = 0; r < rows_; ++r)
                for (int c = 0; c < cols_; ++c)
                    pe_[r][c].ifmap_.clear();

            const int x_first = -info.pad_left_;
            if (info.kw_ - info.stride_x_ > 0)
                FillIf(info, y, x_first, info.kw_ - info.stride_x_);   // prime the window with kw - stride samples
            const int fresh = std::min(info.kw_, info.stride_x_);       // samples added per output pixel

            int x = x_first;
            for (int ox = 0; ox < info.psum_.width_; ++ox) {
                FillIf(info, y, x + info.kw_ - fresh, fresh);
                PECalculate(info, oy, ox, info.accumulate_ != 0, last_ky);
                x += info.stride_x_;
            }
            y += info.stride_y_;
        }

        if (_G.debug_flag && _G.debug_tcu_sel == 1 && _G.debug_dmw_h == ky) {
            // Dump the weight buffer the last FillWeight wrote (hex, 8 digits, one word per line).
            std::ofstream dump(debug_file.c_str(), std::ios::app);
            for (int r = 0; r < kRows; ++r)
                for (int c = 0; c < kCols; ++c) {
                    if (info.kw_ <= 0)
                        break;
                    for (int j = 0; j < info.kw_; ++j)
                        dump << std::setfill('0') << std::setw(8) << std::hex
                             << static_cast<unsigned int>(pe_[r][c].weight_[weight_prev_][j]) << std::endl;
                }
            _G.debug_flag = 0;
            dump.close();
        }

        ++ky;
        if (info.kh_ <= ky)
            break;
        FillWeight(info, ky);
        info.accumulate_ = 1;   // all further kernel rows accumulate into the PSUM
        y_first = ky - info.pad_top_;
    }
}
