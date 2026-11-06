// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

// TCU dataflow of one convolution (TCU::ComputeConv):
//   for each kernel row ky:            FillWeight(ky) loads weight[ky][0..kw) into every PE (double buffered)
//     for each output row oy:          LoadIfRow fetches the IF samples the row's windows cover (pad value outside), then
//       for each output column ox:     the kw samples at x = ox * stride_x + tap - pad_left are multiplied with the kw
//                                      weights, summed over the PE rows (input channels) and written/accumulated into
//                                      the 32 lanes of PSUM_L1.
// The hardware (and the original simulator) streams the samples through a sliding-window queue in every PE; the
// window always holds the samples at those positions, so they are indexed directly here.
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

void report_matrix_exceed() {
    std::cout << "[Error: Matrix Exceed]" << std::endl;
}

}  // namespace

// Loads the weights of kernel row `kernel_row` into the inactive weight buffer of every PE and swaps buffers.
void TCU::FillWeight(ComputeInfo& info, int kernel_row) {
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
                    // verified against asm: the self-call in the binary is the compiler's duplicated
                    // (.cold / signed-case) copy of this same loop, lifted as one.
                    weight_[weight_sel_][first_row + r][kx][col] = value;
                }
            }
        }
    }
    weight_prev_ = weight_sel_;
    weight_sel_ = (weight_sel_ + 1) % 2;
}

// Fetches the IF samples at (y, x .. x+count-1) of every PE row into line[row * count + i]. Positions outside the
// tensor give info.pad_value.
void TCU::LoadIfRow(const ComputeInfo& info, int y, int x, int count, int32_t* line) {
    const L1Tensor& t = info.ifmap_;
    const bool row_inside = y >= 0 && y < t.height_;
    for (int row = 0; row < rows_; ++row) {
        int32_t* dst = line + static_cast<size_t>(row) * count;
        for (int i = 0; i < count; ++i) {
            const int xi = x + i;
            int32_t value = info.pad_value_;
            if (row_inside && xi >= 0 && xi < t.width_) {
                int v = static_cast<int8_t>(t.data_[t.base_ + t.row_stride_ * y + t.col_stride_ * xi + t.chan_stride_ * row]);
                if (!info.if_signed_)
                    v = static_cast<uint8_t>(v) - static_cast<int>(info.if_zero_points_[row]);
                value = v;
            }
            dst[i] = value;
        }
    }
}

void TCU::ComputeConv(ComputeInfo& info) {
    rows_ = info.ifmap_.channels_;
    cols_ = info.psum_.channels_;
    cols_per_group_ = cols_ / info.groups_;
    rows_per_group_ = rows_ / info.groups_;

    if (info.kh_ <= 0)
        return;

    const int kw = info.kw_;
    // The PE queue drops min(stride_x, kw) samples per output pixel and only grows when stride_x > 0, so the window of
    // output column ox starts at x_first + ox * step.
    const int step = std::max(info.stride_x_, 0);
    const int out_w = info.psum_.width_;
    const int x_first = -info.pad_left_;
    const int span = out_w > 0 ? std::max((out_w - 1) * step + kw, 0) : 0;   // IF samples covered by one output row
    std::vector<int32_t> line(static_cast<size_t>(std::max(rows_, 0)) * span);

    int32_t* psum = reinterpret_cast<int32_t*>(info.psum_.data_);
    const L1Tensor& t = info.psum_;
    auto lane_ptr = [&](int lane, int out_y, int out_x) {
        const int64_t off = static_cast<int64_t>(static_cast<int64_t>(t.chan_stride_) * lane) / 4 +   // (uint64)(stride * lane) >> 2
                            (static_cast<int64_t>(t.base_) >> 2) +
                            static_cast<int64_t>(t.col_stride_) * out_x + static_cast<int64_t>(t.row_stride_) * out_y;
        return psum + static_cast<int32_t>(off);
    };

    // Locals, so the compiler knows the loop bounds are invariant and vectorizes the lane loop.
    const int rows = rows_;
    const int rows_per_group = rows_per_group_;
    const int cols_per_group = cols_per_group_;

    FillWeight(info, 0);
    int ky = 0;
    int y_first = -info.pad_top_;
    while (true) {
        const bool accumulate = info.accumulate_ != 0;
        const int32_t (&weights)[kRows][kTaps][kCols] = weight_[weight_prev_];
        int y = y_first;
        for (int oy = 0; oy < info.psum_.height_; ++oy) {
            if (out_w > 0)
                LoadIfRow(info, y, x_first, span, line.data());
            for (int ox = 0; ox < out_w; ++ox) {
                // For every PE column: sum over kernel taps and PE rows of weight * IF sample.
                int32_t acc[kCols] = {};
                for (int row0 = 0; row0 < rows; row0 += rows_per_group) {
                    const int col0 = cols_per_group * (row0 / rows_per_group);
                    for (int r = 0; r < rows_per_group; ++r) {
                        const int row = row0 + r;
                        const int32_t* in = line.data() + static_cast<size_t>(row) * span + static_cast<size_t>(ox) * step;
                        for (int tap = 0; tap < kw; ++tap) {
                            const int32_t v = in[tap];
                            const int32_t* w = &weights[row][tap][col0];
                            int32_t* a = acc + col0;
                            for (int c = 0; c < cols_per_group; ++c)
                                a[c] += w[c] * v;
                        }
                    }
                }

                for (int lane = 0; lane < kCols; ++lane) {
                    int32_t sum = acc[lane];
                    if (info.shift_mode_ == 1)
                        sum *= 16;
                    else if (info.shift_mode_ != 0)
                        sum >>= 4;

                    int32_t* dst = lane_ptr(lane, oy, ox);
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
                             << static_cast<unsigned int>(weight_[weight_prev_][r][j][c]) << std::endl;
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
