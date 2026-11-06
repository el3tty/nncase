// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#pragma once
// output (TCU1..TCU4.cpp). Conv2D::Compute fills a TCU::ComputeInfo and calls TCU::ComputeConv,
// which loads the weights into the PEs (FillWeight) and computes every output pixel directly from the input
// feature map, accumulating into the PSUM_L1 tensor.
#include <cstddef>
#include <cstdint>
#include <vector>

struct TCU {
    static constexpr int kRows = 24;   // PE rows    = input channels
    static constexpr int kCols = 32;   // PE columns = output channels (lanes)
    static constexpr int kTaps = 8;    // weights per PE and buffer (kernel width, <= 8)

    // L1 tensor view (the layout of L1Helper, see act0.cpp L1View). 40 bytes.
    struct L1Tensor {
        uint8_t* data_ = nullptr;
        int32_t  max_channels_ = 0; // channel capacity of the buffer: 24 for the IF buffer (PE rows), 32 for _G.PSUM_L1 (PE lanes)
        int32_t  chan_stride_ = 0;   // bytes per channel plane (IF: 1024, PSUM: 4096)
        int32_t  base_ = 0;          // byte offset of element (0,0,0)
        int32_t  row_stride_ = 0;    // per row index
        int32_t  col_stride_ = 0;    // per column index
        int32_t  channels_ = 0;
        int32_t  height_ = 0;
        int32_t  width_ = 0;
    };

    // Everything one convolution pass needs. Built by Conv2D::Compute (224 bytes on the original stack).
    struct ComputeInfo {
        L1Tensor ifmap_;                 // input feature tensor (Conv2D's _G.if_l1_buffer); height/width are the unpadded extents
        uint8_t  wide_rows_ = 0;         // weight block height: 0 -> 24 rows, otherwise 32
        const uint8_t* weights_ = nullptr; // DmLoadW::weights_
        int32_t  kh_ = 0;                // kernel rows    (PuCompute::kernel_h_)
        int32_t  kw_ = 0;                // kernel columns (PuCompute::kernel_w_), <= kTaps
        int32_t  weight_line_bytes_ = 0; // weight bytes per output channel (DmLoadW::line_bytes_)
        L1Tensor psum_;                  // output tensor (_G.PSUM_L1); height/width = number of output rows/columns
        int32_t  stride_x_ = 0;          // window step along x   (1 when PuCompute::param_148_ != 0, else PuCompute::fetch_imm17_)
        int32_t  stride_y_ = 0;          // window step along y   (1, or PuCompute::fetch_imm22_)
        int32_t  pad_bottom_ = 0;        // PuCompute::pad_bottom (not read by the TCU functions; role names inferred)
        int32_t  pad_top_ = 0;
        int32_t  pad_left_ = 0;
        int32_t  pad_right_ = 0;         // (not read by the TCU functions; role names inferred)
        uint8_t  pad_value_ = 0;         // value fed for out-of-range IF positions
        alignas(4) int32_t groups_ = 0;  // number of channel groups (PuCompute::groups_)
        uint8_t  weights_signed_ = 0;    // 1: weights are int8, 0: uint8 minus per-channel zero point
        alignas(8) uint8_t owns_zero_points_ = 0;  // always 0 (Conv2D::Compute deletes the table when set)
        const uint8_t* weight_zero_points_ = nullptr; // DmLoadW::zero_points_
        int32_t  weight_dim_[4] = {};    // {1, 1, 1, PuCompute::of2_d2_}; weight_dim[3] bounds the column index
        int32_t  psum_channels_ = 0;              // PuCompute::of2_d2 (unused by the TCU)
        alignas(8) uint8_t if_signed_ = 0; // 1: IF bytes are int8, 0: uint8 minus per-row zero point
        const uint8_t* if_zero_points_ = nullptr; // PuCompute::if_zero_points_
        int32_t  shift_mode_ = 0;        // PSUM scaling: 0 none, 1 << 4, otherwise >> 4
        uint8_t  accumulate_ = 0;        // add to the PSUM already present (set after the first kernel row)
        int32_t  mode_ = 0;              // PuCompute::mode (only used by Conv2D::Compute)
    };

    // PE weights, double buffered: weight_[buffer][PE row][tap][PE column]. The lane (column) index is innermost so the
    // PE columns of one row are contiguous.
    int32_t weight_[2][kRows][kTaps][kCols] = {};
    int32_t weight_sel_ = 0;             // weight buffer FillWeight writes next
    int32_t weight_prev_ = 0;            // buffer written by the previous FillWeight (read by ComputeConv)
    int32_t rows_ = 0;                   // active PE rows    (ifmap.channels)
    int32_t cols_ = 0;                   // active PE columns (psum.channels)
    int32_t rows_per_group_ = 0;         // rows / groups
    int32_t cols_per_group_ = 0;         // cols / groups

    void FillWeight(ComputeInfo& info, int kernel_row);
    void LoadIfRow(const ComputeInfo& info, int y, int x, int count, int32_t* line);
    void ComputeConv(ComputeInfo& info);
};

