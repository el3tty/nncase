#pragma once
// TCU: the 24 x 32 processing-element (PE) array of the K230 NPU C-model. Lifted from IDA/Hex-Rays
// output (TCU1..TCU4.cpp). Conv2D::Compute fills a TCU::ComputeInfo and calls TCU::ComputeConv(),
// which streams the weights (FillWeight) and the input feature map (FillIf) into the PEs and
// accumulates the results into the PSUM_L1 tensor (PECalculate).
#include <cstddef>
#include <cstdint>
#include <vector>

struct TCU {
    static constexpr int kRows = 24;   // PE rows    = input channels
    static constexpr int kCols = 32;   // PE columns = output channels (lanes)
    static constexpr int kTaps = 8;    // weights per PE and buffer (kernel width, <= 8)

    // L1 tensor view (the layout of L1Helper, see act0.cpp L1View). 40 bytes.
    struct L1Tensor {
        uint8_t* data_ = nullptr;    // +0
        int32_t  max_channels_ = 0; // +8   channel capacity of the buffer: 24 for the IF buffer (PE rows), 32 for PSUM_L1 (PE lanes)
        int32_t  chan_stride_ = 0;   // +12  bytes per channel plane (IF: 1024, PSUM: 4096)
        int32_t  base_ = 0;          // +16  byte offset of element (0,0,0)
        int32_t  row_stride_ = 0;    // +20  per row index
        int32_t  col_stride_ = 0;    // +24  per column index
        int32_t  channels_ = 0;      // +28
        int32_t  height_ = 0;        // +32
        int32_t  width_ = 0;         // +36
    };

    // Everything one convolution pass needs. Built by Conv2D::Compute (224 bytes on the original stack).
    struct ComputeInfo {
        L1Tensor ifmap_;                 // +0    input feature tensor (Conv2D's if_l1_buffer); height/width are the unpadded extents
        uint8_t  wide_rows_ = 0;         // +40   weight block height: 0 -> 24 rows, otherwise 32
        const uint8_t* weights_ = nullptr; // +48 DmLoadW::weights_
        int32_t  kh_ = 0;                // +56   kernel rows    (PuCompute::kernel_h_)
        int32_t  kw_ = 0;                // +60   kernel columns (PuCompute::kernel_w_), <= kTaps
        int32_t  weight_line_bytes_ = 0; // +64   weight bytes per output channel (DmLoadW::line_bytes_)
        L1Tensor psum_;                  // +72   output tensor (PSUM_L1); height/width = number of output rows/columns
        int32_t  stride_x_ = 0;          // +112  window step along x   (1 when PuCompute::param_148_ != 0, else PuCompute::fetch_imm17_)
        int32_t  stride_y_ = 0;          // +116  window step along y   (1, or PuCompute::fetch_imm22_)
        int32_t  pad_bottom_ = 0;        // +120  PuCompute::pad_bottom (not read by the TCU functions; role names inferred)
        int32_t  pad_top_ = 0;           // +124
        int32_t  pad_left_ = 0;          // +128
        int32_t  pad_right_ = 0;         // +132  (not read by the TCU functions; role names inferred)
        uint8_t  pad_value_ = 0;         // +136  value fed for out-of-range IF positions
        alignas(4) int32_t groups_ = 0;  // +140  number of channel groups (PuCompute::groups_)
        uint8_t  weights_signed_ = 0;    // +144  1: weights are int8, 0: uint8 minus per-channel zero point
        alignas(8) uint8_t owns_zero_points_ = 0;  // +152  always 0 (Conv2D::Compute deletes the table when set)
        const uint8_t* weight_zero_points_ = nullptr; // +160 DmLoadW::zero_points_
        int32_t  weight_dim_[4] = {};    // +168  {1, 1, 1, PuCompute::of2_d2_}; weight_dim[3] bounds the column index
        int32_t  psum_channels_ = 0;              // +184  PuCompute::of2_d2 (unused by the TCU)
        alignas(8) uint8_t if_signed_ = 0; // +192  1: IF bytes are int8, 0: uint8 minus per-row zero point
        const uint8_t* if_zero_points_ = nullptr; // +200 PuCompute::if_zero_points_
        int32_t  shift_mode_ = 0;        // +208  PSUM scaling: 0 none, 1 << 4, otherwise >> 4
        uint8_t  accumulate_ = 0;        // +212  add to the PSUM already present (set after the first kernel row)
        int32_t  mode_ = 0;              // +216  PuCompute::mode (only used by Conv2D::Compute)
    };

    // One processing element: input queue plus two (double-buffered) rows of weights.
    struct PE {
        std::vector<int32_t> ifmap_;     // +0   input values waiting in this PE (sliding window)
        int32_t weight_[2][kTaps] = {};  // +24  weight[buffer][tap]
        int32_t pop_count_ = 0;          // +88  inputs removed after the last tap (the smaller of pop_count / taps)
        int32_t taps_ = 0;               // +92  number of accumulation steps per output (= kernel width)
    };

    PE      pe_[kRows][kCols];           // +0     (flat PE index = 32 * row + col)
    int32_t weight_sel_ = 0;             // +73728 weight buffer FillWeight writes next
    int32_t weight_prev_ = 0;            // +73732 buffer written by the previous FillWeight (read by PECalculate)
    int32_t rows_ = 0;                   // +73736 active PE rows    (ifmap.channels)
    int32_t cols_ = 0;                   // +73740 active PE columns (psum.channels)
    int32_t rows_per_group_ = 0;         // +73744 rows / groups
    int32_t cols_per_group_ = 0;         // +73748 cols / groups

    // @0x46d810 (Source 1)
    void FillWeight(ComputeInfo& info, int kernel_row);
    // @0x46dbe0 (Source 2)
    void PECalculate(ComputeInfo& info, int out_y, int out_x, bool accumulate, bool last_kernel_row);
    // @0x46e5b0 (Source 3)
    void FillIf(ComputeInfo& info, int y, int x, int count);
    // @0x46e950 (Source 4)
    void ComputeConv(ComputeInfo& info);
};

