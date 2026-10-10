#pragma once
// Conv2D: the convolution engine ("PU") of the K230 NPU C-model. Lifted from IDA/Hex-Rays output
// (Conv2D1..Conv2D5.cpp).
//
// The PU configuration instructions (pu*instruction.cpp) write the Conv2D singleton by raw byte offset
// (TODO(layout) there); the layout below reproduces those offsets exactly, so the
// raw accesses keep working. PuComputeInstruction snapshots the configuration into a PuCompute and queues it;
// Conv2D::Compute() later pops it together with the weight / IF-load descriptors and runs the TCU, forwarding
// the result to ACT0 (Activate) when the compute mode is 1.
//
// Address -> field mapping (offset: writer -> PuCompute field):
//     0: PuFetchifConf1 stride_w -> fetch_imm17          4: stride_h -> fetch_imm22
//     8,12,16: PuFetchifConf1 shape words 2,1,0       20: cleared by PuFetchifConf1
//    24: PuFetchifConf3 rgroups_val -> groups             28: rs1_val -> if_base
//    32,36,40,44: PuFetchifConf3 shape words 3..0 -> if3_d3, in_channels, in_height, in_width
//    48: PuFetchifConf4 rpad_value_val (byte) -> pad_value   52,56,60,64: shape words 2,3,1,0 -> pad_bottom,pad_top,pad_left,pad_right
//  68..91: PuFetchifConf_deq byte table -> if_zero_points   92: quant_type -> if_deq_mode
//    96,100: PuWConf kernel_h, kernel_w -> kernel_h, kernel_w
//   104,108,112: PuOfConf1 shape words 2,1,0 -> of1_d2,of1_d1,of1_d0   116: cleared by PuOfConf1
//   120,124,128,132: PuOfConf2 shape words 3..0 -> of2_d3,of2_d2,of2_d1,of2_d0   136: raddr_d_val -> psum_base
//   140,141,142,144,148: PuComputeConf flags/mode/param -> accumulate, clear_psum, flag_142, mode, param_148
//   152: PuCompute of_shift_mode -> shift_mode        156: set by Compute() (sticky copy of flag_142), starts as 1
//  160,240,320,400,480,576: the six std::deque<std::shared_ptr<...>> work queues (DmLoadL1, DmLoadW, DmStoreOf,
//   DmLoadAct0, Act0Compute, PuCompute) pushed to by the Dm* / Act0* / PuCompute instructions.
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include "engines/dm.h"
#include "engines/act0.h"

// Snapshot of the PU configuration taken by Conv2D::GetPuCompute() (160 bytes, same layout as Conv2D+0).
struct PuCompute {
    uint32_t fetch_imm17_ = 0;    // +0   PuFetchifConf1 stride_w
    uint32_t fetch_imm22_ = 0;    // +4   PuFetchifConf1 stride_h
    uint32_t if_shape_d2_ = 0;    // +8   PuFetchifConf1 shape word 2
    uint32_t if_shape_d1_ = 0;    // +12
    uint32_t if_shape_d0_ = 0;    // +16
    uint32_t reserved20_ = 0;     // +20  always 0
    uint32_t groups_ = 0;         // +24  PuFetchifConf3 rgroups_val: number of channel groups
    uint32_t if_base_ = 0;        // +28  PuFetchifConf3 raddr_s_val: byte offset of the tile in the IF buffer
    uint32_t if3_d3_ = 0;         // +32  PuFetchifConf3 shape word 3
    uint32_t in_channels_ = 0;    // +36  shape word 2: input channels = PE rows
    uint32_t in_height_ = 0;      // +40  shape word 1: padded input height
    uint32_t in_width_ = 0;       // +44  shape word 0: padded input width
    uint8_t  pad_value_ = 0;      // +48  PuFetchifConf4 rpad_value_val (low byte): value fed outside the tensor
    uint8_t  pad49_[3] = {};      // +49
    uint32_t pad_bottom_ = 0;     // +52  shape word 2   (asm @0x4681e0: word 52/56 are subtracted from in_height, 60/64 from in_width; side names inferred)
    uint32_t pad_top_ = 0;        // +56  shape word 3
    uint32_t pad_left_ = 0;       // +60  shape word 1
    uint32_t pad_right_ = 0;      // +64  shape word 0
    uint8_t  if_zero_points_[24] = {};  // +68 PuFetchifConf_deq: per-input-channel zero points
    uint32_t if_deq_mode_ = 0;    // +92  PuFetchifConf_deq quant_type: 1 = unsigned IF with zero points
    uint32_t kernel_h_ = 0;       // +96  PuWConf kernel_h
    uint32_t kernel_w_ = 0;       // +100 PuWConf kernel_w
    uint32_t of1_d2_ = 0;         // +104 PuOfConf1 shape word 2
    uint32_t of1_d1_ = 0;         // +108
    uint32_t of1_d0_ = 0;         // +112
    uint32_t of1_zero_ = 0;       // +116 always 0
    uint32_t of2_d3_ = 0;         // +120 PuOfConf2 shape word 3
    uint32_t of2_d2_ = 0;         // +124 (bounds the weight column index in TCU::FillWeight)
    uint32_t of2_d1_ = 0;         // +128
    uint32_t of2_d0_ = 0;         // +132
    uint32_t psum_base_ = 0;      // +136 PuOfConf2 raddr_d_val: byte offset in _G.PSUM_L1 of the output
    uint8_t  accumulate_ = 0;     // +140 PuComputeConf load_psum: accumulate into the PSUM already present
    uint8_t  clear_psum_ = 0;     // +141 PuComputeConf clr_psum: zero _G.PSUM_L1 before computing
    uint8_t  flag_142_ = 0;       // +142 PuComputeConf release_if
    uint8_t  pad143_ = 0;         // +143
    uint32_t mode_ = 0;           // +144 PuComputeConf dest_target: 1 = result is forwarded to ACT0
    uint32_t param_148_ = 0;      // +148 PuComputeConf mode: non-zero selects window step 1 and col_stride = fetch_imm17 (asm @0x4686ab)
    uint32_t shift_mode_ = 0;     // +152 PuCompute of_shift_mode: PSUM scaling (0 none, 1 << 4, else >> 4)
    uint8_t  if_flag_ = 0;        // +156 non-zero: Compute() takes an IF tile from the DmLoadL1 queue
    uint8_t  pad157_ = 0;           // +157
    uint8_t  pad158_[2] = {};     // +158
};

struct Conv2D {
    PuCompute cfg_;                                              // +0    current PU configuration
    std::deque<std::shared_ptr<DmLoadL1>>   if_queue_;           // +160  IF tiles waiting for Compute()
    std::deque<std::shared_ptr<DmLoadW>>    weight_queue_;       // +240  DmLoadW descriptors
    std::deque<std::shared_ptr<DmStoreOf>>  store_queue_;        // +320  DmStoreOf descriptors
    std::deque<std::shared_ptr<DmLoadAct0>> act0_param_queue_;   // +400  DmLoadAct0 descriptors
    std::deque<std::shared_ptr<Act0Compute>> act0_queue_;        // +480  Act0Compute descriptors
    std::shared_ptr<Act0Compute> cur_act0_;                      // +560  current ACT0 descriptor (installed by Act0Src1ConfInstruction)
    std::deque<std::shared_ptr<PuCompute>>  compute_queue_;      // +576  queued PuCompute snapshots

    Conv2D();
    ~Conv2D();

    // @0x467450 (Source 1): snapshot of the current configuration.
    std::shared_ptr<PuCompute> GetPuCompute() const;
    // @0x467590 (Source 2): pops the queued ACT0 descriptors (+ DM store) and runs Act0::Compute.
    void Activate();
    // @0x4681e0 (Source 3): runs one queued PuCompute through the TCU.
    void Compute();
    // @0x468f10 (Source 4): the singleton (a function-local static in the original).
    static Conv2D* GetConv2D();

    // Compatibility shims for callers still written against the garbage generated prototypes
    // (pucomputeinstruction.cpp calls these through a reinterpret_cast of the result slot).
    // They write the shared_ptr into *this, which must be raw storage holding a shared_ptr<PuCompute>-sized slot.
    Conv2D* GetPuCompute(int64_t* conv);
    void Compute(int64_t, int64_t, int64_t, int, int) { Compute(); }
};

