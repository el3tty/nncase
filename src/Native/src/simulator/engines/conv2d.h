#pragma once
// Conv2D: the convolution engine ("PU") of the K230 NPU C-model.
//
// PuComputeInstruction snapshots the configuration into a PuCompute and queues it; Conv2D::Compute
// later pops it together with the weight / IF-load descriptors and runs the TCU, forwarding the result
// to ACT0 (Activate) when the compute mode is 1.
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include "engines/dm.h"
#include "engines/act0.h"

// Snapshot of the PU configuration taken by Conv2D::GetPuCompute (160 bytes, same layout as ).
struct PuCompute {
    uint32_t fetch_imm17_ = 0;    // PuFetchifConf1 stride_w
    uint32_t fetch_imm22_ = 0;    // PuFetchifConf1 stride_h
    uint32_t if_shape_d2_ = 0;    // PuFetchifConf1 shape word 2
    uint32_t if_shape_d1_ = 0;
    uint32_t if_shape_d0_ = 0;
    uint32_t reserved20_ = 0;     // always 0
    uint32_t groups_ = 0;         // PuFetchifConf3 rgroups_val: number of channel groups
    uint32_t if_base_ = 0;        // PuFetchifConf3 raddr_s_val: byte offset of the tile in the IF buffer
    uint32_t if3_d3_ = 0;         // PuFetchifConf3 shape word 3
    uint32_t in_channels_ = 0;    // shape word 2: input channels = PE rows
    uint32_t in_height_ = 0;      // shape word 1: padded input height
    uint32_t in_width_ = 0;       // shape word 0: padded input width
    uint8_t  pad_value_ = 0;      // PuFetchifConf4 rpad_value_val (low byte): value fed outside the tensor
    uint32_t pad_bottom_ = 0;     // shape word 2   (asm @0x4681e0: word 52/56 are subtracted from in_height, 60/64 from in_width; side names inferred)
    uint32_t pad_top_ = 0;        // shape word 3
    uint32_t pad_left_ = 0;       // shape word 1
    uint32_t pad_right_ = 0;      // shape word 0
    uint8_t  if_zero_points_[24] = {};  // PuFetchifConf_deq: per-input-channel zero points
    uint32_t if_deq_mode_ = 0;    // PuFetchifConf_deq quant_type: 1 = unsigned IF with zero points
    uint32_t kernel_h_ = 0;       // PuWConf kernel_h
    uint32_t kernel_w_ = 0;       // PuWConf kernel_w
    uint32_t of1_d2_ = 0;         // PuOfConf1 shape word 2
    uint32_t of1_d1_ = 0;
    uint32_t of1_d0_ = 0;
    uint32_t of1_zero_ = 0;       // always 0
    uint32_t of2_d3_ = 0;         // PuOfConf2 shape word 3
    uint32_t of2_d2_ = 0;         // (bounds the weight column index in TCU::FillWeight)
    uint32_t of2_d1_ = 0;
    uint32_t of2_d0_ = 0;
    uint32_t psum_base_ = 0;      // PuOfConf2 raddr_d_val: byte offset in _G.PSUM_L1 of the output
    uint8_t  accumulate_ = 0;     // PuComputeConf load_psum: accumulate into the PSUM already present
    uint8_t  clear_psum_ = 0;     // PuComputeConf clr_psum: zero _G.PSUM_L1 before computing
    uint8_t  flag_142_ = 0;       // PuComputeConf release_if
    uint32_t mode_ = 0;           // PuComputeConf dest_target: 1 = result is forwarded to ACT0
    uint32_t param_148_ = 0;      // PuComputeConf mode: non-zero selects window step 1 and col_stride = fetch_imm17 (asm @0x4686ab)
    uint32_t shift_mode_ = 0;     // PuCompute of_shift_mode: PSUM scaling (0 none, 1 << 4, else >> 4)
    uint8_t  if_flag_ = 0;        // non-zero: Compute takes an IF tile from the DmLoadL1 queue
};

struct Conv2D {
    PuCompute cfg_;                                              // current PU configuration
    std::deque<std::shared_ptr<DmLoadL1>>   if_queue_;           // IF tiles waiting for Compute
    std::deque<std::shared_ptr<DmLoadW>>    weight_queue_;       // DmLoadW descriptors
    std::deque<std::shared_ptr<DmStoreOf>>  store_queue_;        // DmStoreOf descriptors
    std::deque<std::shared_ptr<DmLoadAct0>> act0_param_queue_;   // DmLoadAct0 descriptors
    std::deque<std::shared_ptr<Act0Compute>> act0_queue_;        // Act0Compute descriptors
    std::shared_ptr<Act0Compute> cur_act0_;                      // current ACT0 descriptor (installed by Act0Src1ConfInstruction)
    std::deque<std::shared_ptr<PuCompute>>  compute_queue_;      // queued PuCompute snapshots

    Conv2D();
    ~Conv2D();

    // Returns the object to its freshly constructed state (default configuration, empty queues). Called when a
    // simulator run ends, because the singleton survives between runs when the simulator is loaded as a DLL.
    void Reset();

    // snapshot of the current configuration.
    std::shared_ptr<PuCompute> GetPuCompute() const;
    // pops the queued ACT0 descriptors (+ DM store) and runs Act0::Compute.
    void Activate();
    // runs one queued PuCompute through the TCU.
    void Compute();
    // the singleton (a function-local static in the original).
    static Conv2D* GetConv2D();

    // Compatibility shims for callers still written against the garbage generated prototypes
    // (pucomputeinstruction.cpp calls these through a reinterpret_cast of the result slot).
    // They write the shared_ptr into *this, which must be raw storage holding a shared_ptr<PuCompute>-sized slot.
    Conv2D* GetPuCompute(int64_t* conv);
    void Compute(int64_t, int64_t, int64_t, int, int) { Compute(); }
};

