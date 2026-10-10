#pragma once
// PDP0: pooling / depthwise-convolution engine of the NPU C-model.
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>

struct CheckPoint;
struct L1Helper;                               // L1 tensor view (defined by its owner, not yet lifted)
struct WeightsHelper;                          // weight tile view (defined by its owner, not yet lifted)
template <typename T> struct Matrix4;          // 4-D matrix (defined by its owner, not yet lifted)
struct DmLoadW;                                // weight-load descriptor   (see DmLoadWInstruction)
struct DmStoreOf;                              // output-store descriptor  (see DmStoreOfInstruction)
struct DmLoadAct0;                             // activation-load descriptor (see DmLoadAct0Instruction)
struct Act0Compute;                            // activation compute descriptor (see Act0::GetAct0Compute)

// Snapshot of the PDP0 configuration registers (x10063) taken by
// PDP0::GetPdp0Compute when a PuPdp0ComputeInstruction is issued.
// Field index N is the 32-bit word at byte offset 4*N; "reg" is the word index into
// PDP0::GetPDP0 used by the pupdp0*instruction files (reg = 16384 + N).
struct Pdp0Compute {
    uint32_t mode_;           // [0]  reg 16384, PuPdp0ModeConf: 0 = depthwise conv, 1/2 = pooling kinds
    uint32_t stride_x_;       // [1]  reg 16385, PuPdp0FetchifConf1 imm17
    uint32_t stride_y_;       // [2]  reg 16386, PuPdp0FetchifConf1 imm22
    uint32_t in_dim3_;        // [3]  reg 16387, PuPdp0FetchifConf3 shape word 3
    uint32_t in_c_;           // [4]  reg 16388, input channels (shape word 2)
    uint32_t in_h_;           // [5]  reg 16389, input height before padding removal (shape word 1)
    uint32_t in_w_;           // [6]  reg 16390, input width before padding removal (shape word 0)
    uint32_t pad_value_;      // [7]  reg 16391, PuPdp0FetchifConf4 rs1: value of out-of-bounds pixels (depthwise)
    uint32_t pad_bottom_;     // [8]  reg 16392, shape word 2   (asm @0x46ce50: height = in_h - word9 - word8, width = in_w - word10 - word11; word names bottom/right are inferred)
    uint32_t pad_top_;        // [9]  reg 16393, shape word 3
    uint32_t pad_left_;       // [10] reg 16394, shape word 1
    uint32_t pad_right_;      // [11] reg 16395, shape word 0
    uint32_t in_zero_point_;  // [12] reg 16396, subtracted from unsigned inputs; written by PuPdp0Conf_deq (rs1_val)
    uint32_t unsigned_flag_;  // [13] reg 16397, ==1: signed-input flag is cleared; written by PuPdp0Conf_deq 
    uint32_t kernel_w_;       // [14] reg 16398, PuPdp0WConf imm22
    uint32_t kernel_h_;       // [15] reg 16399, PuPdp0WConf imm17
    uint32_t out_dim3_;       // [16] reg 16400, PuPdp0OfConf shape1 word 3
    uint32_t out_c_;          // [17] reg 16401
    uint32_t out_h_;          // [18] reg 16402
    uint32_t out_w_;          // [19] reg 16403
    uint32_t of_shape0_[3];   // [20..22] reg 16404..16406, PuPdp0OfConf shape0 words 2..0
    uint32_t of_reserved_;    // [23] reg 16407, always written as 0
    uint32_t psum_offset_;    // [24] reg 16408, PuPdp0Compute compute_param: byte offset into PSUM L1
};

// The PDP0 singleton. The raw byte offsets in the comments are what the pu*/dm*/act0*
// instruction files use through PDP0::GetPDP0.
struct PDP0 {
    Pdp0Compute regs_;             // configuration registers (words 16384..16408)
    std::deque<std::shared_ptr<DmLoadW>>     weight_queue_;      // DmLoadWInstruction -> ComputeDW
    std::deque<std::shared_ptr<DmStoreOf>>   store_of_queue_;    // DmStoreOfInstruction -> Activate
    std::deque<std::shared_ptr<DmLoadAct0>>  load_act0_queue_;   // DmLoadAct0Instruction -> Activate
    std::deque<std::shared_ptr<Act0Compute>> act0_queue_;        // Act0ComputeInstruction -> Activate
    std::shared_ptr<Act0Compute>             cur_act0_;          // installed by Act0Src1ConfInstruction
    std::deque<std::shared_ptr<Pdp0Compute>> compute_queue_;     // PuPdp0ComputeInstruction -> Compute

    // Meyers singleton (function-local static).
    static PDP0* GetPDP0();
    // PDP02  is the destructor (releases the five deques and the shared_ptr); defaulted.
    ~PDP0() = default;

    // Returns the object to its initial state (zeroed registers, empty queues). Called when a simulator run ends,
    // because the singleton survives between runs when the simulator is loaded as a DLL.
    void Reset();

    // pooling. `cfg` is the compute descriptor popped from compute_queue.
    static void ComputePDP0(L1Helper& in, L1Helper& out, bool is_signed, std::shared_ptr<Pdp0Compute> cfg);
    // depthwise convolution.
    static void ComputeDW(L1Helper& in, WeightsHelper& weights, L1Helper& out, bool in_signed, bool weights_signed,
                          Matrix4<uint8_t>& weight_zero_points, std::shared_ptr<Pdp0Compute> cfg);
    // snapshot of the configuration registers.
    std::shared_ptr<Pdp0Compute> GetPdp0Compute() const;
    // pops one entry from act0_queue / load_act0_queue / store_of_queue and runs Act0::Compute.
    void Activate();
    // runs the oldest queued Pdp0Compute (pooling or depthwise conv), then Activate.
    void Compute();
};
