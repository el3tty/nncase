#pragma once
// MeshNet: the MFU "act1" engine of the K230 NPU C-model.  Two engines live in this class:
//   * MfuAct1()   - element-wise  dst = linefit(src1 (+|*) src2)  over GLB / PSUM_L1 tensors (the Act1 instructions),
//   * MnCompute() - a small data-flow graph of 34 nodes ("mesh") of bf16 operators (MNE) with an optional reduction.
// Lifted from IDA/Hex-Rays output (MeshNet1..MeshNet16 in sources/).
//
// The MeshNet is a singleton (MeshNet::GetMeshNet()).  The Mfu*Act1Conf* instruction handlers configure it by raw
// byte offset (`GetMeshNet() + offset`), so every register below carries its original byte offset in the comment
//.
//
// TODO(layout): the Act1 register block (+4032..+4219) is written by the Mfu*Act1Conf*Instruction files with raw
// offsets (qword slots 513..522 and dword slots).  The names used here describe how MfuAct1() / MnCompute() read them.
// TODO(layout): globals.cpp defines `uint32_t MeshNet_MeshNetInst[0x4000]` as the storage of the original singleton;
// this lift keeps the instance as a function-local static in GetMeshNet() instead (same size, same offsets).
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include "math/numeric_types.h"
#include "engines/mne.h"

// One vertex of the mesh graph (48 bytes, 34 of them from MeshNet+8).  Nodes 0..15 are MNE operators;
// the others are inputs / constants / the sink (node 33).  The first 16 bytes overlay the MNE object
// (op at +0, configuration word at +44).
struct MeshNode {
    MNE::Op op_;                              // +0   operator selected by MNE::MneProc
    uint8_t in_[3];                           // +8   producer node of operand 0/1/2 (0xFF = unconnected)
    uint8_t pad11_[5];                        // +11
    BF16::bfloat16 * operand_[3];             // +16  operand pointers (point at the producers' `result`)
    BF16::bfloat16 result_;                   // +40  value produced by the node (bf16)
    uint8_t pad42_[2];                        // +42
    uint32_t cfg_;                            // +44  operator configuration word
};

struct MeshNet {
    // ---- reduction operator -------------------------------------------------------------------------------
    REDUCE_ELEMENT::Fn reduce_fn_;            // +0    set by MeshNetReduce (NOT a vtable: the class has no virtuals)

    // ---- the mesh graph -------------------------------------------------------------------------------------
    MeshNode node_[34];                       // +8    node[k] is at 8 + 48*k; its in[] bytes are at 16 + 48*k .. 18 + 48*k
    uint8_t  reduce_enable_;                  // +1640 MnCompute: 1 = reduce groups of reduce_len results
    uint8_t  order_count_;                    // +1641 number of entries of `order` as counted by MnConstruct / MnCompute
    std::vector<uint8_t> order_;              // +1648 evaluation order of the nodes (sink first, built by MnConstruct)
    uint8_t  reserved1672_[8];                // +1672
    uint8_t  reserved1680_[128];              // +1680 zeroed at construction, otherwise unused
    std::ofstream log_[4];                    // +1808 debug trace streams (+1808, +2320, +2832, +3344)
    uint64_t string_slot_;                    // +3856 an empty COW std::string in the original (never touched)
    uint8_t  reserved3864_[8];                // +3864
    uint8_t  reduce_op_;                      // +3872 MeshNetReduce: 1 min, 2 add, 3 sub, 4 mul, otherwise max
    uint8_t  reserved3873_[3];                // +3873
    uint32_t node17_cfg_;                     // +3876 configuration word used instead of node[17].cfg
    uint8_t  node17_per_channel_;             // +3880 non-zero: node 17 receives the current channel as its function-set index
    uint8_t  mne_arg_[14];                    // +3881 MneProc argument of nodes 0..11, 13, 15 (index = node 0..11, then 12 -> node 13, 13 -> node 15)
    uint8_t  cfg_slot_;                       // +3895 MeshNetOpRoutConfig: which input byte (0..31) to write
    uint8_t  cfg_source_;                     // +3896 MeshNetRoutOpConfig: source node id (0..18)
    uint8_t  cfg_code_;                       // +3897 connection code (1..14, mapped by ProducerNodeFromCode)
    uint8_t  cfg_target_;                     // +3898 target selector (0..8, mapped by ConsumerNodeFromTarget)
    uint8_t  reserved3899_[5];                // +3899
    uint64_t src0_addr_;                      // +3904 MnCompute: GLB address of input 0
    uint64_t src1_addr_;                      // +3912 MnCompute: GLB address of input 1
    uint64_t dst0_addr_;                      // +3920 MnCompute: GLB address of output 0
    uint64_t dst1_addr_;                      // +3928 MnCompute: GLB address of output 1 (reduced results)
    uint8_t  pad3936_[2];                     // +3936
    uint16_t const_val_[4];                   // +3938 MnCompute: bf16 constants feeding nodes 27..30
    uint8_t  pad3946_[6];                     // +3946
    uint64_t chw_in0_;                        // +3952 MnCompute: strides of input 0 (GetCHW)
    uint64_t chw_in1_;                        // +3960 MnCompute: strides of input 1
    uint64_t chw_out_;                        // +3968 MnCompute: strides of output 0
    uint32_t in0_slice_len_;                  // +3976 GetBroadAddress arguments of input 0
    uint32_t in0_rpt_a_;                      // +3980
    uint32_t in0_rpt_b_;                      // +3984
    uint32_t in1_slice_len_;                  // +3988 GetBroadAddress arguments of input 1
    uint32_t in1_rpt_a_;                      // +3992
    uint32_t in1_rpt_b_;                      // +3996
    uint8_t  reserved4000_[4];                // +4000
    uint32_t in0_len_;                        // +4004 total length of input 0 (0 = broadcast disabled)
    uint64_t in0_dims_;                       // +4008 MnCompute: logical dims of input 0, 4 halfwords (w, h, c, n)
    uint32_t in1_len_;                        // +4016 total length of input 1
    uint8_t  reserved4020_[4];                // +4020
    uint64_t in1_dims_;                       // +4024 logical dims of input 1
    uint16_t dq0_scale_;                      // +4032 input 0 de-quantisation scale (bf16 bits)
    uint8_t  dq0_zero_;                       // +4034 input 0 zero point
    uint8_t  dq0_signed_;                     // +4035 input 0 is int8
    uint8_t  dq0_enable_;                     // +4036 input 0 holds 8-bit data
    uint8_t  pad4037_;                        // +4037
    uint16_t dq1_scale_;                      // +4038 input 1 de-quantisation scale
    uint8_t  dq1_zero_;                       // +4040
    uint8_t  dq1_signed_;                     // +4041
    uint8_t  dq1_enable_;                     // +4042
    uint8_t  pad4043_;                        // +4043
    uint16_t q0_scale_;                       // +4044 output 0 quantisation scale (bf16 bits)
    uint16_t q0_zero_;                        // +4046 output 0 zero point (bf16 bits)
    uint8_t  q0_signed_;                      // +4048 output 0 is int8
    uint8_t  q0_enable_;                      // +4049 output 0 holds 8-bit data
    uint16_t q1_scale_;                       // +4050 output 1 quantisation scale
    uint16_t q1_zero_;                        // +4052
    uint8_t  q1_signed_;                      // +4054
    uint8_t  q1_enable_;                      // +4055
    uint32_t elem_count_;                     // +4056 MnCompute: number of elements to process
    uint8_t  reserved4060_[4];                // +4060
    uint64_t out_dims_;                       // +4064 logical dims of the output, 4 halfwords (w, h, c, n)
    uint16_t reduce_init_;                    // +4072 MnReduceProc: initial value (bf16 bits)
    uint8_t  reserved4074_[2];                // +4074
    uint32_t reduce_len_;                     // +4076 MnReduceProc: results per reduction group
    uint8_t  reserved4080_[4];                // +4080
    uint8_t  write_both_;                     // +4084 reduce mode: also store every un-reduced result into output 0
    uint8_t  reserved4085_[3];                // +4085

    // ---- MfuAct1 register block ----------------------------------------------------------------------------
    uint64_t a1_src1_chw_;                    // +4088 src1 strides (3 halfwords, MFU::GetCHW)
    uint64_t a1_src2_chw_;                    // +4096 src2 strides
    uint64_t a1_dst_chw_;                     // +4104 dst pitches (halfwords: w, h, c)
    uint32_t a1_s1_slice_len_;                // +4112 src1 slice length (GetBroadAddress / L1_slice_rpt_buffer_len)
    uint32_t a1_s1_rpt_a_;                    // +4116 src1 repeat factors
    uint32_t a1_s1_rpt_b_;                    // +4120
    uint32_t a1_s2_slice_len_;                // +4124 src2 slice length
    uint32_t a1_s2_rpt_a_;                    // +4128 src2 repeat factors
    uint32_t a1_s2_rpt_b_;                    // +4132
    uint8_t  a1_s1_no_l1_check_;              // +4136 skip the L1 slice buffer check for src1
    uint8_t  a1_s2_no_l1_check_;              // +4137 skip the L1 slice buffer check for src2
    uint8_t  reserved4138_[2];                // +4138
    uint32_t a1_s1_rpt_c_;                    // +4140 src1 third repeat factor
    uint64_t a1_src1_dims_;                   // +4144 src1 logical dims (w, h, c, n)
    uint16_t a1_src1_psum_;                   // +4152 non-zero: src1 comes from PSUM_L1 instead of GLB
    uint8_t  reserved4154_[2];                // +4154
    uint32_t a1_s2_rpt_c_;                    // +4156 src2 third repeat factor
    uint64_t a1_src2_dims_;                   // +4160 src2 logical dims
    uint16_t a1_src2_psum_;                   // +4168 src2 flag (only used for the alignment check)
    uint8_t  reserved4170_[2];                // +4170
    uint32_t a1_dst_len_;                     // +4172 number of destination elements
    uint64_t a1_dst_dims_;                    // +4176 loop extents (w, h, c, n)
    uint16_t a1_s1_scale_;                    // +4184 src1 de-quantisation scale (fp16 bits)
    uint16_t a1_s1_zero_;                     // +4186 src1 zero point
    uint8_t  a1_s1_type_;                     // +4188 src1 element type: 0 fp16, 1 uint8, 2 int8, 3 int16
    uint8_t  a1_s1_shift_;                    // +4189 src1 exponent shift
    uint16_t a1_s2_scale_;                    // +4190
    uint16_t a1_s2_zero_;                     // +4192
    uint8_t  a1_s2_type_;                     // +4194
    uint8_t  a1_s2_shift_;                    // +4195
    uint8_t  a1_dst_type_;                    // +4196 destination element type
    uint8_t  a1_fit_shift_;                   // +4197 exponent shift of the line-fit stage
    uint8_t  a1_op_mul_;                      // +4198 non-zero: src1 * src2, zero: src1 + src2
    uint8_t  a1_per_channel_;                 // +4199 non-zero: line-fit parameter set = channel index
    uint8_t  a1_use_mfu_fit_;                 // +4200 non-zero: mfu_linefit (16 segments), zero: act1_linefit
    uint8_t  reserved4201_[3];                // +4201
    uint32_t a1_src1_addr_;                   // +4204 GLB (or PSUM_L1) address of src1
    uint32_t a1_src2_addr_;                   // +4208
    uint32_t a1_dst_addr_;                    // +4212
    uint32_t a1_fit_addr_;                    // +4216 GLB address of the line-fit parameter table

    MeshNet();
    ~MeshNet();                              // not virtual (the original object has no vptr)

    // Singleton accessor.  @0x445f80 (MeshNet16)
    // Callers (Mfu*Act1* instructions) address the object by raw byte offset (TODO(layout)).
    static MeshNet * GetMeshNet();

    // Connection tables.  The compiler lookup tables CSWTCH.874 / .876 of the original are expanded to switch statements in meshnet.cpp.
    void MeshNetOpRoutConfig();              // @0x438620 (MeshNet2)   cfg_slot <- source node of cfg_code
    void MeshNetRoutOpConfig();              // @0x438830 (MeshNet3)   node[target].in[0] <- cfg_source
    void MeshNetRoutRoutConfig();            // @0x438870 (MeshNet4)   node[target].in[0] <- source node of cfg_code

    // Linear element index -> broadcast source index.  @0x4388c0 (MeshNet5)
    static uint32_t GetBroadAddress(uint32_t len, uint32_t rpt_a, uint32_t slice_len, uint32_t rpt_b, uint32_t index,
                                    uint8_t k0, uint8_t k1, uint8_t k2);

    // Reduces `values` with reduce_fn.  @0x438930 (MeshNet6)
    BF16::bfloat16 MnReduceProc(const std::vector<BF16::bfloat16> & values) const;

    // Piece-wise linear approximation (fp16): table at `table_addr` (GLB address) selected by `param_set`.
    void mfu_linefit(FP16::fp16 * x, FP16::fp16 * y, uint32_t table_addr, uint8_t ** glb, uint16_t param_set,
                     uint8_t segments);      // @0x438a20 (MeshNet7)
    void act1_linefit(FP16::fp16 * x, FP16::fp16 * y, uint32_t table_addr, uint8_t ** glb, uint16_t param_set);
                                             // @0x438c70 (MeshNet8)

    // Number of 32-byte L1 lines a repeated slice occupies; exits when the slice is not a multiple of ShapeW.
    // @0x438ea0 (MeshNet9)
    static uint32_t L1_slice_rpt_buffer_len(uint32_t addr, uint64_t dims, uint64_t chw, uint32_t slice_len,
                                            uint32_t elem_bytes);

    void MeshNetOp();                        // @0x43a660 (MeshNet10)   run MneProc on every node
    void MeshNetReduce();                    // @0x43a8a0 (MeshNet11)   select reduce_fn from reduce_op
    void MfuAct1();                          // @0x43da10 (MeshNet12)
    void MnConstruct(uint8_t index);         // @0x440020 (MeshNet13)
    void MnPrune();                          // @0x440200 (MeshNet14)
    void MnCompute();                        // @0x442570 (MeshNet15)
};
