#pragma once
// MeshNet: the MFU "act1" engine of the K230 NPU C-model.  Two engines live in this class:
//   * MfuAct1   - element-wise  dst = linefit(src1 (+|*) src2)  over GLB / PSUM_L1 tensors (the Act1 instructions),
//   * MnCompute - a small data-flow graph of 34 nodes ("mesh") of bf16 operators (MNE) with an optional reduction.
//
// The MeshNet is a singleton (MeshNet::GetMeshNet).
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include "math/numeric_types.h"
#include "engines/mne.h"

// One vertex of the mesh graph (34 of them).  Nodes 0..15 are MNE operators (op_ / op_config_ come from MNE);
// the others are inputs / constants / the sink (node 33).
struct MeshNode : MNE {
    uint8_t in_[3];                           // producer node of operand 0/1/2 (0xFF = unconnected)
    BF16::bfloat16 * operand_[3];             // operand pointers (point at the producers' `result`)
    BF16::bfloat16 result_;                   // value produced by the node (bf16)
};

struct MeshNet {
    // ---- reduction operator -------------------------------------------------------------------------------
    REDUCE_ELEMENT::Fn reduce_fn_;            // set by MeshNetReduce (NOT a vtable: the class has no virtuals)

    // ---- the mesh graph -------------------------------------------------------------------------------------
    MeshNode node_[34];                       // node[k] is at 8 + 48*k; its in[] bytes are at 16 + 48*k .. 18 + 48*k
    uint8_t  reduce_enable_;                  // MnCompute: 1 = reduce groups of reduce_len results
    uint8_t  order_count_;                    // number of entries of `order` as counted by MnConstruct / MnCompute
    std::vector<uint8_t> order_;              // evaluation order of the nodes (sink first, built by MnConstruct)
    std::ofstream log_[4];                    // debug trace streams 
    uint8_t  reduce_op_;                      // MeshNetReduce: 1 min, 2 add, 3 sub, 4 mul, otherwise max
    uint32_t node17_cfg_;                     // configuration word used instead of node[17].cfg
    uint8_t  node17_per_channel_;             // non-zero: node 17 receives the current channel as its function-set index
    uint8_t  mne_arg_[14];                    // MneProc argument of nodes 0..11, 13, 15 (index = node 0..11, then 12 -> node 13, 13 -> node 15)
    uint8_t  cfg_slot_;                       // MeshNetOpRoutConfig: which input byte (0..31) to write
    uint8_t  cfg_source_;                     // MeshNetRoutOpConfig: source node id (0..18)
    uint8_t  cfg_code_;                       // connection code (1..14, mapped by ProducerNodeFromCode)
    uint8_t  cfg_target_;                     // target selector (0..8, mapped by ConsumerNodeFromTarget)
    uint64_t src0_addr_;                      // MnCompute: GLB address of input 0
    uint64_t src1_addr_;                      // MnCompute: GLB address of input 1
    uint64_t dst0_addr_;                      // MnCompute: GLB address of output 0
    uint64_t dst1_addr_;                      // MnCompute: GLB address of output 1 (reduced results)
    uint16_t const_val_[4];                   // MnCompute: bf16 constants feeding nodes 27..30
    uint64_t chw_in0_;                        // MnCompute: strides of input 0 (GetCHW)
    uint64_t chw_in1_;                        // MnCompute: strides of input 1
    uint64_t chw_out_;                        // MnCompute: strides of output 0
    uint32_t in0_slice_len_;                  // GetBroadAddress arguments of input 0
    uint32_t in0_rpt_a_;
    uint32_t in0_rpt_b_;
    uint32_t in1_slice_len_;                  // GetBroadAddress arguments of input 1
    uint32_t in1_rpt_a_;
    uint32_t in1_rpt_b_;
    uint32_t in0_len_;                        // total length of input 0 (0 = broadcast disabled)
    uint64_t in0_dims_;                       // MnCompute: logical dims of input 0, 4 halfwords (w, h, c, n)
    uint32_t in1_len_;                        // total length of input 1
    uint64_t in1_dims_;                       // logical dims of input 1
    uint16_t dq0_scale_;                      // input 0 de-quantisation scale (bf16 bits)
    uint8_t  dq0_zero_;                       // input 0 zero point
    uint8_t  dq0_signed_;                     // input 0 is int8
    uint8_t  dq0_enable_;                     // input 0 holds 8-bit data
    uint16_t dq1_scale_;                      // input 1 de-quantisation scale
    uint8_t  dq1_zero_;
    uint8_t  dq1_signed_;
    uint8_t  dq1_enable_;
    uint16_t q0_scale_;                       // output 0 quantisation scale (bf16 bits)
    uint16_t q0_zero_;                        // output 0 zero point (bf16 bits)
    uint8_t  q0_signed_;                      // output 0 is int8
    uint8_t  q0_enable_;                      // output 0 holds 8-bit data
    uint16_t q1_scale_;                       // output 1 quantisation scale
    uint16_t q1_zero_;
    uint8_t  q1_signed_;
    uint8_t  q1_enable_;
    uint32_t elem_count_;                     // MnCompute: number of elements to process
    uint64_t out_dims_;                       // logical dims of the output, 4 halfwords (w, h, c, n)
    uint16_t reduce_init_;                    // MnReduceProc: initial value (bf16 bits)
    uint32_t reduce_len_;                     // MnReduceProc: results per reduction group
    uint8_t  write_both_;                     // reduce mode: also store every un-reduced result into output 0

    // ---- MfuAct1 register block ----------------------------------------------------------------------------
    uint64_t a1_src1_chw_;                    // src1 strides (3 halfwords, MFU::GetCHW)
    uint64_t a1_src2_chw_;                    // src2 strides
    uint64_t a1_dst_chw_;                     // dst pitches (halfwords: w, h, c)
    uint32_t a1_s1_slice_len_;                // src1 slice length (GetBroadAddress / L1_slice_rpt_buffer_len)
    uint32_t a1_s1_rpt_a_;                    // src1 repeat factors
    uint32_t a1_s1_rpt_b_;
    uint32_t a1_s2_slice_len_;                // src2 slice length
    uint32_t a1_s2_rpt_a_;                    // src2 repeat factors
    uint32_t a1_s2_rpt_b_;
    uint8_t  a1_s1_no_l1_check_;              // skip the L1 slice buffer check for src1
    uint8_t  a1_s2_no_l1_check_;              // skip the L1 slice buffer check for src2
    uint32_t a1_s1_rpt_c_;                    // src1 third repeat factor
    uint64_t a1_src1_dims_;                   // src1 logical dims (w, h, c, n)
    uint16_t a1_src1_psum_;                   // non-zero: src1 comes from _G.PSUM_L1 instead of GLB
    uint32_t a1_s2_rpt_c_;                    // src2 third repeat factor
    uint64_t a1_src2_dims_;                   // src2 logical dims
    uint16_t a1_src2_psum_;                   // src2 flag (only used for the alignment check)
    uint32_t a1_dst_len_;                     // number of destination elements
    uint64_t a1_dst_dims_;                    // loop extents (w, h, c, n)
    uint16_t a1_s1_scale_;                    // src1 de-quantisation scale (fp16 bits)
    uint16_t a1_s1_zero_;                     // src1 zero point
    uint8_t  a1_s1_type_;                     // src1 element type: 0 fp16, 1 uint8, 2 int8, 3 int16
    uint8_t  a1_s1_shift_;                    // src1 exponent shift
    uint16_t a1_s2_scale_;
    uint16_t a1_s2_zero_;
    uint8_t  a1_s2_type_;
    uint8_t  a1_s2_shift_;
    uint8_t  a1_dst_type_;                    // destination element type
    uint8_t  a1_fit_shift_;                   // exponent shift of the line-fit stage
    uint8_t  a1_op_mul_;                      // non-zero: src1 * src2, zero: src1 + src2
    uint8_t  a1_per_channel_;                 // non-zero: line-fit parameter set = channel index
    uint8_t  a1_use_mfu_fit_;                 // non-zero: mfu_linefit (16 segments), zero: act1_linefit
    uint32_t a1_src1_addr_;                   // GLB (or _G.PSUM_L1) address of src1
    uint32_t a1_src2_addr_;
    uint32_t a1_dst_addr_;
    uint32_t a1_fit_addr_;                    // GLB address of the line-fit parameter table

    MeshNet();
    ~MeshNet();                              // not virtual (the original object has no vptr)

    // Returns the object to its freshly constructed state (registers zeroed, nodes unconnected, evaluation order empty,
    // trace streams closed). Called when a simulator run ends, because the singleton survives between runs when the
    // simulator is loaded as a DLL.
    void Reset();

private:
    // Initializes the POD members (called by the constructor and by Reset); order_ and log_ are not touched.
    void Init();

public:

    // Singleton accessor.
    static MeshNet * GetMeshNet();

    // Connection tables.  The compiler lookup tables CSWTCH.874 / .876 of the original are expanded to switch statements in meshnet.cpp.
    void MeshNetOpRoutConfig();              // cfg_slot <- source node of cfg_code
    void MeshNetRoutOpConfig();              // node[target].in[0] <- cfg_source
    void MeshNetRoutRoutConfig();            // node[target].in[0] <- source node of cfg_code

    // Linear element index -> broadcast source index.
    static uint32_t GetBroadAddress(uint32_t len, uint32_t rpt_a, uint32_t slice_len, uint32_t rpt_b, uint32_t index,
                                    uint8_t k0, uint8_t k1, uint8_t k2);

    // Reduces `values` with reduce_fn.
    BF16::bfloat16 MnReduceProc(const std::vector<BF16::bfloat16> & values) const;

    // Piece-wise linear approximation (fp16): table at `table_addr` (GLB address) selected by `param_set`.
    void mfu_linefit(FP16::fp16 * x, FP16::fp16 * y, uint32_t table_addr, uint8_t ** glb, uint16_t param_set,
                     uint8_t segments);
    void act1_linefit(FP16::fp16 * x, FP16::fp16 * y, uint32_t table_addr, uint8_t ** glb, uint16_t param_set);

    // Number of 32-byte L1 lines a repeated slice occupies; exits when the slice is not a multiple of ShapeW.
    static uint32_t L1_slice_rpt_buffer_len(uint32_t addr, uint64_t dims, uint64_t chw, uint32_t slice_len,
                                            uint32_t elem_bytes);

    void MeshNetOp();                        // run MneProc on every node
    void MeshNetReduce();                    // select reduce_fn from reduce_op
    void MfuAct1();
    void MnConstruct(uint8_t index);
    void MnPrune();
    void MnCompute();
};
