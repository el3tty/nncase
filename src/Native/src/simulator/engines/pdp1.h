#pragma once
// PDP1: the MFU pooling / reduction engine (driven by the MfuPdp1*Instruction family).
#include <cstddef>
#include <cstdint>
#include <vector>
#include "math/numeric_types.h"
#include "engines/memaccessor.h"

// One reduction window slot (32 bytes in the original; PDP1 owns 16 of them, only slot 0 is used
// by the code).
struct PdpWindow {
    std::vector<FP16::fp16> data_;  // window elements (written by PDP1::PdpDM)
    FP24::fp24 acc_;                // running max / min / sum, as fp24
    uint32_t count_;                // number of elements summed (average modes)
};

// PDP1 configuration registers, written by the MfuPdp1*Instruction::operation functions
// (store widths verified against the asm).
struct Pdp1Config {
    uint32_t src_addr_;          // MfuPdp1Compute rs1_val; bank[31:28] | offset[27:0] (_G.GLB)
    uint32_t dst_addr_;          // MfuPdp1Compute rd_val
    uint8_t  window_w_;          // Conf4 rs1_val (<= 64)
    uint8_t  window_h_;          // Conf4 rs2_val (<= 16)
    uint16_t avg_scale_;         // Conf4 rs3_val: fp16 factor applied in mode 3 (average)
    uint8_t  enable_h2c_;        // Conf4 flag_a (requires pad_top == 0)
    uint8_t  enable_bw_;         // Conf4 flag_b (requires 1x1 windows and strides)
    uint16_t dim1_count_;        // Compute shape bits 63..48: outer loop count
    uint16_t dim2_count_;        // Compute shape bits 47..32
    uint16_t shape_h_;           // Compute shape bits 31..16: input height
    uint16_t shape_w_;           // Compute shape bits 15..0:  input width
    uint16_t out_w_;             // Conf2 rs1_val: window positions along W
    uint16_t out_h_;             // Conf2 rs2_val: window positions along H
    uint8_t  stride_w_;          // Conf1 {rs2:rs1} low byte
    uint8_t  stride_h_;          // Conf1 high byte
    uint8_t  pad_top_;           // Conf3 shape bits 63..48
    uint8_t  pad_bottom_;        // Conf3 shape bits 47..32
    uint8_t  pad_left_;          // Conf3 shape bits 31..16
    uint8_t  pad_right_;         // Conf3 shape bits 15..0 (low byte; 0x54AA06 is its high byte)
    uint8_t  pool_mode_;         // Conf1 mode: 0 min, 1 max, 2 average by 1/count, 3 sum scaled by avg_scale (TIR PDP_FUNCTION min/max/average/sum)
    uint16_t src_pitch_;         // Conf1 shape_a: source row pitch (elements, multiple of 16/32 bytes)
    uint16_t src_plane_rows_;    // source rows per plane
    uint16_t src_dim2_;          // source planes per dim1 step
    uint8_t  conf3_rs1_;         // Conf3 rs1_val (unused by the lifted code)
    uint8_t  conf3_rs2_;         // Conf3 rs2_val (unused by the lifted code)
    uint16_t pad_value_;         // Conf3 rs3_val: fp16 value of out-of-bounds elements
    uint16_t dst_pitch_;         // Conf1 shape_b: destination row pitch
    uint16_t dst_plane_rows_;    // destination rows per plane
    uint16_t dst_dim2_;          // destination planes per dim1 step
    uint16_t dequant_scale_;     // Conf_deq rs1_val (fp16)
    uint16_t dequant_zero_;      // Conf_deq rs2_val
    uint8_t  src_dtype_;         // Conf_deq cfg low byte: 0 fp16, 1 uint8, 2 int8, 3 int16
    uint8_t  dequant_flag_;      // Conf_deq cfg high byte
    uint16_t quant_scale_;       // Conf_quant rs1_val (fp16)
    uint16_t quant_zero_;        // Conf_quant rs2_val
    uint8_t  dst_dtype_;         // Conf_quant cfg low byte (same encoding as src_dtype)
    uint8_t  quant_flag_;        // Conf_quant cfg high byte
};

struct PDP1 {
    std::vector<PdpWindow> windows_;     // 16 slots, created by the singleton constructor
    Pdp1Config cfg_;                     // configuration registers (see above)

    PDP1();
    // releases the window vector (and the per-window element vectors).
    virtual ~PDP1();

    // Singleton accessor; PDP1::GetPDP1 is not in the dump (inlined, "singleton lazy-init elided").
    static PDP1* GetPDP1();

    // Returns the object to its freshly constructed state (16 empty window slots, zeroed configuration). Called when
    // a simulator run ends, because the singleton survives between runs when the simulator is loaded as a DLL.
    void Reset();

    // fills windows[0].data with the window_h x window_w elements of the window
    // at output position (out_row, out_col) of block (dim1, dim2), dequantised from `src`.
    // The always-1 fifth parameter of the demangled signature is not used and dropped.
    void PdpDM(MemAccessor src, int dim1, int dim2, int out_row, int out_col);
    // validates the configuration, then runs the whole reduction.
    void PdpRedCompute();
};
