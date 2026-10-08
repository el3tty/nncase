#pragma once
// PDP1: the MFU pooling / reduction engine ("PDP1", driven by the MfuPdp1*Instruction family),
// lifted from IDA/Hex-Rays output (nncase K230 C-model).
#include <cstddef>
#include <cstdint>
#include <vector>
#include "math/numeric_types.h"
#include "engines/memaccessor.h"

// One reduction window slot (32 bytes in the original; PDP1 owns 16 of them, only slot 0 is used
// by the lifted code).
struct PdpWindow {
    std::vector<FP16::fp16> data_;  // +0   window elements (written by PDP1::PdpDM)
    FP24::fp24 acc_;                // +24  running max / min / sum, as fp24
    uint32_t count_;                // +28  number of elements summed (average modes)
};
static_assert(sizeof(PdpWindow) == 32, "PdpWindow layout");

// PDP1 configuration registers. In the decompiled binary the PDP1 singleton is a static object
// whose fields are addressed as absolute globals; the original object base is 0x54A940, so the
// field at byte offset N of PDP1 is the global at 0x54A940 + N.  The writers are the MfuPdp1*Instruction::operation()
// functions; they write PDP1::GetPDP1()->cfg (store widths verified against the asm).
struct Pdp1Config {
    uint32_t src_addr_;          // +164 0x54A9E4  MfuPdp1Compute rs1_val; bank[31:28] | offset[27:0] (g_GLB)
    uint32_t dst_addr_;          // +168 0x54A9E8  MfuPdp1Compute rd_val
    uint8_t  window_w_;          // +172 0x54A9EC  Conf4 rs1_val (<= 64)
    uint8_t  window_h_;          // +173 0x54A9ED  Conf4 rs2_val (<= 16)
    uint16_t avg_scale_;         // +174 0x54A9EE  Conf4 rs3_val: fp16 factor applied in mode 3 (average)
    uint8_t  enable_h2c_;        // +176 0x54A9F0  Conf4 flag_a (requires pad_top == 0)
    uint8_t  enable_bw_;         // +177 0x54A9F1  Conf4 flag_b (requires 1x1 windows and strides)
    uint8_t  reserved_178_[2];   // +178 0x54A9F2
    uint16_t dim1_count_;        // +180 0x54A9F4  Compute shape bits 63..48: outer loop count
    uint16_t dim2_count_;        // +182 0x54A9F6  Compute shape bits 47..32
    uint16_t shape_h_;           // +184 0x54A9F8  Compute shape bits 31..16: input height
    uint16_t shape_w_;           // +186 0x54A9FA  Compute shape bits 15..0:  input width
    uint16_t out_w_;             // +188 0x54A9FC  Conf2 rs1_val: window positions along W
    uint16_t out_h_;             // +190 0x54A9FE  Conf2 rs2_val: window positions along H
    uint8_t  stride_w_;          // +192 0x54AA00  Conf1 {rs2:rs1} low byte
    uint8_t  stride_h_;          // +193 0x54AA01  Conf1 high byte
    uint8_t  pad_top_;           // +194 0x54AA02  Conf3 shape bits 63..48
    uint8_t  pad_bottom_;        // +195 0x54AA03  Conf3 shape bits 47..32
    uint8_t  pad_left_;          // +196 0x54AA04  Conf3 shape bits 31..16
    uint8_t  pad_right_;         // +197 0x54AA05  Conf3 shape bits 15..0 (low byte; 0x54AA06 is its high byte)
    uint8_t  reserved_198_;      // +198 0x54AA06
    uint8_t  pool_mode_;         // +199 0x54AA07  Conf1 mode: 0 min, 1 max, 2 average by 1/count, 3 sum scaled by avg_scale (TIR PDP_FUNCTION min/max/average/sum)
    uint8_t  reserved_200_[8];   // +200 0x54AA08
    uint16_t src_pitch_;         // +208 0x54AA10  Conf1 shape_a: source row pitch (elements, multiple of 16/32 bytes)
    uint16_t src_plane_rows_;    // +210 0x54AA12  source rows per plane
    uint16_t src_dim2_;          // +212 0x54AA14  source planes per dim1 step
    uint16_t reserved_214_;      // +214 0x54AA16
    uint8_t  conf3_rs1_;         // +216 0x54AA18  Conf3 rs1_val (unused by the lifted code)
    uint8_t  conf3_rs2_;         // +217 0x54AA19  Conf3 rs2_val (unused by the lifted code)
    uint16_t pad_value_;         // +218 0x54AA1A  Conf3 rs3_val: fp16 value of out-of-bounds elements
    uint8_t  reserved_220_[4];   // +220 0x54AA1C
    uint16_t dst_pitch_;         // +224 0x54AA20  Conf1 shape_b: destination row pitch
    uint16_t dst_plane_rows_;    // +226 0x54AA22  destination rows per plane
    uint16_t dst_dim2_;          // +228 0x54AA24  destination planes per dim1 step
    uint8_t  reserved_230_[10];  // +230 0x54AA26
    uint16_t dequant_scale_;     // +240 0x54AA30  Conf_deq rs1_val (fp16)
    uint16_t dequant_zero_;      // +242 0x54AA32  Conf_deq rs2_val
    uint8_t  src_dtype_;         // +244 0x54AA34  Conf_deq cfg low byte: 0 fp16, 1 uint8, 2 int8, 3 int16
    uint8_t  dequant_flag_;      // +245 0x54AA35  Conf_deq cfg high byte
    uint16_t quant_scale_;       // +246 0x54AA36  Conf_quant rs1_val (fp16)
    uint16_t quant_zero_;        // +248 0x54AA38  Conf_quant rs2_val
    uint8_t  dst_dtype_;         // +250 0x54AA3A  Conf_quant cfg low byte (same encoding as src_dtype)
    uint8_t  quant_flag_;        // +251 0x54AA3B  Conf_quant cfg high byte
};

struct PDP1 {
    uint8_t reserved_8_[128];            // +8    unused by the lifted code (vptr is at +0)
    std::vector<PdpWindow> windows_;     // +136  16 slots, created by the singleton constructor
    uint32_t reserved_160_;              // +160  alignment padding before cfg
    Pdp1Config cfg_;                     // +164  configuration registers (see above)

    PDP1();
    // @0x426b20 (PDP11): releases the window vector (and the per-window element vectors).
    virtual ~PDP1();

    // Singleton accessor; PDP1::GetPDP1() is not in the dump (inlined, "singleton lazy-init elided").
    static PDP1* GetPDP1();

    // @0x440470 (PDP12): fills windows[0].data with the window_h x window_w elements of the window
    // at output position (out_row, out_col) of block (dim1, dim2), dequantised from `src`.
    // The always-1 fifth parameter of the demangled signature is not used and dropped.
    void PdpDM(MemAccessor src, int dim1, int dim2, int out_row, int out_col);
    // @0x4406b0 (PDP13): validates the configuration, then runs the whole reduction.
    void PdpRedCompute();
};
static_assert(offsetof(PDP1, windows_) == 136, "PDP1 layout");
static_assert(offsetof(PDP1, cfg_) == 164, "PDP1 layout");
static_assert(offsetof(Pdp1Config, dequant_scale_) == 76, "Pdp1Config layout");
static_assert(offsetof(Pdp1Config, quant_flag_) == 87, "Pdp1Config layout");
static_assert(sizeof(Pdp1Config) == 88, "Pdp1Config layout");
