#pragma once
// MFU (matrix function unit: memset / memcpy / transpose / reduce / sample / (de)quantise helpers) of the
// K230 NPU C-model.  Lifted from IDA/Hex-Rays output (MFU1..MFU15 in sources/).
//
// The MFU is a singleton (MFU::GetMFU()) whose configuration registers are written by the Mfu*Instruction
// classes.  In the original binary the instance lives at 0x54BB00 and the instruction handlers poke it by
// raw byte offset.
// The members below carry those offsets; each is checked by a static_assert in mfu.cpp.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include "math/numeric_types.h"
#include "engines/memaccessor.h"

struct L1Helper;                                       // L1 tensor view (defined by its owner, not yet lifted)
template <typename T> struct Matrix4;                  // 4-D matrix (defined by its owner, not yet lifted)

struct MFU {
    // ---- configuration registers (byte offsets of the original object in the comments) --------------------
    uint32_t reg0_;                // +0    not accessed by any lifted function (the ctor @0x445da0 does not write it either)
    uint32_t src_addr_;            // +4    GLB address of the source tensor ((addr >> 28) selects g_GLB[], low 28 bits offset)
    uint32_t dst_addr_;            // +8    GLB address of the destination tensor
    uint32_t aux_addr_;            // +12   Sample: GLB address of the sampling grid
    uint16_t reduce_init_;         // +16   ReduceFun: initial accumulator value (bfloat16 bits)
    uint8_t  pad18_[14];           // +18
    uint16_t dim_[4];              // +32   loop extents d0..d3 (set from a shape register)
    uint8_t  reduce_op_;           // +40   ReduceFun: 1 min, 2 add, 3 sub, 4 mul, otherwise max
    uint8_t  reduce_mode_;         // +41   ReduceFun: how many trailing dims are reduced (0 = d3 .. 3 = all)
    uint8_t  pad42_[14];           // +42
    uint64_t shape_src_;           // +56   Memcpy / ReduceFun / Sample: source strides (3 halfwords, see GetCHW)
    uint64_t shape_dst_;           // +64   Memcpy / Memset / ReduceFun / Sample: destination strides
    uint8_t  elem16_;              // +72   Memset: 1 = 16-bit elements
    uint8_t  pad73_[7];            // +73
    uint64_t trans_shape_src_;     // +80   Trans: source strides
    uint64_t trans_shape_dst_;     // +88   Trans: destination strides
    uint8_t  trans_elem16_;        // +96   Trans: 1 = 16-bit elements
    uint8_t  trans_type_;          // +97   Trans: dimension permutation (0..23)
    uint16_t memset_value_;        // +98   Memset: fill value
    uint8_t  pad100_[4];           // +100
    uint64_t shape_grid_;          // +104  Sample: strides of the sampling grid
    uint32_t interp_mode_;         // +112  Sample: non-zero = bilinear, 0 = nearest
    uint8_t  pad116_[4];           // +116
    uint32_t dequant_zero_;        // +120  de-quantisation zero point (low byte used)
    uint16_t dequant_scale_;       // +124  de-quantisation scale (bfloat16 bits)
    uint8_t  dequant_signed_;      // +126  source bytes are int8
    uint8_t  dequant_enable_;      // +127  source tensor holds 8-bit quantised data
    uint32_t quant_zero_;          // +128  quantisation zero point (bfloat16 bits in the low half)
    uint16_t quant_scale_;         // +132  quantisation scale (bfloat16 bits)
    uint8_t  quant_signed_;        // +134  produce int8 instead of uint8
    uint8_t  quant_enable_;        // +135  destination tensor holds 8-bit quantised data
    uint8_t  pad136_[8];           // +136
    uint16_t dim_sample_[4];       // +144  Sample loop extents (batch, channel, height, width)
    std::ofstream log_[4];         // +152  debug trace streams (+152, +664, +1176, +1688); log[3] is used by Mfu*Instruction
    uint64_t string_slot_;         // +2200 an empty COW std::string in the original (never touched)
    uint8_t  busy_;                // +2208 set while Memcpy / Memset run (written by the instruction handlers)
    uint8_t  reserved_[135];       // +2209 unused containers of the original (all null); object ends at +2344

    MFU();
    ~MFU();     // verified against asm @0x426e80 (MFUD2): plain destructor, no vptr; it only frees the trace-stream/vector storage

    // Singleton accessor.  @0x445da0 (MFU15)
    // Returns void* because callers (Mfu*Instruction) address the object by raw byte offset (TODO(layout)).
    static void* GetMFU();

    // Splits a shape register (four 16-bit fields) into the three strides used by the tensor addressing
    //   element(i, j, k, l) = l + w * (k + h * (j + c * i)).
    // verified against asm @0x437940 (_ZN3MFU6GetCHWEmbPjS0_S0_.constprop.0): w = bits 0..15, h = 16..31, c = 32..47.
    static void GetCHW(uint64_t shape, uint32_t & c, uint32_t & h, uint32_t & w);

    // Fills the 16-slot reduction buffer with the identity of operation `op`; slot 15 gets `init`.  @0x4382d0 (MFU2)
    static void ReduceInit(FP24::fp24 * buf, FP24::fp24 init, uint8_t op);

    // BF16 piece-wise linear activation over an L1 tile.  @0x438fd0 (MFU3)
    // `this` is unused by the original.
    void Activate(Matrix4<BF16::bfloat16> & params, L1Helper & psum, bool is_signed);

    // fp16 -> int8/uint8/int16 quantisation of one value.  @0x439480 (MFU4)
    // out = clamp(round(value * scale * 2^shift + zero_point)); mode 1 = uint8, 2 = int8, other = int16.
    static void quant_new(FP16::fp16 const & value, FP16::fp16 const & scale, FP16::fp16 const & zero_point,
                          uint8_t const & mode, signed char shift, short & out);
    // int -> fp16 de-quantisation: out = fp16(q - zero_point) * 2^-shift * scale.  @0x4395a0 (MFU5)
    static void dequant_new(short const & q, FP16::fp16 const & scale, short const & zero_point,
                            uint8_t const & mode, signed char shift, FP16::fp16 & out);
    // Quantise `value` and store it at element `index` of `dst` (mode 0 = raw fp16, 3 = int16, else 8 bit).  @0x439610 (MFU6)
    static void quant_new_MemSt(int index, MemAccessor dst, FP16::fp16 scale, FP16::fp16 zero_point,
                                uint8_t mode, uint8_t shift, FP16::fp16 & value);
    // Load element `index` of `src` (mode 1 uint8, 2 int8, 3 int16, else fp16) and de-quantise it.  @0x43d970 (MFU14)
    static void dequant_new_MemAt(int index, MemAccessor src, FP16::fp16 scale, short zero_point,
                                  uint8_t mode, uint8_t shift, FP16::fp16 & out);

    // bf16 -> 8-bit quantisation: out = clamp(round(x * scale + zero_point)).  @0x4396d0 (MFU7)
    static void quant(BF16::bfloat16 const & x, BF16::bfloat16 const & scale, uint32_t const & zero_point,
                      uint8_t const & is_signed, uint8_t & out);
    // 8-bit -> bf16 de-quantisation: out = bf16(q - zero_point) * scale.  @0x4398a0 (MFU8)
    static void dequant(uint8_t const & q, BF16::bfloat16 const & scale, uint8_t const & zero_point,
                        uint8_t const & is_signed, BF16::bfloat16 & out);

    void Memset();                // @0x439a00 (MFU9)
    void Memcpy();                // @0x439c00 (MFU10)
    void Trans();                 // @0x439e30 (MFU11)
    void ReduceFun(uint64_t * unused);   // @0x43aae0 (MFU12)
    void Sample();                // @0x43be70 (MFU13)
};
