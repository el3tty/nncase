#pragma once
// MFU (matrix function unit: memset / memcpy / transpose / reduce / sample / (de)quantise helpers) of the
// K230 NPU C-model.
//
// The MFU is a singleton (MFU::GetMFU) whose configuration registers are written by the Mfu*Instruction
// classes.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include "math/numeric_types.h"
#include "engines/memaccessor.h"

struct L1Helper;                                       // L1 tensor view (defined by its owner, not yet lifted)
template <typename T> struct Matrix4;                  // 4-D matrix (defined by its owner, not yet lifted)

struct MFU {
    // ---- configuration registers (byte offsets of the original object in the comments) --------------------
    uint32_t reg0_;                // not accessed by any lifted function (the ctor  does not write it either)
    uint32_t src_addr_;            // GLB address of the source tensor ((addr >> 28) selects _G.GLB[], low 28 bits offset)
    uint32_t dst_addr_;            // GLB address of the destination tensor
    uint32_t aux_addr_;            // Sample: GLB address of the sampling grid
    uint16_t reduce_init_;         // ReduceFun: initial accumulator value (bfloat16 bits)
    uint16_t dim_[4];              // loop extents d0..d3 (set from a shape register)
    uint8_t  reduce_op_;           // ReduceFun: 1 min, 2 add, 3 sub, 4 mul, otherwise max
    uint8_t  reduce_mode_;         // ReduceFun: how many trailing dims are reduced (0 = d3 .. 3 = all)
    uint64_t shape_src_;           // Memcpy / ReduceFun / Sample: source strides (3 halfwords, see GetCHW)
    uint64_t shape_dst_;           // Memcpy / Memset / ReduceFun / Sample: destination strides
    uint8_t  elem16_;              // Memset: 1 = 16-bit elements
    uint64_t trans_shape_src_;     // Trans: source strides
    uint64_t trans_shape_dst_;     // Trans: destination strides
    uint8_t  trans_elem16_;        // Trans: 1 = 16-bit elements
    uint8_t  trans_type_;          // Trans: dimension permutation (0..23)
    uint16_t memset_value_;        // Memset: fill value
    uint64_t shape_grid_;          // Sample: strides of the sampling grid
    uint32_t interp_mode_;         // Sample: non-zero = bilinear, 0 = nearest
    uint32_t dequant_zero_;        // de-quantisation zero point (low byte used)
    uint16_t dequant_scale_;       // de-quantisation scale (bfloat16 bits)
    uint8_t  dequant_signed_;      // source bytes are int8
    uint8_t  dequant_enable_;      // source tensor holds 8-bit quantised data
    uint32_t quant_zero_;          // quantisation zero point (bfloat16 bits in the low half)
    uint16_t quant_scale_;         // quantisation scale (bfloat16 bits)
    uint8_t  quant_signed_;        // produce int8 instead of uint8
    uint8_t  quant_enable_;        // destination tensor holds 8-bit quantised data
    uint16_t dim_sample_[4];       // Sample loop extents (batch, channel, height, width)
    std::ofstream log_[4];         // debug trace streams; log[3] is used by Mfu*Instruction
    uint8_t  busy_;                // set while Memcpy / Memset run (written by the instruction handlers)

    MFU();
    ~MFU();     // verified against asm : plain destructor, no vptr; it only frees the trace-stream/vector storage

    // Returns the object to its freshly constructed state (registers zeroed, trace streams closed). Called when a
    // simulator run ends, because the singleton survives between runs when the simulator is loaded as a DLL.
    void Reset();

    // Singleton accessor.
    static void* GetMFU();

private:
    // Initializes the POD members (called by the constructor and by Reset); log_ is not touched.
    void Init();

public:

    // Splits a shape register (four 16-bit fields) into the three strides used by the tensor addressing
    //   element(i, j, k, l) = l + w * (k + h * (j + c * i)).
    // verified against asm : w = bits 0..15, h = 16..31, c = 32..47.
    static void GetCHW(uint64_t shape, uint32_t & c, uint32_t & h, uint32_t & w);

    // Fills the 16-slot reduction buffer with the identity of operation `op`; slot 15 gets `init`.
    static void ReduceInit(FP24::fp24 * buf, FP24::fp24 init, uint8_t op);

    // BF16 piece-wise linear activation over an L1 tile.
    // `this` is unused by the original.
    void Activate(Matrix4<BF16::bfloat16> & params, L1Helper & psum, bool is_signed);

    // fp16 -> int8/uint8/int16 quantisation of one value.
    // out = clamp(round(value * scale * 2^shift + zero_point)); mode 1 = uint8, 2 = int8, other = int16.
    static void quant_new(FP16::fp16 const & value, FP16::fp16 const & scale, FP16::fp16 const & zero_point,
                          uint8_t const & mode, signed char shift, short & out);
    // int -> fp16 de-quantisation: out = fp16(q - zero_point) * 2^-shift * scale.
    static void dequant_new(short const & q, FP16::fp16 const & scale, short const & zero_point,
                            uint8_t const & mode, signed char shift, FP16::fp16 & out);
    // Quantise `value` and store it at element `index` of `dst` (mode 0 = raw fp16, 3 = int16, else 8 bit).
    static void quant_new_MemSt(int index, MemAccessor dst, FP16::fp16 scale, FP16::fp16 zero_point,
                                uint8_t mode, uint8_t shift, FP16::fp16 & value);
    // Load element `index` of `src` (mode 1 uint8, 2 int8, 3 int16, else fp16) and de-quantise it.
    static void dequant_new_MemAt(int index, MemAccessor src, FP16::fp16 scale, short zero_point,
                                  uint8_t mode, uint8_t shift, FP16::fp16 & out);

    // bf16 -> 8-bit quantisation: out = clamp(round(x * scale + zero_point)).
    static void quant(BF16::bfloat16 const & x, BF16::bfloat16 const & scale, uint32_t const & zero_point,
                      uint8_t const & is_signed, uint8_t & out);
    // 8-bit -> bf16 de-quantisation: out = bf16(q - zero_point) * scale.
    static void dequant(uint8_t const & q, BF16::bfloat16 const & scale, uint8_t const & zero_point,
                        uint8_t const & is_signed, BF16::bfloat16 & out);

    void Memset();
    void Memcpy();
    void Trans();
    void ReduceFun(uint64_t * unused);
    void Sample();
};
