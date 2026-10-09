#pragma once
// ACT0: activation unit (piecewise-linear "line fit" on the PSUM, then quantisation) of the K230 NPU
// C-model. Lifted from IDA/Hex-Rays output (Act01..Act06.cpp).
#include <cstddef>
#include <cstdint>
#include <memory>
#include "math/numeric_types.h"

struct DmLoadAct0;        // see dm.h
struct DmStoreOf;         // see dm.h
struct L1Helper;          // L1 tensor view (defined by its owner, not yet lifted)
struct Tensor4DHelper;    // 4-D tensor view of a DM store (defined by its owner, not yet lifted)
template <typename T> struct Matrix4;   // 4-D matrix (defined by its owner, not yet lifted)

// Output data type of the activation.
enum ActOutputType : int {
    kActUInt8 = 0,    // saturated to 0..255
    kActInt8  = 1,    // truncated and saturated to -127..127 (verified against asm: .rodata 0x522060 / 5220A0)
    kActFp16  = 2,    // raw fp16 bits
    kActInt16 = 3     // truncated and saturated to -32767..32767 (Tensor4DHelper overload only)
};

// Descriptor created by Act0::GetAct0Compute() and completed by Act0ComputeInstruction.
// TODO(layout): Act0ComputeInstruction fills +24..+36 through raw offsets.
struct Act0Compute {
    uint32_t batch_ = 0;       // +0   Act0 + 0x40000
    uint32_t channels_ = 0;    // +4   Act0 + 0x40004: number of channels (rows of the parameter matrix)
    uint32_t height_ = 0;      // +8   Act0 + 0x40008
    uint32_t width_ = 0;       // +12  Act0 + 0x4000C
    uint32_t shift_ = 0;       // +16  Act0 + 0x40010: PSUM scale, the PSUM is multiplied by 2^-shift
    uint32_t engine_ = 0;      // +20  Act0 + 0x40014 (0 Conv2D, 1 PDP0)
    uint32_t out_base_ = 0;    // +24  Act0ComputeInstruction raddr_d_val: byte offset in PSUM_L1 of the L1 output
    uint8_t  out_route_ = 0;   // +28  Act0ComputeInstruction target: 0 = PSUM_L1 only, 1 = DM store only, else both
    uint32_t out_type_ = 0;    // +32  Act0ComputeInstruction dest_datatype: ActOutputType
    uint8_t  per_channel_ = 0; // +36  Act0ComputeInstruction is_by_channel: 1 = one parameter row per channel
    uint32_t* psum_ = nullptr; // +40  PSUM buffer to read (Act0 or Act0 + 0x20000)
};

// The Act0 singleton (global Act0_act0). It starts with two PSUM buffers followed by the
// configuration written by Act0Src1ConfInstruction (shape, shift, engine) and Act0ComputeInstruction.
// Layout verified against asm @0x446ac0 (GetAct0Compute copies +0x40000..+0x40024) and
// @0x422ec0 (Act0Src1ConfInstruction::operation, stores at 0x5CC760..0x5CC774 = Act0 + 0x40000..0x40014).
struct Act0 {
    uint32_t psum0_[0x8000];   // +0        first PSUM buffer (128 KiB)
    uint32_t psum1_[0x8000];   // +0x20000  second PSUM buffer
    uint32_t batch_;           // +0x40000  Act0Src1Conf rshape bits 63..48
    uint32_t channels_;        // +0x40004  rshape bits 47..32: number of channels (rows of the parameter matrix)
    uint32_t height_;          // +0x40008  rshape bits 31..16
    uint32_t width_;           // +0x4000C  rshape bits 15..0
    uint32_t shift_;           // +0x40010  Act0Src1Conf src1_param: the PSUM is multiplied by 2^-shift
    uint32_t engine_;          // +0x40014  Act0Src1Conf channel: 0 = Conv2D (reads psum0), 1 = PDP0 (reads psum1)
    uint32_t out_base_;        // +0x40018  Act0Compute only (copied to Act0Compute, then overwritten)
    uint8_t  out_route_;       // +0x4001C  "
    uint8_t  pad_1d_[3];       // +0x4001D
    uint32_t out_type_;        // +0x40020  "
    uint8_t  per_channel_;     // +0x40024  "

    // @0x446ac0 (Source 1)
    std::shared_ptr<Act0Compute> GetAct0Compute() const;

    // @0x446b80 (Source 2): line-fit activation of `psum` into an L1 buffer.
    static void Activate(Matrix4<FP16::fp16> & params, L1Helper & psum, L1Helper & out, uint32_t shift, ActOutputType type);
    // @0x449460 (Source 3)
    static void PrintActDmWriteCKP(L1Helper & shape, Tensor4DHelper & out, ActOutputType type);
    // @0x449620 (Source 4): line-fit activation of `psum` into a DM store tensor.
    static void Activate(Matrix4<FP16::fp16> & params, L1Helper & psum, Tensor4DHelper & out, uint32_t shift, ActOutputType type);
    // @0x44c650 (Source 5)
    static void Compute(std::shared_ptr<Act0Compute> compute, std::shared_ptr<DmLoadAct0> load, std::shared_ptr<DmStoreOf> store);
    // @0x44d030 (Source 6)
    static void PrintDmOfCKP(L1Helper & shape, Tensor4DHelper & out, ActOutputType type, int address);
};
