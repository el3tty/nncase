#pragma once
// DM (data-memory / load-store front end) of the K230 NPU C-model.
// Lifted from IDA/Hex-Rays output (Dm1..Dm7.cpp).
//
// The Dm singleton collects the operands written by the DmLoad*/DmStoreOf*/Conf instructions.
// Its Get* methods snapshot the relevant part of it into a shared descriptor which the
// Conv2D / PDP0 engines queue and consume later (see Act0::Compute, PDP0::Compute).
// The descriptors below are plain structs: the original shared_ptr control blocks put the
// object 16 bytes behind the block start, so the offsets in the comments are object offsets.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

struct L1Helper;          // L1 tensor view (defined by its owner, not yet lifted)
enum L2DataType : int;

// Weight-load descriptor created by Dm::GetLoadW().
// TODO(layout): pdp0.cpp / conv2d.cpp read it through raw views {unused_0@0, signed_flag@4, weights@8, zero_points@16}.
struct DmLoadW {
    uint32_t line_bytes_ = 0;            // +0   Dm+12 (loadw_len)
    uint32_t deq_mode_ = 0;              // +4   Dm+16 (loadw_deq_mode); 1 = unsigned weights (PDP0::Compute)
    uint8_t* weights_ = nullptr;         // +8   malloc'ed copy of line_bytes * lines bytes taken from the GLB
    const uint8_t* zero_points_ = nullptr;  // +16  Dm+32: GLB pointer of the second source (not copied)
    uint8_t  use_pdp0_ = 0;              // +24  Dm+40
    DmLoadW() = default;
    DmLoadW(const DmLoadW&) = delete;
    DmLoadW& operator=(const DmLoadW&) = delete;
    // verified against asm @0x44e450 (_M_dispose): free(weights) if non-null.
    ~DmLoadW() { std::free(weights_); }
};

// Activation-parameter load descriptor created by Dm::GetLoadAct0().
struct DmLoadAct0 {
    const int16_t* params_ = nullptr;    // +0   Dm+96: GLB pointer to the fp16 parameter table (7 per channel)
    uint8_t use_pdp0_ = 0;               // +8   Dm+104
    uint8_t load_flag_ = 0;              // +9   Dm+105
};

// Output-feature store descriptor created by Dm::GetStoreOf().
struct DmStoreOf {
    uint32_t tile_shape_[4] = {};        // +0   Dm+108..123: store shape (dim0 = C, dim1 = H, dim2 = W, dim3 = 0)
    uint32_t mode_ = 0;                  // +16  Dm+124
    uint32_t unused_20_ = 0;                   // +20  (always 0)
    uint8_t* dst_ = nullptr;             // +24  Dm+128: GLB destination pointer
    uint32_t full_shape_[4] = {};        // +32  Dm+136..151: shape of the whole destination tensor
    uint32_t mmu_addr_ = 0;              // +48  Dm+152
    uint8_t  use_pdp0_ = 0;              // +52  Dm+156
};

// L1 (input-feature) load descriptor created by Dm::GetDmLoadL1() and consumed by Dm::LoadL1().
struct DmLoadL1 {
    uint8_t* if_snapshot_ = nullptr;     // +0   malloc(0x6000) copy of _G.IF_L1 taken at creation time (never read back here)
    uint32_t shape_[4] = {};             // +8   Dm+44..59 (L1 conf shape; shape[1], shape[2] are source pitches)
    uint32_t mode_ = 0;                  // +24  Dm+60: 2 = 16-bit elements (split into two byte planes)
    uint32_t dims_[4] = {};              // +28  Dm+64..79 (dims[1] = n_y, dims[2] = n_z, dims[3] = n_x; dims[0] unused)
    uint32_t pad44_ = 0;                   // +44  (padding)
    const uint8_t* src_ = nullptr;       // +48  Dm+80: GLB source pointer
    uint32_t layout_ = 0;                // +56  Dm+88: <= 1 contiguous; otherwise low16 = count, high16 = step
    uint32_t pad60_ = 0;                   // +60  (padding)
    DmLoadL1() = default;
    DmLoadL1(const DmLoadL1&) = delete;
    DmLoadL1& operator=(const DmLoadL1&) = delete;
    // verified against asm @0x44e440 (_M_dispose): free(if_snapshot).
    ~DmLoadL1() { std::free(if_snapshot_); }
};

// The Dm singleton. Fields are named by the instruction that writes them; the offsets are the ones
// the other instruction files still use through raw pointers (TODO(layout)).
struct Dm {
    uint32_t unused_0_;                    // +0
    uint32_t conf_a_;                // +4   DmLoadWConf
    uint32_t conf_b_;                // +8   DmLoadWConf
    uint32_t loadw_len_;             // +12  DmLoadWConf: bytes per weight line
    uint32_t loadw_deq_mode_;        // +16  DmLoadWConf_deq
    uint32_t unused_20_;                   // +20
    const uint8_t* loadw_src0_;      // +24  DmLoadW: GLB pointer 0 (weights)
    const uint8_t* loadw_src1_;      // +32  DmLoadW: GLB pointer 1 (zero points)
    uint8_t  loadw_use_pdp0_;        // +40  DmLoadW
    uint8_t  pad41_;                   // +41
    uint16_t loadw_lines_;           // +42  DmLoadW: number of lines
    uint32_t l1_shape_[4];           // +44  DmLoadL1Conf
    uint32_t l1_mode_;               // +60  DmLoadL1Conf
    uint32_t l1_dims_[4];            // +64  DmLoadL1: four 16-bit dims of the shape register (verified against asm @0x423140)
    const uint8_t* l1_src_;          // +80  DmLoadL1: GLB pointer
    uint32_t l1_layout_;             // +88  DmLoadL1
    uint32_t pad92_;                   // +92
    const int16_t* loadact0_src_;    // +96  DmLoadAct0: GLB pointer
    uint8_t  loadact0_use_pdp0_;     // +104 DmLoadAct0
    uint8_t  loadact0_flag_;         // +105 DmLoadAct0
    uint8_t  pad106_[2];               // +106
    uint32_t of_shape_[4];           // +108 DmStoreOfConf
    uint32_t of_mode_;               // +124 DmStoreOfConf
    uint8_t* of_dst_;                // +128 DmStoreOf: GLB pointer
    uint32_t of_full_shape_[4];      // +136 DmStoreOf
    uint32_t of_mmu_addr_;           // +152 DmStoreOf
    uint8_t  of_use_pdp0_;           // +156 DmStoreOf

    // @0x44d700 (Source 1). Returns the singleton as raw bytes because the instruction files
    // still address it by byte offset. TODO(layout)
    static char* GetDm();
    // @0x44d790 (Source 2)
    static void PrintDmIfAndPuIfInCkp(L1Helper & helper, L2DataType dataType);
    // @0x44d9c0 .. @0x44db70 (Sources 3-6): snapshot the current operands into a queueable descriptor
    std::shared_ptr<DmLoadW>   GetLoadW() const;
    std::shared_ptr<DmStoreOf> GetStoreOf() const;
    std::shared_ptr<DmLoadAct0> GetLoadAct0() const;
    std::shared_ptr<DmLoadL1>  GetDmLoadL1() const;
    // @0x44dc20 (Source 7)
    static void LoadL1(std::shared_ptr<DmLoadL1> load);
};

