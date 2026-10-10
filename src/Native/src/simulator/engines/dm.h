#pragma once
// DM (data-memory / load-store front end) of the K230 NPU C-model.
//
// The Dm singleton collects the operands written by the DmLoad*/DmStoreOf*/Conf instructions.
// Its Get* methods snapshot the relevant part of it into a shared descriptor which the
// Conv2D / PDP0 engines queue and consume later (see Act0::Compute, PDP0::Compute).
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

struct L1Helper;          // L1 tensor view (defined by its owner, not yet lifted)
enum L2DataType : int;

// Weight-load descriptor created by Dm::GetLoadW.
struct DmLoadW {
    uint32_t line_bytes_ = 0;            // (loadw_len)
    uint32_t deq_mode_ = 0;              // (loadw_deq_mode); 1 = unsigned weights (PDP0::Compute)
    uint8_t* weights_ = nullptr;         // malloc'ed copy of line_bytes * lines bytes taken from the GLB
    const uint8_t* zero_points_ = nullptr;  // GLB pointer of the second source (not copied)
    uint8_t  use_pdp0_ = 0;
    DmLoadW() = default;
    DmLoadW(const DmLoadW&) = delete;
    DmLoadW& operator=(const DmLoadW&) = delete;
    // verified against asm @0x44e450 (_M_dispose): free(weights) if non-null.
    ~DmLoadW() { std::free(weights_); }
};

// Activation-parameter load descriptor created by Dm::GetLoadAct0.
struct DmLoadAct0 {
    const int16_t* params_ = nullptr;    // GLB pointer to the fp16 parameter table (7 per channel)
    uint8_t use_pdp0_ = 0;
    uint8_t load_flag_ = 0;
};

// Output-feature store descriptor created by Dm::GetStoreOf.
struct DmStoreOf {
    uint32_t tile_shape_[4] = {};        // store shape (dim0 = C, dim1 = H, dim2 = W, dim3 = 0)
    uint32_t mode_ = 0;
    uint8_t* dst_ = nullptr;             // GLB destination pointer
    uint32_t full_shape_[4] = {};        // shape of the whole destination tensor
    uint32_t mmu_addr_ = 0;
    uint8_t  use_pdp0_ = 0;
};

// L1 (input-feature) load descriptor created by Dm::GetDmLoadL1 and consumed by Dm::LoadL1.
struct DmLoadL1 {
    uint8_t* if_snapshot_ = nullptr;     // malloc(0x6000) copy of _G.IF_L1 taken at creation time (never read back here)
    uint32_t shape_[4] = {};             // (L1 conf shape; shape[1], shape[2] are source pitches)
    uint32_t mode_ = 0;                  // 2 = 16-bit elements (split into two byte planes)
    uint32_t dims_[4] = {};              // (dims[1] = n_y, dims[2] = n_z, dims[3] = n_x; dims[0] unused)
    const uint8_t* src_ = nullptr;       // GLB source pointer
    uint32_t layout_ = 0;                // <= 1 contiguous; otherwise low16 = count, high16 = step
    DmLoadL1() = default;
    DmLoadL1(const DmLoadL1&) = delete;
    DmLoadL1& operator=(const DmLoadL1&) = delete;
    // verified against asm @0x44e440 (_M_dispose): free(if_snapshot).
    ~DmLoadL1() { std::free(if_snapshot_); }
};

// The Dm singleton. Fields are named by the instruction that writes them; the byte offsets in the comments are
// those of the original binary (the layout here no longer reproduces them).
struct Dm {
    uint32_t conf_a_;                // DmLoadWConf
    uint32_t conf_b_;                // DmLoadWConf
    uint32_t loadw_len_;             // DmLoadWConf: bytes per weight line
    uint32_t loadw_deq_mode_;        // DmLoadWConf_deq
    const uint8_t* loadw_src0_;      // DmLoadW: GLB pointer 0 (weights)
    const uint8_t* loadw_src1_;      // DmLoadW: GLB pointer 1 (zero points)
    uint8_t  loadw_use_pdp0_;        // DmLoadW
    uint16_t loadw_lines_;           // DmLoadW: number of lines
    uint32_t l1_shape_[4];           // DmLoadL1Conf
    uint32_t l1_mode_;               // DmLoadL1Conf
    uint32_t l1_dims_[4];            // DmLoadL1: four 16-bit dims of the shape register (verified against asm @0x423140)
    const uint8_t* l1_src_;          // DmLoadL1: GLB pointer
    uint32_t l1_layout_;             // DmLoadL1
    const int16_t* loadact0_src_;    // DmLoadAct0: GLB pointer
    uint8_t  loadact0_use_pdp0_;     // DmLoadAct0
    uint8_t  loadact0_flag_;         // DmLoadAct0
    uint32_t of_shape_[4];           // DmStoreOfConf
    uint32_t of_mode_;               // DmStoreOfConf
    uint8_t* of_dst_;                // DmStoreOf: GLB pointer
    uint32_t of_full_shape_[4];      // DmStoreOf
    uint32_t of_mmu_addr_;           // DmStoreOf
    uint8_t  of_use_pdp0_;           // DmStoreOf

    // The singleton lives in the raw global block _G.Dm_dm.
    static Dm* GetDm();
    static void PrintDmIfAndPuIfInCkp(L1Helper & helper, L2DataType dataType);
    // Snapshot the current operands into a queueable descriptor
    std::shared_ptr<DmLoadW>   GetLoadW() const;
    std::shared_ptr<DmStoreOf> GetStoreOf() const;
    std::shared_ptr<DmLoadAct0> GetLoadAct0() const;
    std::shared_ptr<DmLoadL1>  GetDmLoadL1() const;
    static void LoadL1(std::shared_ptr<DmLoadL1> load);
};

