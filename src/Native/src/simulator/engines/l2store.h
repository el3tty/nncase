#pragma once
// L2 store unit (GLB -> DDR) of the K230 NPU C-model.
//
// The singleton is filled by the L2Store*Instruction classes (insn_l2.cpp).
#include <cstddef>
#include <cstdint>

struct L2Store {
    // Destination (DDR) tensor layout, in elements: planes per tensor, rows per plane, row pitch.
    uint32_t ddr_dim0_;
    uint32_t ddr_dim1_;
    uint32_t ddr_pitch_;
    uint32_t ddr_dim3_;        // always 0
    // Source (GLB) tensor layout.
    uint32_t glb_dim0_;
    uint32_t glb_dim1_;
    uint32_t glb_pitch_;
    uint32_t glb_dim3_;        // always 0
    // Data format (mode == 0x201: float16 in GLB -> float32 in DDR; mode0 != 0: 16-bit elements;
    // otherwise 8-bit elements).
    uint8_t mode0_;
    uint8_t mode1_;
    uint8_t * ddr_ptr_;        // destination pointer (_G.DDR + ddr_offset)
    uint8_t * glb_ptr_;        // source pointer inside the GLB bank
    // Extent of the transfer.
    uint32_t batch_count_;     // number of whole tensors
    uint32_t plane_count_;     // planes per tensor
    uint32_t row_count_;       // rows per plane
    uint32_t row_len_;         // elements per row
    uint32_t ddr_offset_;
    uint32_t glb_mmu_addr_;    // GLB offset + 32 * MMU base of the bank
    uint32_t glb_bank_;

    uint16_t Mode() const { return (uint16_t)(mode0_ | (mode1_ << 8)); }

    // Copies a 4-D block GLB -> DDR with optional float16 -> float32 conversion.
    // Returns batch_count (the original returned the leftover register value).
    int64_t Store();
    // Dumps the DDR write data / address to store_ddr_wdata / store_ddr_waddr.
    int64_t PrintStoreDDRCheckPoint(uint32_t rel_offset, uint32_t num_bytes);
    // Dumps the GLB read address / data to store_glb_raddr / store_glb_rdata.
    int64_t PrintLoadGlbCheckPoint(uint32_t rel_offset, uint32_t num_bytes);
};

