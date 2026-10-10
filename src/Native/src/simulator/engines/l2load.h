#pragma once
// L2 load unit (DDR -> GLB) of the K230 NPU C-model.
//
// The singleton is filled by the L2Load*Instruction classes (insn_l2.cpp).
#include <cstddef>
#include <cstdint>

struct L2Load {
    // Destination (GLB) tensor layout, in elements: planes per tensor, rows per plane, row pitch.
    uint32_t dst_dim0_;
    uint32_t dst_dim1_;
    uint32_t dst_pitch_;
    uint32_t dst_dim3_;        // always 0
    // Source (DDR) tensor layout.
    uint32_t src_dim0_;
    uint32_t src_dim1_;
    uint32_t src_pitch_;
    uint32_t src_dim3_;        // always 0
    // Extent of the transfer.
    uint32_t batch_count_;     // number of whole tensors
    uint32_t plane_count_;     // planes per tensor
    uint32_t row_count_;       // rows per plane
    uint32_t row_len_;         // elements per row
    // LoadW configuration.
    uint32_t wconf_lo_;        // low half of the 64-bit wconf operand (unused by LoadW)
    uint32_t weight_count_;    // high half of wconf: number of weight elements
    uint32_t compressed_;      // non-zero: weights are stored in the 16-byte-block bitmap format
    uint32_t row_len_m1_;      // elements per destination row group, minus one
    uint8_t * ddr_ptr_;        // source pointer (_G.DDR + ddr_offset)
    uint8_t * glb_ptr_;        // destination pointer inside the GLB bank
    // Data format of Load (mode == 0x201: float32 in DDR -> float16 in GLB; mode0 != 0: 16-bit
    // elements; otherwise 8-bit elements).
    uint8_t mode0_;
    uint8_t mode1_;
    // Data format of LoadW: w_mode1 = element size code (0: 1 B, 1: 2 B, 2: 4 B, 3: packed 4-bit,
    // 4: packed 6-bit, 5: 2 B); w_mode0/w_mode1 == 1/2 converts float32 to float16.
    uint8_t w_mode0_;
    uint8_t w_mode1_;
    uint32_t ddr_offset_;
    uint32_t glb_mmu_addr_;    // GLB offset + 32 * MMU base of the bank
    uint32_t glb_bank_;

    uint16_t Mode() const { return (uint16_t)(mode0_ | (mode1_ << 8)); }
    uint16_t WMode() const { return (uint16_t)(w_mode0_ | (w_mode1_ << 8)); }

    // Copies a 4-D block DDR -> GLB with optional float32 -> float16 conversion.
    // Returns plane_count (the original returned the leftover register value).
    int64_t Load();
    // Dumps the DDR read data/addresses to the load_ddr_rdata/load_ddr_raddr check files.
    int64_t PrintLoadDDRCheckPoint(uint32_t rel_offset, uint32_t num_bytes);
    // Dumps the 32-byte GLB write beats to load_glb_write.chk.
    int64_t PrintWriteGlbCheckPoint(uint32_t rel_offset, uint32_t num_bytes, uint32_t row_len_m1, int elem_bytes);
    // Expands the 16-byte-block bitmap format (2 byte mask + non-zero bytes) into dst.
    uint8_t* Decompress(uint8_t * dst, uint8_t * src, int dst_len);
    // Weight load: DDR -> GLB with unpacking/decompression.
    void LoadW();
};

