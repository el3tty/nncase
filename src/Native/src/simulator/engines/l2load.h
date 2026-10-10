#pragma once
// L2 load unit (DDR -> GLB) of the K230 NPU C-model.
// Lifted from IDA/Hex-Rays output (L2Load1..L2Load5 in sources/).
//
// The L2Load singleton is the raw word array `L2Load_L2LoadInst` (0x54C440 in the original binary); the
// instruction classes (insn_l2.cpp) cast it to L2Load* and fill the members.  The object layout below must therefore match the
// original byte offsets exactly.
//
// Address map of the singleton (unk_ symbol -> member), all derived from the stores done by the
// instruction classes:
//   L2Load_L2LoadInst[0]  0x54C440  +0   dst_dim0       (L2LoadConf, shape0 >> 32)
//   0x54C444  +4   dst_dim1       (L2LoadConf)
//   0x54C448  +8   dst_pitch      (L2LoadConf)
//   0x54C44C  +12  dst_dim3       (L2LoadConf, cleared)
//   0x54C450  +16  src_dim0       (L2LoadConf, shape1 >> 32)
//   0x54C454  +20  src_dim1       (L2LoadConf)
//   0x54C458  +24  src_pitch      (L2LoadConf)
//   0x54C45C  +28  src_dim3       (L2LoadConf, cleared)
//   0x54C460  +32  batch_count    (L2LoadInstruction, shape >> 48)
//   0x54C464  +36  plane_count    (L2LoadInstruction, shape >> 32)
//   0x54C468  +40  row_count      (L2LoadInstruction, shape >> 16)
//   0x54C46C  +44  row_len        (L2LoadInstruction, shape & 0xFFFF)
//   0x54C470  +48  wconf (64 bit; low half = +48, high half = weight_count at +52)
//   0x54C478  +56  compressed     (L2LoadWConf, load_flag != 0)
//   0x54C47C  +60  row_len_m1     (L2LoadW, len_val)
//   0x54C480  +64  ddr_ptr        (L2Load / L2LoadW, _G.DDR + ddr_offset)
//   0x54C488  +72  glb_ptr        (L2Load / L2LoadW, _G.GLB[bank] + (glb_addr & 0xFFFFFFF))
//   0x54C490  +80  mode0          (L2LoadConf)
//   0x54C491  +81  mode1          (L2LoadConf)
//   0x54C492  +82  w_mode0        (L2LoadWConf)
//   0x54C493  +83  w_mode1        (L2LoadWConf)
//   0x54C494  +84  ddr_offset     (L2Load / L2LoadW)
//   0x54C498  +88  glb_mmu_addr   (L2Load / L2LoadW)
//   0x54C49C  +92  glb_bank       (L2Load / L2LoadW)
#include <cstddef>
#include <cstdint>

struct L2Load {
    // Destination (GLB) tensor layout, in elements: planes per tensor, rows per plane, row pitch.
    uint32_t dst_dim0_;        // +0
    uint32_t dst_dim1_;        // +4
    uint32_t dst_pitch_;       // +8
    uint32_t dst_dim3_;        // +12  always 0
    // Source (DDR) tensor layout.
    uint32_t src_dim0_;        // +16
    uint32_t src_dim1_;        // +20
    uint32_t src_pitch_;       // +24
    uint32_t src_dim3_;        // +28  always 0
    // Extent of the transfer.
    uint32_t batch_count_;     // +32  number of whole tensors
    uint32_t plane_count_;     // +36  planes per tensor
    uint32_t row_count_;       // +40  rows per plane
    uint32_t row_len_;         // +44  elements per row
    // LoadW configuration.
    uint32_t wconf_lo_;        // +48  low half of the 64-bit wconf operand (unused by LoadW)
    uint32_t weight_count_;    // +52  high half of wconf: number of weight elements
    uint32_t compressed_;      // +56  non-zero: weights are stored in the 16-byte-block bitmap format
    uint32_t row_len_m1_;      // +60  elements per destination row group, minus one
    uint8_t * ddr_ptr_;        // +64  source pointer (_G.DDR + ddr_offset)
    uint8_t * glb_ptr_;        // +72  destination pointer inside the GLB bank
    // Data format of Load (mode == 0x201: float32 in DDR -> float16 in GLB; mode0 != 0: 16-bit
    // elements; otherwise 8-bit elements).
    uint8_t mode0_;            // +80
    uint8_t mode1_;            // +81
    // Data format of LoadW: w_mode1 = element size code (0: 1 B, 1: 2 B, 2: 4 B, 3: packed 4-bit,
    // 4: packed 6-bit, 5: 2 B); w_mode0/w_mode1 == 1/2 converts float32 to float16.
    uint8_t w_mode0_;          // +82
    uint8_t w_mode1_;          // +83
    uint32_t ddr_offset_;      // +84
    uint32_t glb_mmu_addr_;    // +88  GLB offset + 32 * MMU base of the bank
    uint32_t glb_bank_;        // +92

    uint16_t Mode() const { return (uint16_t)(mode0_ | (mode1_ << 8)); }
    uint16_t WMode() const { return (uint16_t)(w_mode0_ | (w_mode1_ << 8)); }

    // Copies a 4-D block DDR -> GLB with optional float32 -> float16 conversion.
    // Returns plane_count (the original returned the leftover register value).  @0x434c00 (L2Load1)
    int64_t Load();
    // Dumps the DDR read data/addresses to the load_ddr_rdata/load_ddr_raddr check files.  @0x434fe0 (L2Load2)
    int64_t PrintLoadDDRCheckPoint(uint32_t rel_offset, uint32_t num_bytes);
    // Dumps the 32-byte GLB write beats to load_glb_write.chk.  @0x435410 (L2Load3)
    int64_t PrintWriteGlbCheckPoint(uint32_t rel_offset, uint32_t num_bytes, uint32_t row_len_m1, int elem_bytes);
    // Expands the 16-byte-block bitmap format (2 byte mask + non-zero bytes) into dst.  @0x4361f0 (L2Load4)
    uint8_t* Decompress(uint8_t * dst, uint8_t * src, int dst_len);
    // Weight load: DDR -> GLB with unpacking/decompression.  @0x436340 (L2Load5)
    void LoadW();
};

