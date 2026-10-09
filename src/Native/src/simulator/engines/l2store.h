#pragma once
// L2 store unit (GLB -> DDR) of the K230 NPU C-model.
// Lifted from IDA/Hex-Rays output (L2Store1..L2Store3 in sources/).
//
// The instruction classes (l2storeconfinstruction.cpp / l2storeinstruction.cpp) use the singleton
// `L2Store_L2StoreInst` (0x53A6C0 in the original binary), cast to L2Store*.  The object layout must
// match the original byte offsets.  In the original binary the
// instruction classes wrote the fields through loose globals; this is the address map:
//   L2Store_L2StoreInst[0]  0x53A6C0  +0   ddr_dim0    (L2StoreConf, shape0 >> 32)
//   0x53A6C4  +4   ddr_dim1    (L2StoreConf, shape0 >> 16)
//   0x53A6C8  +8   ddr_pitch   (L2StoreConf, shape0 & 0xFFFF)
//   0x53A6CC  +12  ddr_dim3    (L2StoreConf, cleared)
//   0x53A6D0  +16  glb_dim0    (L2StoreConf, shape1 >> 32)
//   0x53A6D4  +20  glb_dim1    (L2StoreConf)
//   0x53A6D8  +24  glb_pitch   (L2StoreConf)
//   0x53A6DC  +28  glb_dim3    (L2StoreConf, cleared)
//   0x53A6E0  +32  mode0/mode1 (L2StoreConf, 64-bit store: bytes +34..+39 are zero)
//   0x53A6E8  +40  ddr_ptr     (L2StoreInstruction, g_DDR + ddr_offset)
//   0x53A6F0  +48  glb_ptr     (L2StoreInstruction, g_GLB[bank] + (glb_addr & 0xFFFFFFF))
//   0x53A6F8  +56  batch_count (L2StoreInstruction, shape >> 48)
//   0x53A6FC  +60  plane_count (shape >> 32)
//   0x53A700  +64  row_count   (shape >> 16)
//   0x53A704  +68  row_len     (shape & 0xFFFF)
//   0x53A708  +72  ddr_offset  (L2StoreInstruction)
//   0x53A70C  +76  glb_mmu_addr (GLB offset + 32 * MMU base of the bank)
//   0x53A710  +80  glb_bank
// These globals are no longer declared; the instruction classes write the members directly.
#include <cstddef>
#include <cstdint>

struct L2Store {
    // Destination (DDR) tensor layout, in elements: planes per tensor, rows per plane, row pitch.
    uint32_t ddr_dim0_;        // +0
    uint32_t ddr_dim1_;        // +4
    uint32_t ddr_pitch_;       // +8
    uint32_t ddr_dim3_;        // +12  always 0
    // Source (GLB) tensor layout.
    uint32_t glb_dim0_;        // +16
    uint32_t glb_dim1_;        // +20
    uint32_t glb_pitch_;       // +24
    uint32_t glb_dim3_;        // +28  always 0
    // Data format (mode == 0x201: float16 in GLB -> float32 in DDR; mode0 != 0: 16-bit elements;
    // otherwise 8-bit elements).
    uint8_t mode0_;            // +32
    uint8_t mode1_;            // +33
    uint8_t pad34_[6];         // +34  zero (the conf instruction stores the mode as a 64-bit word)
    uint8_t * ddr_ptr_;        // +40  destination pointer (g_DDR + ddr_offset)
    uint8_t * glb_ptr_;        // +48  source pointer inside the GLB bank
    // Extent of the transfer.
    uint32_t batch_count_;     // +56  number of whole tensors
    uint32_t plane_count_;     // +60  planes per tensor
    uint32_t row_count_;       // +64  rows per plane
    uint32_t row_len_;         // +68  elements per row
    uint32_t ddr_offset_;      // +72
    uint32_t glb_mmu_addr_;    // +76  GLB offset + 32 * MMU base of the bank
    uint32_t glb_bank_;        // +80

    uint16_t Mode() const { return (uint16_t)(mode0_ | (mode1_ << 8)); }

    // Copies a 4-D block GLB -> DDR with optional float16 -> float32 conversion.
    // Returns batch_count (the original returned the leftover register value).  @0x436b30 (L2Store1)
    int64_t Store();
    // Dumps the DDR write data / address to store_ddr_wdata / store_ddr_waddr.  @0x436e00 (L2Store2)
    int64_t PrintStoreDDRCheckPoint(uint32_t rel_offset, uint32_t num_bytes);
    // Dumps the GLB read address / data to store_glb_raddr / store_glb_rdata.  @0x437560 (L2Store3)
    int64_t PrintLoadGlbCheckPoint(uint32_t rel_offset, uint32_t num_bytes);
};

