#pragma once
// AI2D image pre-processing engine of the K230 NPU.
//
// The engine is a singleton that is configured through the Extrw / Extraw / MmuConf instructions
// (which store into the register file at the start of the object) and executed by writing the
// "start" register (ai2d_proc).  It loads a rectangle of the source image from DDR or GLB, computes
// source coordinates with a 2x3 affine transform (M0..M5), optionally resamples (nearest/bilinear),
// converts YUV->RGB, pads and stores the result to DDR or GLB.  Every step also dumps check point
// files (CheckPoint streams kAi2d*).
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

struct AI2D {
    uint8_t *ddr_;                   // DDR base (copied from _G.DDR by ai2d_proc)
    uint8_t *glb_[16];               // GLB bank base pointers, selected by address bits [31:28]
    uint32_t src_ch_ptr_[4];         // source plane addresses (bits [31:28] = GLB bank when src_ind == 0)
    int32_t  src_x_;
    uint32_t src_y_;
    uint32_t dst_ch_ptr_[4];         // destination plane addresses
    int32_t  dst_x_;
    uint32_t dst_y_;
    uint32_t m_raw_[6];              // M0..M5 as written by Extrw/para_parser (IEEE-754 float bits)
    float    m_[6];                  // M0..M5 latched by ai_2d_para_update:
                                    //        src_x = M0*x + M1*y + M2 ; src_y = M3*x + M4*y + M5
    uint32_t interpolation_;         // 0 = nearest, else bilinear
    uint32_t cord_round_;            // rounding mode of the coordinate fixed-point conversion
    uint32_t channel_;               // number of planes processed by inter_calc (derived)
    uint32_t channel_cfg_;           // channel field of the register (inter_calc derives `channel` from it)
    uint32_t dst_channel_;           // number of destination planes written by slice_store
    uint32_t src_width_layout_[4];   // row pitch of each source plane
    uint32_t dst_width_layout_[4];   // row pitch of each destination plane
    uint32_t src_height_shape_;
    uint32_t src_width_shape_;
    uint32_t dst_height_shape_;
    uint32_t dst_width_shape_;
    uint32_t pad_height_;            // dst_height + pad_t + pad_b (computed by slice_store)
    uint32_t pad_width_;             // dst_width + pad_l + pad_r (computed by slice_store)
    uint32_t src_format_;            // 0 NV12, 1 NV21, 2 I420, 3 planar, 4 packed RGB888, 5 raw16
    uint32_t dst_format_;            // same encoding
    uint8_t  bound_smooth_;
    uint32_t csc_en_;                // 1 = run YUV444toRGB888 before storing
    uint32_t src_ind_;               // 1 = source is in DDR, 0 = in GLB
    uint32_t dst_ind_;               // 1 = destination is in DDR, 0 = in GLB
    int32_t  shift_;                 // raw16 shift (signed)
    uint32_t bound_ind_;             // 0 = constant border (bound_val), else replicate edge
    uint32_t bound_val_;
    uint32_t pad_l_;
    uint32_t pad_r_;
    uint32_t pad_t_;
    uint32_t pad_b_;
    uint32_t pad_mod_;               // 0 constant, 1 replicate, 2 mirror
    uint32_t const_pad_;             // only set by para_parser_back
    uint8_t  const_pad_ch_[4];       // constant padding value per plane
    uint8_t  is_signed_;             // "signed" register bit (data is int8/int16)
    uint8_t  cmd_id_;
    int32_t  yuv2rgb_coef_[12];      // 12-bit signed (sign-extended by ai_2d_para_update)
    uint32_t intr_mask_;
    uint32_t calc_enable_;
    uint32_t ddr_addr_offset_;       // added to DDR addresses in the check point dumps (field name is a guess; the layout is verified)
    uint32_t glb_addr_offset_;       // added to GLB addresses in the check point dumps (field name is a guess; the layout is verified)
    uint32_t glb_start_[16];         // MmuConf segment start
    uint32_t glb_depth_[16];         // MmuConf segment depth
    uint8_t *cur_glb_base_;          // GLB bank base of the last Print*Glb call
    uint8_t  plane_in_[4][0x10000];  // loaded source planes (src_height x src_width bytes each)
    uint8_t **interp_planes_;        // [4] -> 4 MiB planes: resampled image (dst_width x dst_height)
    uint8_t **padded_planes_;        // [4] -> 4 MiB planes: padded image (pad_width x pad_height)
    uint16_t **src_xi_;              // [dst_width][dst_height] integer source x of each output pixel
    uint16_t **src_yi_;              // [dst_width][dst_height] integer source y
    uint8_t **src_xf_;               // [dst_width][dst_height] 8-bit x fraction
    uint8_t **src_yf_;               // [dst_width][dst_height] 8-bit y fraction
    float    plane_scale_[4];        // bytes per sample of the destination planes (see slice_factor_cnt)
    uint8_t  plane_row_div_[4];      // vertical sub-sampling divisor of each plane (see slice_factor_cnt)

    // Singleton accessor: the object lives in the global register block AI2D_Ai2dInst.
    static AI2D *GetAI2D();

    ~AI2D();
    void ai_2d_para_update();
    void ai_2d_para_print();
    void PrintLoadDDRCheckPoint(uint32_t addr, uint32_t len);
    void PrintLoadGlbCheckPoint(uint32_t addr, uint32_t len);
    void PrintStoreDDRCheckPoint(uint32_t addr, uint32_t len);
    void PrintWriteGlbCheckPoint(uint32_t addr, uint32_t len);
    void slice_factor_cnt(int format);
    void print_rx_data(uint32_t addr, int pitch, float scale, uint8_t row_div);
    void print_tx_data(uint32_t addr, int pitch, float scale, uint8_t row_div, uint32_t row);
    void print_ddr_data();
    void slice_ld();
    void cord_calc();
    void inter_calc();
    void YUV444toRGB888();
    int64_t raw16_data_shift(int value);
    void slice_padding();
    void slice_store();
    void mem_init();
    void mem_delete();
    void ai2d_proc();
    void para_parser_back(const char *path);
    void para_parser(const char *path);
};


constexpr size_t kAi2dWords = (sizeof(AI2D) + 3) / 4;   // minimum size of _G.AI2D_Ai2dInst
