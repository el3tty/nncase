#pragma once
// AI2D image pre-processing engine of the K230 NPU (lifted from IDA/Hex-Rays output).
//
// The engine is a singleton that is configured through the Extrw / Extraw / MmuConf instructions
// (which store into the register file at the start of the object) and executed by writing the
// "start" register (ai2d_proc).  It loads a rectangle of the source image from DDR or GLB, computes
// source coordinates with a 2x3 affine transform (M0..M5), optionally resamples (nearest/bilinear),
// converts YUV->RGB, pads and stores the result to DDR or GLB.  Every step also dumps check point
// files (CheckPoint streams kAi2d*).
//
// Layout note: in the binary the object is the global `AI2D_Ai2dInst`.  Offset 0 holds the DDR base
// pointer (NOT a vptr: the destructor is not virtual), +8..+135 the 16 GLB bank base pointers, and the
// 32-bit configuration registers follow.  The object is about 256 KiB because it contains four 64 KiB
// source-plane buffers.
// TODO(layout): globals.h declares `uint32_t AI2D_Ai2dInst[0x4000]` (64 KiB) which is smaller than
// sizeof(AI2D); the owner of globals.{h,cpp} must enlarge it to at least kAi2dWords words (8-byte
// aligned), otherwise the plane buffers overrun neighbouring globals.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

struct AI2D {
    // ---- register file, as 32-bit word indices into the object (what Extrw/Extraw/MmuConf write) ----
    enum Reg : unsigned {
        kSrcCh0Ptr = 34, kSrcCh1Ptr = 35, kSrcCh2Ptr = 36, kSrcCh3Ptr = 37,   // +136..+148
        kSrcX = 38, kSrcY = 39,                                                // +152, +156
        kDstCh0Ptr = 40, kDstCh1Ptr = 41, kDstCh2Ptr = 42, kDstCh3Ptr = 43,   // +160..+172
        kDstX = 44, kDstY = 45,                                                // +176, +180
        kM0 = 46, kM1 = 47, kM2 = 48, kM3 = 49, kM4 = 50, kM5 = 51,            // +184..+204 (raw float bits)
        kInterpolation = 58, kCordRound = 59, kChannel = 60, kChannelCfg = 61, kDstChannel = 62,
        kSrcCh0WidthLayout = 63, kSrcCh1WidthLayout = 64, kSrcCh2WidthLayout = 65, kSrcCh3WidthLayout = 66,
        kDstCh0WidthLayout = 67, kDstCh1WidthLayout = 68, kDstCh2WidthLayout = 69, kDstCh3WidthLayout = 70,
        kSrcHeightShape = 71, kSrcWidthShape = 72, kDstHeightShape = 73, kDstWidthShape = 74,
        kSrcFormat = 77, kDstFormat = 78,                                      // +308, +312
        kBoundSmooth = 79,                                                     // +316 (byte)
        kCscEn = 80, kSrcInd = 81, kDstInd = 82, kShift = 83, kBoundInd = 84, kBoundVal = 85,
        kPadL = 86, kPadR = 87, kPadT = 88, kPadB = 89, kPadMod = 90, kConstPad = 91,
        kConstPadCh = 92,                                                      // +368..+371 (4 bytes)
        kSignedCmdId = 93,                                                     // +372 signed, +373 cmd_id (bytes)
        kYuv2RgbCoef0 = 94,                                                    // +376 .. kYuv2RgbCoef0+11
        kIntrMask = 106, kCalcEnable = 107,                                    // +424, +428
        kGlbStart = 110, kGlbDepth = 126                                       // +440, +504 (16 words each, MmuConf)
    };

    uint8_t *ddr_;                   // +0     DDR base (copied from g_DDR by ai2d_proc)
    uint8_t *glb_[16];               // +8     GLB bank base pointers, selected by address bits [31:28]
    uint32_t src_ch_ptr_[4];         // +136   source plane addresses (bits [31:28] = GLB bank when src_ind == 0)
    int32_t  src_x_;                 // +152
    uint32_t src_y_;                 // +156
    uint32_t dst_ch_ptr_[4];         // +160   destination plane addresses
    int32_t  dst_x_;                 // +176
    uint32_t dst_y_;                 // +180
    uint32_t m_raw_[6];              // +184   M0..M5 as written by Extrw/para_parser (IEEE-754 float bits)
    float    m_[6];                  // +208   M0..M5 latched by ai_2d_para_update:
                                    //        src_x = M0*x + M1*y + M2 ; src_y = M3*x + M4*y + M5
    uint32_t interpolation_;         // +232   0 = nearest, else bilinear
    uint32_t cord_round_;            // +236   rounding mode of the coordinate fixed-point conversion
    uint32_t channel_;               // +240   number of planes processed by inter_calc (derived)
    uint32_t channel_cfg_;           // +244   channel field of the register (inter_calc derives `channel` from it)
    uint32_t dst_channel_;           // +248   number of destination planes written by slice_store
    uint32_t src_width_layout_[4];   // +252   row pitch of each source plane
    uint32_t dst_width_layout_[4];   // +268   row pitch of each destination plane
    uint32_t src_height_shape_;      // +284
    uint32_t src_width_shape_;       // +288
    uint32_t dst_height_shape_;      // +292
    uint32_t dst_width_shape_;       // +296
    uint32_t pad_height_;            // +300   dst_height + pad_t + pad_b (computed by slice_store)
    uint32_t pad_width_;             // +304   dst_width + pad_l + pad_r (computed by slice_store)
    uint32_t src_format_;            // +308   0 NV12, 1 NV21, 2 I420, 3 planar, 4 packed RGB888, 5 raw16
    uint32_t dst_format_;            // +312   same encoding
    uint8_t  bound_smooth_;          // +316
    uint8_t  pad316_[3];            // +317
    uint32_t csc_en_;                // +320   1 = run YUV444toRGB888 before storing
    uint32_t src_ind_;               // +324   1 = source is in DDR, 0 = in GLB
    uint32_t dst_ind_;               // +328   1 = destination is in DDR, 0 = in GLB
    int32_t  shift_;                 // +332   raw16 shift (signed)
    uint32_t bound_ind_;             // +336   0 = constant border (bound_val), else replicate edge
    uint32_t bound_val_;             // +340
    uint32_t pad_l_;                 // +344
    uint32_t pad_r_;                 // +348
    uint32_t pad_t_;                 // +352
    uint32_t pad_b_;                 // +356
    uint32_t pad_mod_;               // +360   0 constant, 1 replicate, 2 mirror
    uint32_t const_pad_;             // +364   only set by para_parser_back
    uint8_t  const_pad_ch_[4];       // +368   constant padding value per plane
    uint8_t  is_signed_;             // +372   "signed" register bit (data is int8/int16)
    uint8_t  cmd_id_;                // +373
    uint8_t  pad374_[2];            // +374
    int32_t  yuv2rgb_coef_[12];      // +376   12-bit signed (sign-extended by ai_2d_para_update)
    uint32_t intr_mask_;             // +424
    uint32_t calc_enable_;           // +428
    uint32_t ddr_addr_offset_;       // +432   added to DDR addresses in the check point dumps (field name is a guess; the layout is verified)
    uint32_t glb_addr_offset_;       // +436   added to GLB addresses in the check point dumps (field name is a guess; the layout is verified)
    uint32_t glb_start_[16];         // +440   MmuConf segment start (written as AI2D_Ai2dInst[seg + 110])
    uint32_t glb_depth_[16];         // +504   MmuConf segment depth (written as AI2D_Ai2dInst[seg + 126])
    uint8_t  pad568_[8];            // +568   padding, never accessed in the asm
    uint8_t *cur_glb_base_;          // +576   GLB bank base of the last Print*Glb call
    uint8_t  plane_in_[4][0x10000];  // +584   loaded source planes (src_height x src_width bytes each)
    uint8_t **interp_planes_;        // +262728 [4] -> 4 MiB planes: resampled image (dst_width x dst_height)
    uint8_t **padded_planes_;        // +262736 [4] -> 4 MiB planes: padded image (pad_width x pad_height)
    uint16_t **src_xi_;              // +262744 [dst_width][dst_height] integer source x of each output pixel
    uint16_t **src_yi_;              // +262752 [dst_width][dst_height] integer source y
    uint8_t **src_xf_;               // +262760 [dst_width][dst_height] 8-bit x fraction
    uint8_t **src_yf_;               // +262768 [dst_width][dst_height] 8-bit y fraction
    float    plane_scale_[4];        // +262776 bytes per sample of the destination planes (see slice_factor_cnt)
    uint8_t  plane_row_div_[4];      // +262792 vertical sub-sampling divisor of each plane (see slice_factor_cnt)
    uint8_t  tail_pad_[4];          // +262796

    // Singleton accessor: the object lives in the global register block AI2D_Ai2dInst.
    static AI2D *GetAI2D();
    // 32-bit word view of the register file (word n == byte offset 4*n), for Extrw/Extraw/MmuConf.
    uint32_t *regs() { return reinterpret_cast<uint32_t *>(this); }

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

static_assert(offsetof(AI2D, glb_) == 8, "AI2D layout");
static_assert(offsetof(AI2D, src_ch_ptr_) == 136, "AI2D layout");
static_assert(offsetof(AI2D, m_raw_) == 184, "AI2D layout");
static_assert(offsetof(AI2D, m_) == 208, "AI2D layout");
static_assert(offsetof(AI2D, interpolation_) == 232, "AI2D layout");
static_assert(offsetof(AI2D, src_height_shape_) == 284, "AI2D layout");
static_assert(offsetof(AI2D, bound_smooth_) == 316, "AI2D layout");
static_assert(offsetof(AI2D, const_pad_ch_) == 368, "AI2D layout");
static_assert(offsetof(AI2D, yuv2rgb_coef_) == 376, "AI2D layout");
static_assert(offsetof(AI2D, intr_mask_) == 424, "AI2D layout");
static_assert(offsetof(AI2D, glb_start_) == 440, "AI2D layout");
static_assert(offsetof(AI2D, glb_depth_) == 504, "AI2D layout");
static_assert(offsetof(AI2D, cur_glb_base_) == 576, "AI2D layout");
static_assert(offsetof(AI2D, plane_in_) == 584, "AI2D layout");
static_assert(offsetof(AI2D, interp_planes_) == 262728, "AI2D layout");
static_assert(offsetof(AI2D, plane_scale_) == 262776, "AI2D layout");
static_assert(offsetof(AI2D, plane_row_div_) == 262792, "AI2D layout");

constexpr size_t kAi2dWords = (sizeof(AI2D) + 3) / 4;   // minimum size of AI2D_Ai2dInst
