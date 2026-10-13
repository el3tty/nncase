# AI2D format conversion packed RGB888 (src_format 4, pitch in bytes) -> planar (dst_format 3, three planes)
# IN @0x4000: u8 [164, 199, 85, 186, 41, 230, 143, 43, 59, 82, 63, 156, 144, 133, 35, 77, 32, 117, 215, 223, 161, 45, 168, 116, 219, 142, 45, 105, 42, 96, 34, 95, 112, 183, 230, 146, 9, 201, 13, 85, 171, 1, 31, 142, 208]
# OUT @0x5000: u8 [164, 186, 143, 82, 144, 77, 215, 45, 219, 105, 34, 183, 9, 85, 31]
# OUT @0x5100: u8 [199, 41, 43, 63, 133, 32, 223, 168, 142, 42, 95, 230, 201, 171, 142]
# OUT @0x5200: u8 [85, 230, 59, 156, 35, 117, 161, 116, 45, 96, 112, 146, 13, 1, 208]
    li x1, 0x4000
    extraw 0, x1      # src_ch0_ptr
    extraw 4, x0      # (unused)
    extraw 8, x0      # (unused)
    extraw 12, x0      # (unused)
    li x1, 0x5000
    extraw 16, x1      # dst_ch0_ptr
    li x1, 0x5100
    extraw 20, x1      # dst_ch1_ptr
    li x1, 0x5200
    extraw 24, x1      # dst_ch2_ptr
    extraw 28, x0      # (unused)
    li x1, 0xf
    extrw 32, x1      # src width/layout 0,1
    extrw 36, x0      # src width/layout 2,3
    li x1, 0x50005
    extrw 40, x1      # dst width/layout 0,1
    li x1, 0x5
    extrw 44, x1      # dst width/layout 2,3
    li x1, 0x44800000
    extrw 48, x1      # M0
    extrw 52, x0      # M1
    extrw 56, x0      # M3
    li x1, 0x44800000
    extrw 60, x1      # M4
    extrw 64, x0      # (unused)
    extrw 68, x0      # (unused)
    li x1, 0x3400001b
    extrw 72, x1      # formats and mode
    extrw 76, x0      # bound_val
    extrw 80, x0      # (unused)
    extrw 84, x0      # (unused)
    extrw 88, x0      # (unused)
    extrw 92, x0      # (unused)
    extrw 96, x0      # const_pad ch0
    li x1, 0x4000000
    extrw 100, x1      # cmd_id, dst_ind/src_ind, const_pad ch1..3
    extrw 104, x0      # (unused)
    extrw 108, x0      # (unused)
    extrw 112, x0      # pad bottom/top
    extrw 116, x0      # pad right/left
    li x1, 0x30005
    extrw 120, x1      # source height/width
    li x1, 0x30005
    extrw 124, x1      # destination height/width
    extrw 128, x0      # M2
    extrw 132, x0      # M5
    extrw 136, x0      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    ai2d_compute
    fence
    end
