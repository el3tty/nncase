# AI2D format conversion planar (three planes) -> packed RGB888 (dst_format 4, pitch in bytes)
# IN @0x4000: u8 [164, 186, 143, 82, 144, 77, 215, 45, 219, 105, 34, 183, 9, 85, 31]
# IN @0x4100: u8 [199, 41, 43, 63, 133, 32, 223, 168, 142, 42, 95, 230, 201, 171, 142]
# IN @0x4200: u8 [85, 230, 59, 156, 35, 117, 161, 116, 45, 96, 112, 146, 13, 1, 208]
# OUT @0x5000: u8 [164, 199, 85, 186, 41, 230, 143, 43, 59, 82, 63, 156, 144, 133, 35, 77, 32, 117, 215, 223, 161, 45, 168, 116, 219, 142, 45, 105, 42, 96, 34, 95, 112, 183, 230, 146, 9, 201, 13, 85, 171, 1, 31, 142, 208]
    li x1, 0x4000
    extrw 0, x1      # src_ch0_ptr
    li x1, 0x4100
    extrw 4, x1      # src_ch1_ptr
    li x1, 0x4200
    extrw 8, x1      # src_ch2_ptr
    li x1, 0x5000
    extrw 16, x1      # dst_ch0_ptr
    li x1, 0x50005
    extrw 32, x1      # src width/layout 0,1
    li x1, 0x5
    extrw 36, x1      # src width/layout 2,3
    li x1, 0xf
    extrw 40, x1      # dst width/layout 0,1
    li x1, 0x0
    extrw 44, x1      # dst width/layout 2,3
    li x1, 0x44800000
    extrw 48, x1      # M0
    li x1, 0x0
    extrw 52, x1      # M1
    li x1, 0x0
    extrw 128, x1      # M2
    li x1, 0x0
    extrw 56, x1      # M3
    li x1, 0x44800000
    extrw 60, x1      # M4
    li x1, 0x0
    extrw 132, x1      # M5
    li x1, 0x4300001b
    extrw 72, x1      # formats and mode
    li x1, 0x0
    extrw 76, x1      # bound_val
    li x1, 0x0
    extrw 96, x1      # const_pad ch0
    li x1, 0x0
    extrw 100, x1      # dst_ind/src_ind, const_pad ch1..3
    li x1, 0x0
    extrw 112, x1      # pad bottom/top
    li x1, 0x0
    extrw 116, x1      # pad right/left
    li x1, 0x30005
    extrw 120, x1      # source height/width
    li x1, 0x30005
    extrw 124, x1      # destination height/width
    li x1, 0x0
    extrw 136, x1      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    end
