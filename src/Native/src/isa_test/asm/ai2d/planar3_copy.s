# AI2D planar copy of three channels (channel_cfg 3, dst_channel 3), one pointer/pitch per plane
# IN @0x4000: u8 [12, 134, 232, 162, 187, 106, 110, 116, 241, 93, 230, 234, 17, 5, 79, 49]
# IN @0x4100: u8 [182, 76, 105, 232, 220, 182, 175, 165, 201, 87, 149, 4, 219, 76, 233, 75]
# IN @0x4200: u8 [6, 120, 225, 213, 85, 158, 107, 220, 56, 51, 247, 183, 121, 140, 76, 109]
# OUT @0x5000: u8 [12, 134, 232, 162, 187, 106, 110, 116, 241, 93, 230, 234, 17, 5, 79, 49]
# OUT @0x5100: u8 [182, 76, 105, 232, 220, 182, 175, 165, 201, 87, 149, 4, 219, 76, 233, 75]
# OUT @0x5200: u8 [6, 120, 225, 213, 85, 158, 107, 220, 56, 51, 247, 183, 121, 140, 76, 109]
    li x1, 0x4000
    extraw 0, x1      # src_ch0_ptr
    li x1, 0x4100
    extraw 4, x1      # src_ch1_ptr
    li x1, 0x4200
    extraw 8, x1      # src_ch2_ptr
    extraw 12, x0      # (unused)
    li x1, 0x5000
    extraw 16, x1      # dst_ch0_ptr
    li x1, 0x5100
    extraw 20, x1      # dst_ch1_ptr
    li x1, 0x5200
    extraw 24, x1      # dst_ch2_ptr
    extraw 28, x0      # (unused)
    li x1, 0x40004
    extrw 32, x1      # src width/layout 0,1
    li x1, 0x4
    extrw 36, x1      # src width/layout 2,3
    li x1, 0x40004
    extrw 40, x1      # dst width/layout 0,1
    li x1, 0x4
    extrw 44, x1      # dst width/layout 2,3
    li x1, 0x44800000
    extrw 48, x1      # M0
    extrw 52, x0      # M1
    extrw 56, x0      # M3
    li x1, 0x44800000
    extrw 60, x1      # M4
    extrw 64, x0      # (unused)
    extrw 68, x0      # (unused)
    li x1, 0x3300001b
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
    li x1, 0x40004
    extrw 120, x1      # source height/width
    li x1, 0x40004
    extrw 124, x1      # destination height/width
    extrw 128, x0      # M2
    extrw 132, x0      # M5
    extrw 136, x0      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    ai2d_compute
    fence
    end
