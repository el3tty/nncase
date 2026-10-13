# AI2D planar copy of three channels (channel_cfg 3, dst_channel 3), one pointer/pitch per plane
# IN @0x4000: u8 [12, 134, 232, 162, 187, 106, 110, 116, 241, 93, 230, 234, 17, 5, 79, 49]
# IN @0x4100: u8 [182, 76, 105, 232, 220, 182, 175, 165, 201, 87, 149, 4, 219, 76, 233, 75]
# IN @0x4200: u8 [6, 120, 225, 213, 85, 158, 107, 220, 56, 51, 247, 183, 121, 140, 76, 109]
# OUT @0x5000: u8 [12, 134, 232, 162, 187, 106, 110, 116, 241, 93, 230, 234, 17, 5, 79, 49]
# OUT @0x5100: u8 [182, 76, 105, 232, 220, 182, 175, 165, 201, 87, 149, 4, 219, 76, 233, 75]
# OUT @0x5200: u8 [6, 120, 225, 213, 85, 158, 107, 220, 56, 51, 247, 183, 121, 140, 76, 109]
    li x1, 0x4000
    extrw 0, x1      # src_ch0_ptr
    li x1, 0x4100
    extrw 4, x1      # src_ch1_ptr
    li x1, 0x4200
    extrw 8, x1      # src_ch2_ptr
    li x1, 0x5000
    extrw 16, x1      # dst_ch0_ptr
    li x1, 0x5100
    extrw 20, x1      # dst_ch1_ptr
    li x1, 0x5200
    extrw 24, x1      # dst_ch2_ptr
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
    li x1, 0x3300001b
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
    li x1, 0x40004
    extrw 120, x1      # source height/width
    li x1, 0x40004
    extrw 124, x1      # destination height/width
    li x1, 0x0
    extrw 136, x1      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    end
