# AI2D bilinear 2x upscale 4x4 -> 8x8 (interpolation 1): the fraction is (x_dst * 512 >> 2) & 255 = 0 or 128, so the odd
# output pixels are the rounded mean ((a + b + 1) >> 1) of their neighbours; bound_ind 1 replicates the edge for the last one
# IN @0x4000: u8 [149, 100, 163, 225, 67, 123, 219, 225, 19, 155, 4, 19, 33, 123, 30, 185]
# OUT @0x5000: u8 [149, 125, 100, 132, 163, 194, 225, 225, 108, 110, 112, 151, 191, 208, 225, 225, 67, 95, 123, 171, 219, 222, 225, 225, 43, 91, 139, 125, 112, 117, 122, 122, 19, 87, 155, 80, 4, 12, 19, 19, 26, 83, 139, 78, 17, 60, 102, 102, 33, 78, 123, 77, 30, 108, 185, 185, 33, 78, 123, 77, 30, 108, 185, 185]
    li x1, 0x4000
    extraw 0, x1      # src_ch0_ptr
    extraw 4, x0      # (unused, must be 0)
    extraw 8, x0      # (unused, must be 0)
    extraw 12, x0      # (unused, must be 0)
    li x1, 0x5000
    extraw 16, x1      # dst_ch0_ptr
    extraw 20, x0      # (unused, must be 0)
    extraw 24, x0      # (unused, must be 0)
    extraw 28, x0      # (unused, must be 0)
    li x1, 0x4
    extrw 32, x1      # src width/layout 0,1
    extrw 36, x0      # src width/layout 2,3
    li x1, 0x8
    extrw 40, x1      # dst width/layout 0,1
    extrw 44, x0      # dst width/layout 2,3
    li x1, 0x44000000
    extrw 48, x1      # M0
    extrw 52, x0      # M1
    extrw 56, x0      # M3
    li x1, 0x44000000
    extrw 60, x1      # M4
    extrw 64, x0      # (unused, must be 0)
    extrw 68, x0      # (unused, must be 0)
    li x1, 0x33100189
    extrw 72, x1      # formats and mode
    extrw 76, x0      # bound_val
    extrw 80, x0      # (unused, must be 0)
    extrw 84, x0      # (unused, must be 0)
    extrw 88, x0      # (unused, must be 0)
    extrw 92, x0      # (unused, must be 0)
    extrw 96, x0      # const_pad ch0
    li x1, 0x4000000
    extrw 100, x1      # cmd_id, dst_ind/src_ind, const_pad ch1..3
    extrw 104, x0      # (unused, must be 0)
    extrw 108, x0      # (unused, must be 0)
    extrw 112, x0      # pad bottom/top
    extrw 116, x0      # pad right/left
    li x1, 0x40004
    extrw 120, x1      # source height/width
    li x1, 0x80008
    extrw 124, x1      # destination height/width
    extrw 128, x0      # M2
    extrw 132, x0      # M5
    extrw 136, x0      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    ai2d_compute
    fence
    end
