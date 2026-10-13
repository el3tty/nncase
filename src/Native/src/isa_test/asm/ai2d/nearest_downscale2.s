# AI2D nearest 2x downscale 8x8 -> 4x4: M0 = M4 = 2048.0 (2.0), cord_round 2: source x = 2 * x_dst
# IN @0x4000: u8 [229, 73, 221, 209, 127, 159, 130, 83, 178, 146, 21, 119, 67, 203, 186, 47, 16, 12, 93, 107, 233, 137, 76, 67, 110, 142, 124, 20, 53, 252, 143, 59, 35, 48, 4, 43, 200, 218, 250, 54, 43, 239, 215, 205, 178, 39, 131, 157, 140, 204, 79, 29, 136, 54, 166, 117, 198, 149, 44, 219, 135, 209, 234, 239]
# OUT @0x5000: u8 [229, 221, 127, 130, 16, 93, 233, 76, 35, 4, 200, 250, 140, 79, 136, 166]
    li x1, 0x4000
    extraw 0, x1      # src_ch0_ptr
    extraw 4, x0      # (unused)
    extraw 8, x0      # (unused)
    extraw 12, x0      # (unused)
    li x1, 0x5000
    extraw 16, x1      # dst_ch0_ptr
    extraw 20, x0      # (unused)
    extraw 24, x0      # (unused)
    extraw 28, x0      # (unused)
    li x1, 0x8
    extrw 32, x1      # src width/layout 0,1
    extrw 36, x0      # src width/layout 2,3
    li x1, 0x4
    extrw 40, x1      # dst width/layout 0,1
    extrw 44, x0      # dst width/layout 2,3
    li x1, 0x45000000
    extrw 48, x1      # M0
    extrw 52, x0      # M1
    extrw 56, x0      # M3
    li x1, 0x45000000
    extrw 60, x1      # M4
    extrw 64, x0      # (unused)
    extrw 68, x0      # (unused)
    li x1, 0x33000089
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
    li x1, 0x80008
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
