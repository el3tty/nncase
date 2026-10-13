# AI2D nearest 2x downscale 8x8 -> 4x4: M0 = M4 = 2048.0 (2.0), cord_round 2: source x = 2 * x_dst
# IN @0x4000: u8 [229, 73, 221, 209, 127, 159, 130, 83, 178, 146, 21, 119, 67, 203, 186, 47, 16, 12, 93, 107, 233, 137, 76, 67, 110, 142, 124, 20, 53, 252, 143, 59, 35, 48, 4, 43, 200, 218, 250, 54, 43, 239, 215, 205, 178, 39, 131, 157, 140, 204, 79, 29, 136, 54, 166, 117, 198, 149, 44, 219, 135, 209, 234, 239]
# OUT @0x5000: u8 [229, 221, 127, 130, 16, 93, 233, 76, 35, 4, 200, 250, 140, 79, 136, 166]
    li x1, 0x4000
    extrw 0, x1      # src_ch0_ptr
    li x1, 0x5000
    extrw 16, x1      # dst_ch0_ptr
    li x1, 0x8
    extrw 32, x1      # src width/layout 0,1
    li x1, 0x0
    extrw 36, x1      # src width/layout 2,3
    li x1, 0x4
    extrw 40, x1      # dst width/layout 0,1
    li x1, 0x0
    extrw 44, x1      # dst width/layout 2,3
    li x1, 0x45000000
    extrw 48, x1      # M0
    li x1, 0x0
    extrw 52, x1      # M1
    li x1, 0x0
    extrw 128, x1      # M2
    li x1, 0x0
    extrw 56, x1      # M3
    li x1, 0x45000000
    extrw 60, x1      # M4
    li x1, 0x0
    extrw 132, x1      # M5
    li x1, 0x33000089
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
    li x1, 0x80008
    extrw 120, x1      # source height/width
    li x1, 0x40004
    extrw 124, x1      # destination height/width
    li x1, 0x0
    extrw 136, x1      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    end
