# AI2D copies a 4x5 window starting at (src_x 2, src_y 1) of a 6x8 image to (dst_x 3, dst_y 2) of a 7x9 destination
# IN @0x4000: u8[6,8] [
#   [19, 188, 44, 191, 69, 56, 152, 183]
#   [181, 112, 251, 189, 192, 34, 56, 201]
#   [206, 38, 4, 238, 137, 87, 227, 195]
#   [72, 83, 176, 1, 192, 144, 118, 159]
#   [93, 172, 92, 252, 71, 23, 10, 233]
#   [35, 192, 61, 135, 151, 92, 247, 224]
# ]
# OUT @0x5000: u8[7,9] [
#   [0xa5*9]*2
#   [0xa5, 0xa5, 0xa5, 251, 189, 192, 34, 56, 165]
#   [0xa5, 0xa5, 0xa5, 4, 238, 137, 87, 227, 165]
#   [0xa5, 0xa5, 0xa5, 176, 1, 192, 144, 118, 165]
#   [0xa5, 0xa5, 0xa5, 92, 252, 71, 23, 10, 165]
#   [0xa5*9]
# ]
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
    li x1, 0x8
    extrw 32, x1      # src width/layout 0,1
    extrw 36, x0      # src width/layout 2,3
    li x1, 0x9
    extrw 40, x1      # dst width/layout 0,1
    extrw 44, x0      # dst width/layout 2,3
    li x1, 0x44800000
    extrw 48, x1      # M0
    extrw 52, x0      # M1
    extrw 56, x0      # M3
    li x1, 0x44800000
    extrw 60, x1      # M4
    extrw 64, x0      # (unused, must be 0)
    extrw 68, x0      # (unused, must be 0)
    li x1, 0x33000009
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
    li x1, 0x40005
    extrw 120, x1      # source height/width
    li x1, 0x40005
    extrw 124, x1      # destination height/width
    extrw 128, x0      # M2
    extrw 132, x0      # M5
    li x1, 0x10002
    extrw 136, x1      # src_y/src_x
    li x1, 0x80020003
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    ai2d_compute
    fence
    end
