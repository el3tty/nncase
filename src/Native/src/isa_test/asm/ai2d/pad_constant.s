# AI2D constant padding (pad_mod 0): top 1, bottom 2, left 2, right 1 with const_pad channel 0 = 77; 3x4 -> 6x7
# IN @0x4000: u8 [198, 101, 181, 234, 170, 30, 27, 65, 253, 191, 215, 227]
# OUT @0x5000: u8 [77, 77, 77, 77, 77, 77, 77, 77, 77, 198, 101, 181, 234, 77, 77, 77, 170, 30, 27, 65, 77, 77, 77, 253, 191, 215, 227, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77]
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
    li x1, 0x4
    extrw 32, x1      # src width/layout 0,1
    extrw 36, x0      # src width/layout 2,3
    li x1, 0x7
    extrw 40, x1      # dst width/layout 0,1
    extrw 44, x0      # dst width/layout 2,3
    li x1, 0x44800000
    extrw 48, x1      # M0
    extrw 52, x0      # M1
    extrw 56, x0      # M3
    li x1, 0x44800000
    extrw 60, x1      # M4
    extrw 64, x0      # (unused)
    extrw 68, x0      # (unused)
    li x1, 0x33000009
    extrw 72, x1      # formats and mode
    extrw 76, x0      # bound_val
    extrw 80, x0      # (unused)
    extrw 84, x0      # (unused)
    extrw 88, x0      # (unused)
    extrw 92, x0      # (unused)
    li x1, 0x4d000000
    extrw 96, x1      # const_pad ch0
    li x1, 0x4000000
    extrw 100, x1      # cmd_id, dst_ind/src_ind, const_pad ch1..3
    extrw 104, x0      # (unused)
    extrw 108, x0      # (unused)
    li x1, 0x20001
    extrw 112, x1      # pad bottom/top
    li x1, 0x10002
    extrw 116, x1      # pad right/left
    li x1, 0x30004
    extrw 120, x1      # source height/width
    li x1, 0x30004
    extrw 124, x1      # destination height/width
    extrw 128, x0      # M2
    extrw 132, x0      # M5
    extrw 136, x0      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    ai2d_compute
    fence
    end
