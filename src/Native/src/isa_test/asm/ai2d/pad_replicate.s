# AI2D edge replication padding (pad_mod 1): top 1, bottom 2, left 2, right 1
# IN @0x4000: u8 [198, 101, 181, 234, 170, 30, 27, 65, 253, 191, 215, 227]
# OUT @0x5000: u8 [198, 198, 198, 101, 181, 234, 234, 198, 198, 198, 101, 181, 234, 234, 170, 170, 170, 30, 27, 65, 65, 253, 253, 253, 191, 215, 227, 227, 253, 253, 253, 191, 215, 227, 227, 253, 253, 253, 191, 215, 227, 227]
    li x1, 0x4000
    extrw 0, x1      # src_ch0_ptr
    li x1, 0x5000
    extrw 16, x1      # dst_ch0_ptr
    li x1, 0x4
    extrw 32, x1      # src width/layout 0,1
    li x1, 0x0
    extrw 36, x1      # src width/layout 2,3
    li x1, 0x7
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
    li x1, 0x33000409
    extrw 72, x1      # formats and mode
    li x1, 0x0
    extrw 76, x1      # bound_val
    li x1, 0x0
    extrw 96, x1      # const_pad ch0
    li x1, 0x0
    extrw 100, x1      # dst_ind/src_ind, const_pad ch1..3
    li x1, 0x20001
    extrw 112, x1      # pad bottom/top
    li x1, 0x10002
    extrw 116, x1      # pad right/left
    li x1, 0x30004
    extrw 120, x1      # source height/width
    li x1, 0x30004
    extrw 124, x1      # destination height/width
    li x1, 0x0
    extrw 136, x1      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    end
