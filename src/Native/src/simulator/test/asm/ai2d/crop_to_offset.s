# AI2D copies a 4x5 window starting at (src_x 2, src_y 1) of a 6x8 image to (dst_x 3, dst_y 2) of a 7x9 destination
# DDR_IN: u8 [19, 188, 44, 191, 69, 56, 152, 183, 181, 112, 251, 189, 192, 34, 56, 201, 206, 38, 4, 238, 137, 87, 227, 195, 72, 83, 176, 1, 192, 144, 118, 159, 93, 172, 92, 252, 71, 23, 10, 233, 35, 192, 61, 135, 151, 92, 247, 224]
# OUT @0x4000: u8 [165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 251, 189, 192, 34, 56, 165, 165, 165, 165, 4, 238, 137, 87, 227, 165, 165, 165, 165, 176, 1, 192, 144, 118, 165, 165, 165, 165, 92, 252, 71, 23, 10, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165]
    li x1, 0x100000
    extrw 0, x1      # src_ch0_ptr
    li x1, 0x4000
    extrw 16, x1      # dst_ch0_ptr
    li x1, 0x8
    extrw 32, x1      # src width/layout 0,1
    li x1, 0x0
    extrw 36, x1      # src width/layout 2,3
    li x1, 0x9
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
    li x1, 0x33000009
    extrw 72, x1      # formats and mode
    li x1, 0x0
    extrw 76, x1      # bound_val
    li x1, 0x0
    extrw 96, x1      # const_pad ch0
    li x1, 0x1000000
    extrw 100, x1      # dst_ind/src_ind, const_pad ch1..3
    li x1, 0x0
    extrw 112, x1      # pad bottom/top
    li x1, 0x0
    extrw 116, x1      # pad right/left
    li x1, 0x40005
    extrw 120, x1      # source height/width
    li x1, 0x40005
    extrw 124, x1      # destination height/width
    li x1, 0x10002
    extrw 136, x1      # src_y/src_x
    li x1, 0x80020003
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    end
