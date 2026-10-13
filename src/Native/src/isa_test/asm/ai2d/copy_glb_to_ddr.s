# AI2D identity copy from GLB (bank 0 offset 0x4000) to DDR (dst_ind=1); pitch 6 -> 7
# IN @0x4000: u8 [175, 196, 25, 246, 67, 211, 151, 103, 92, 185, 142, 23, 72, 89, 110, 42, 218, 136, 167, 230, 68, 176, 127, 135, 172, 0, 75, 55, 250, 6]
# DDR_OUT: u8 [175, 196, 25, 246, 67, 211, 165, 151, 103, 92, 185, 142, 23, 165, 72, 89, 110, 42, 218, 136, 165, 167, 230, 68, 176, 127, 135, 165, 172, 0, 75, 55, 250, 6, 165]
    li x1, 0x4000
    extrw 0, x1      # src_ch0_ptr
    li x1, DDR_OUT
    extrw 16, x1      # dst_ch0_ptr
    li x1, 0x6
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
    li x1, 0x33000009
    extrw 72, x1      # formats and mode
    li x1, 0x0
    extrw 76, x1      # bound_val
    li x1, 0x0
    extrw 96, x1      # const_pad ch0
    li x1, 0x2000000
    extrw 100, x1      # dst_ind/src_ind, const_pad ch1..3
    li x1, 0x0
    extrw 112, x1      # pad bottom/top
    li x1, 0x0
    extrw 116, x1      # pad right/left
    li x1, 0x50006
    extrw 120, x1      # source height/width
    li x1, 0x50006
    extrw 124, x1      # destination height/width
    li x1, 0x0
    extrw 136, x1      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    end
