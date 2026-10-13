# AI2D identity copy (the two pointer registers are written with extraw, which only reaches byte addresses 0..28)
# AI2D identity copy (planar, 1 channel) from DDR (src_ind=1) to GLB: M0 = M4 = 1024.0 (10 bit fixed point 1.0).
# The destination pitch is 8, so the last two bytes of every row keep the 0xA5 fill.
# Registers are written with extrw (byte address of the 32-bit register); the write to byte address 140 runs the engine.
# DDR_IN: u8 [175, 196, 25, 246, 67, 211, 151, 103, 92, 185, 142, 23, 72, 89, 110, 42, 218, 136, 167, 230, 68, 176, 127, 135, 172, 0, 75, 55, 250, 6]
# OUT @0x4000: u8 [175, 196, 25, 246, 67, 211, 165, 165, 151, 103, 92, 185, 142, 23, 165, 165, 72, 89, 110, 42, 218, 136, 165, 165, 167, 230, 68, 176, 127, 135, 165, 165, 172, 0, 75, 55, 250, 6, 165, 165]
    li x1, 0x100000
    extraw 0, x1      # src_ch0_ptr
    li x1, 0x4000
    extraw 16, x1      # dst_ch0_ptr
    li x1, 0x6
    extrw 32, x1      # src width/layout 0,1
    li x1, 0x0
    extrw 36, x1      # src width/layout 2,3
    li x1, 0x8
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
    li x1, 0x50006
    extrw 120, x1      # source height/width
    li x1, 0x50006
    extrw 124, x1      # destination height/width
    li x1, 0x0
    extrw 136, x1      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    end
