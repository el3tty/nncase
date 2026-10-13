# AI2D identity copy (planar, 1 channel) GLB -> GLB: M0 = M4 = 1024.0 (10 bit fixed point 1.0).
# The destination pitch is 8, so the last two bytes of every row keep the 0xA5 fill.
# Same register sequence as the compiler (GnneActionUpdater.UpdateAi2dResize): all 36 registers in ascending order, the
# pointers (0..7) with extraw, the others with extrw, then ai2d_compute + fence. The engine runs on ai2d_compute, not on the
# write to register 35. Pointers are byte offsets in GLB bank 0 (src_ind = dst_ind = 0).
# IN @0x4000: u8 [175, 196, 25, 246, 67, 211, 151, 103, 92, 185, 142, 23, 72, 89, 110, 42, 218, 136, 167, 230, 68, 176, 127, 135, 172, 0, 75, 55, 250, 6]
# OUT @0x5000: u8 [175, 196, 25, 246, 67, 211, 165, 165, 151, 103, 92, 185, 142, 23, 165, 165, 72, 89, 110, 42, 218, 136, 165, 165, 167, 230, 68, 176, 127, 135, 165, 165, 172, 0, 75, 55, 250, 6, 165, 165]
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
    li x1, 0x6
    extrw 32, x1      # src width/layout 0,1
    extrw 36, x0      # src width/layout 2,3
    li x1, 0x8
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
    extrw 96, x0      # const_pad ch0
    li x1, 0x4000000
    extrw 100, x1      # cmd_id, dst_ind/src_ind, const_pad ch1..3
    extrw 104, x0      # (unused)
    extrw 108, x0      # (unused)
    extrw 112, x0      # pad bottom/top
    extrw 116, x0      # pad right/left
    li x1, 0x50006
    extrw 120, x1      # source height/width
    li x1, 0x50006
    extrw 124, x1      # destination height/width
    extrw 128, x0      # M2
    extrw 132, x0      # M5
    extrw 136, x0      # src_y/src_x
    li x1, 0x80000000
    extrw 140, x1      # dst_y/dst_x + calc_enable: starts the engine
    ai2d_compute
    fence
    end
