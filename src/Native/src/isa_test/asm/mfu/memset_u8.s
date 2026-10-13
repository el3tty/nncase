# mfu_memset fills a strided 3 x 4 tile with the low byte of the register (0x1234 -> 0x34), destination pitch 8
# Stride register = {dim0, dim1, pitch}, shape register = {d0, d1, d2, d3}
# OUT @0x4000: u8 [52, 52, 52, 52, 165, 165, 165, 165, 52, 52, 52, 52, 165, 165, 165, 165, 52, 52, 52, 52, 165, 165, 165, 165]
    li x1, 0x4000
    li x2, 0x1234
    li x3, 1
    li x4, 3
    li x5, 4
    li x6, 8
    ss_pack_stride x3, x4, x6, s1
    ss_pack_shape x3, x3, x4, x5, s2
    mfu_memset x1, x2, s1, s2, 0
    end
