# l2_load with batch = 2: dim0 of the stride register is the number of planes per batch (batch stride = plane stride * dim0)
# DDR_IN: u8 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# OUT @0x4000: u8 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    li x1, DDR_IN
    li x2, 0x4000
    li x3, 2
    ss_pack_shape x3, x3, x3, x3, s1   # 2 x 2 x 2 x 2
    ss_pack_stride x3, x3, x3, s2      # dim0 = 2 planes, dim1 = 2 rows, pitch 2 (both sides dense)
    l2_load_conf s2, s2, 0, 0
    l2_load x2, x1, s1
    end
