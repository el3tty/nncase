# l2_load of 2 planes: the source plane stride is pitch * dim1 = 4 * 3, the destination plane stride is 3 * 2
# DDR_IN: u8 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# OUT @0x4000: u8 [0, 1, 2, 4, 5, 6, 12, 13, 14, 16, 17, 18]
    li x1, 0x100000
    li x2, 0x4000
    li x3, 1
    li x4, 2          # planes / rows
    li x5, 3
    li x6, 4
    ss_pack_shape x3, x4, x4, x5, s1   # 1 batch, 2 planes, 2 rows, 3 elements
    ss_pack_stride x3, x4, x5, s2      # destination: 2 rows per plane, pitch 3
    ss_pack_stride x3, x5, x6, s3      # source: 3 rows per plane, pitch 4
    l2_load_conf s2, s3, 0, 0
    l2_load x2, x1, s1
    end
