# mfu_transpose type 2 = permute (0,2,1,3): [1,2,3,4] -> [1,3,2,4]; the innermost dimension is kept
# IN @0x4000: u8 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# OUT @0x5000: u8 [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23]
    li x1, 0x5000
    li x2, 0x4000
    li x3, 1
    li x4, 2
    li x5, 3
    li x6, 4
    ss_pack_shape x3, x4, x5, x6, s1    # source shape 1 x 2 x 3 x 4
    ss_pack_stride x4, x5, x6, s2       # source: {2 planes, 3 rows, pitch 4}
    ss_pack_stride x5, x4, x6, s3       # destination: {3 planes, 2 rows, pitch 4}
    mfu_transpose_conf s3, s2, 0, 2
    mfu_transpose x1, x2, s1
    end
