# the same transpose for 16 bit elements (l2_datatype = 1): the stride_w unit is 16 elements
# IN @0x4000: u16[3,16] [
#   [257, 514, 771, 1028, 0*12]
#   [1285, 1542, 1799, 2056, 0*12]
#   [2313, 2570, 2827, 3084, 0*12]
# ]
# OUT @0x5000: u16[4,16] [
#   [257, 1285, 2313, 0xa5a5*13]
#   [514, 1542, 2570, 0xa5a5*13]
#   [771, 1799, 2827, 0xa5a5*13]
#   [1028, 2056, 3084, 0xa5a5*13]
# ]
    li x1, 0x5000
    li x2, 0x4000
    li x3, 1
    li x4, 3
    li x5, 4
    li x6, 16
    ss_pack_shape x3, x3, x4, x5, s1
    ss_pack_stride x3, x4, x6, s2
    ss_pack_stride x3, x5, x6, s3
    mfu_transpose_conf s3, s2, 1, 1
    mfu_transpose x1, x2, s1
    end
