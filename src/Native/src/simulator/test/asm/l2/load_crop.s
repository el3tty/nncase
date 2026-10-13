# l2_load of a 3x4 window out of a 4x6 u8 matrix (source pitch 6, destination pitch 4)
# Stride register = {dim0, dim1, pitch} in elements; shape register = {batch, plane, rows, row_len}.
# DDR_IN: u8 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# OUT @0x4000: u8 [8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23]
    li x1, 0x100007       # DDR offset of element (1,1)
    li x2, 0x4000          # GLB destination
    li x3, 1
    li x4, 3
    li x5, 4
    li x6, 6
    ss_pack_shape x3, x3, x4, x5, s1   # 1 x 1 x 3 rows x 4 elements
    ss_pack_stride x3, x4, x5, s2      # destination: dim1 = 3 rows, pitch 4
    ss_pack_stride x3, x5, x6, s3      # source: dim1 = 4 rows, pitch 6
    l2_load_conf s2, s3, 0, 0          # 8 bit elements
    l2_load x2, x1, s1
    end
