# mfu_transpose type 1 = permute (0,1,3,2): transposes a 3x4 u8 matrix. Rows are 32 bytes apart on both sides, because
# transposes that move the innermost dimension need stride_w % 32 == 0 and a 16-byte aligned destination.
# IN @0x4000: u8[3,32] [
#   [1, 2, 3, 4, 0*28]
#   [5, 6, 7, 8, 0*28]
#   [9, 10, 11, 12, 0*28]
# ]
# OUT @0x5000: u8[4,32] [
#   [1, 5, 9, 0xa5*29]
#   [2, 6, 10, 0xa5*29]
#   [3, 7, 11, 0xa5*29]
#   [4, 8, 12, 0xa5*29]
# ]
    li x1, 0x5000
    li x2, 0x4000
    li x3, 1
    li x4, 3
    li x5, 4
    li x6, 32
    ss_pack_shape x3, x3, x4, x5, s1    # source shape 1 x 1 x 3 x 4
    ss_pack_stride x3, x4, x6, s2       # source: 3 rows, pitch 32
    ss_pack_stride x3, x5, x6, s3       # destination: 4 rows, pitch 32
    mfu_transpose_conf s3, s2, 0, 1
    mfu_transpose x1, x2, s1
    end
