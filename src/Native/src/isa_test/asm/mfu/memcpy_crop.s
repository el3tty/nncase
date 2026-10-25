# mfu_memcpy copies a 3x4 window of a 4x6 u8 matrix (GLB to GLB), operands: rd, rs, dst stride, src stride, shape
# IN @0x4000: u8[4,6] [
#   [1, 2, 3, 4, 5, 6]
#   [7, 8, 9, 10, 11, 12]
#   [13, 14, 15, 16, 17, 18]
#   [19, 20, 21, 22, 23, 24]
# ]
# OUT @0x5000: u8[3,4] [
#   [8, 9, 10, 11]
#   [14, 15, 16, 17]
#   [20, 21, 22, 23]
# ]
    li x1, 0x5000
    li x2, 0x4007
    li x3, 1
    li x4, 3
    li x5, 4
    li x6, 6
    ss_pack_shape x3, x3, x4, x5, s1
    ss_pack_stride x3, x4, x5, s2      # destination: 3 rows, pitch 4
    ss_pack_stride x3, x5, x6, s3      # source: 4 rows, pitch 6
    mfu_memcpy x1, x2, s2, s3, s1
    end
