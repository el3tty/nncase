# l2_store of a dense 4x4 tile to DDR rows of 8 bytes; the other 4 bytes of every DDR row keep the 0xA5 fill
# IN @0x4000: u8[4,4] [
#   [1, 2, 3, 4]
#   [5, 6, 7, 8]
#   [9, 10, 11, 12]
#   [13, 14, 15, 16]
# ]
# DDR_OUT: u8[4,8] [
#   [1, 2, 3, 4, 0xa5*4]
#   [5, 6, 7, 8, 0xa5*4]
#   [9, 10, 11, 12, 0xa5*4]
#   [13, 14, 15, 16, 0xa5*4]
# ]
    li x1, DDR_OUT      # DDR destination offset
    li x2, 0x4000           # GLB source
    li x3, 1
    li x4, 4
    li x5, 8
    ss_pack_shape x3, x3, x4, x4, s1
    ss_pack_stride x3, x4, x5, s2     # destination (DDR): pitch 8
    ss_pack_stride x3, x4, x4, s3     # source (GLB): pitch 4
    l2_store_conf s2, s3, 0, 0
    l2_store x1, x2, s1
    end
