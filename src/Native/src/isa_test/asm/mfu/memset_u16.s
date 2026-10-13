# mfu_memset with l2_datatype = 1 writes 16 bit elements (the destination has to be 2-byte aligned)
# the tile starts at the second halfword and is one row of 4 elements
# OUT @0x4000: u16 [42405, 48879, 48879, 48879, 48879, 42405]
    li x1, 0x4002
    li x2, 0xBEEF
    li x3, 1
    li x4, 4
    li x6, 4
    ss_pack_stride x3, x3, x6, s1
    ss_pack_shape x3, x3, x3, x4, s2
    mfu_memset x1, x2, s1, s2, 1
    end
