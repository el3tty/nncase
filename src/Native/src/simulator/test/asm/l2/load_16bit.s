# l2_load with a non-zero l2_datatype copies 16 bit elements (row_len and pitches count halfwords)
# DDR_IN: u16 [4660, 48879, 7, 65535, 300, 2]
# OUT @0x4000: u16 [48879, 7, 65535, 300]
    li x1, 0x100002       # skip the first halfword
    li x2, 0x4000
    li x3, 1
    li x4, 4
    ss_pack_shape x3, x3, x3, x4, s1   # one row of 4 halfwords
    ss_pack_stride x3, x3, x4, s2
    l2_load_conf s2, s2, 1, 1
    l2_load x2, x1, s1
    end
