# 8 bit weights, groups of 4 elements
# l2_load_w_conf: rlen_decompressed = number of weight elements (the high half of the 64 bit wconf), l2_datatype, ddr_datatype (element
# size code 0: 1 byte, 1: 2 bytes, 2: 4 bytes, 3: packed 4 bit, 4: packed 6 bit), enable_decompress.
# l2_load_w copies the weights to the GLB in groups of rvalid_c_num + 1 elements, every group starts a new row of 24 elements.
# DDR_IN: i8 [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
# OUT @0x4000: u8 [1, 254, 3, 252, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 5, 250, 7, 248, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 9, 246, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165]
    li x1, 0x4000
    li x2, DDR_IN
    li x3, 10
    li x4, 10
    li x5, 3
    l2_load_w_conf x4, x3, 0, 0, 0
    l2_load_w x1, x2, x5
    end
