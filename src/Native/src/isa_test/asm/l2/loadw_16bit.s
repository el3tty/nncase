# 16 bit weights (l2_datatype 1, ddr_datatype 1), groups of 3 elements; the row pitch of 24 is counted in elements
# l2_load_w_conf: rlen_decompressed = number of weight elements (the high half of the 64 bit wconf), l2_datatype, ddr_datatype (element
# size code 0: 1 byte, 1: 2 bytes, 2: 4 bytes, 3: packed 4 bit, 4: packed 6 bit), enable_decompress.
# l2_load_w copies the weights to the GLB in groups of rvalid_c_num + 1 elements, every group starts a new row of 24 elements.
# DDR_IN: u16 [4660, 48879, 7, 65535, 300, 2, 1, 2]
# OUT @0x4000: u16 [4660, 48879, 7, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 65535, 300, 2, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 1, 2, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405]
    li x1, 0x4000
    li x2, DDR_IN
    li x3, 8
    li x4, 8
    li x5, 2
    l2_load_w_conf x4, x3, 1, 1, 0
    l2_load_w x1, x2, x5
    end
