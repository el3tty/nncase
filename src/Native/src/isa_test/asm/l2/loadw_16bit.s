# 16 bit weights (l2_datatype 1, ddr_datatype 1), groups of 4 bytes = 2 elements, rows of 24 bytes
# l2_load_w_conf: rlen_decompressed = number of weight elements (the high half of the 64 bit wconf), l2_datatype, ddr_datatype (element
# size code 0: 1 byte, 1: 2 bytes, 2: 4 bytes, 3: packed 4 bit, 4: packed 6 bit), enable_decompress.
# l2_load_w copies the weights to the GLB in groups of rvalid_c_num + 1 BYTES (of the GLB data, whatever the element size), every group
# starts a new row of 24 bytes; with 16 bit weights an odd group size splits an element between two rows.
# DDR_IN: u16 [4660, 48879, 7, 65535, 300, 2, 1, 2]
# OUT @0x4000: u8 [52, 18, 239, 190, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 7, 0, 255, 255, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 44, 1, 2, 0, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 1, 0, 2, 0, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165, 165]
    li x1, 0x4000
    li x2, DDR_IN
    li x3, 8
    li x4, 8
    li x5, 3
    l2_load_w_conf x4, x3, 1, 1, 0
    l2_load_w x1, x2, x5
    end
