# packed 4 bit weights (ddr_datatype 3): every value is stored in the upper nibble of a byte
# l2_load_w_conf: rlen_decompressed = number of weight elements (the high half of the 64 bit wconf), l2_datatype, ddr_datatype (element
# size code 0: 1 byte, 1: 2 bytes, 2: 4 bytes, 3: packed 4 bit, 4: packed 6 bit), enable_decompress.
# l2_load_w copies the weights to the GLB in groups of rvalid_c_num + 1 BYTES (of the GLB data, whatever the element size), every group
# starts a new row of 24 bytes; with 16 bit weights an odd group size splits an element between two rows.
# DDR_IN: u8 [0x21, 0x43, 0xf5, 0x08]
# OUT @0x4000: u8 [0x10, 0x20, 0x30, 0x40, 0x50, 0xf0, 0x80, 0xa5*17]
    li x1, 0x4000
    li x2, DDR_IN
    li x3, 7
    li x4, 7
    li x5, 6
    l2_load_w_conf x4, x3, 0, 3, 0
    l2_load_w x1, x2, x5
    end
