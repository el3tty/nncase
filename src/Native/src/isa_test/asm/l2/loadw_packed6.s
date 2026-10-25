# packed 6 bit weights (ddr_datatype 4): four values per three bytes, every value ends up in the upper 6 bits of a byte (value * 4)
# l2_load_w_conf: rlen_decompressed = number of weight elements (the high half of the 64 bit wconf), l2_datatype, ddr_datatype (element
# size code 0: 1 byte, 1: 2 bytes, 2: 4 bytes, 3: packed 4 bit, 4: packed 6 bit), enable_decompress.
# l2_load_w copies the weights to the GLB in groups of rvalid_c_num + 1 BYTES (of the GLB data, whatever the element size), every group
# starts a new row of 24 bytes; with 16 bit weights an odd group size splits an element between two rows.
# DDR_IN: u8 [0x81, 0xf0, 0x83, 0x85, 0x01]
# OUT @0x4000: u8 [0x04, 0x08, 0xfc, 0x80, 0x14, 0x18, 0xa5*18]
    li x1, 0x4000
    li x2, DDR_IN
    li x3, 6
    li x4, 6
    li x5, 5
    l2_load_w_conf x4, x3, 0, 4, 0
    l2_load_w x1, x2, x5
    end
