# compressed weights (enable_decompress = 1): 16 byte blocks with a presence mask, missing bytes are zero; groups of 16
# l2_load_w_conf: rlen_decompressed = number of weight elements (the high half of the 64 bit wconf), l2_datatype, ddr_datatype (element
# size code 0: 1 byte, 1: 2 bytes, 2: 4 bytes, 3: packed 4 bit, 4: packed 6 bit), enable_decompress.
# l2_load_w copies the weights to the GLB in groups of rvalid_c_num + 1 BYTES (of the GLB data, whatever the element size), every group
# starts a new row of 24 bytes; with 16 bit weights an odd group size splits an element between two rows.
# DDR_IN: u8 [0x19, 0x80, 0x0b, 0x16, 0x21, 0x2c, 0x03, 0x40, 0x37, 0x42, 0x4d]
# OUT @0x4000: u8[2,24] [
#   [0x0b, 0x00, 0x00, 0x16, 0x21, 0x00*10, 0x2c, 0xa5*8]
#   [0x37, 0x42, 0x00*12, 0x4d, 0x00, 0xa5*8]
# ]
    li x1, 0x4000
    li x2, DDR_IN
    li x3, 32
    li x4, 11
    li x5, 15
    l2_load_w_conf x4, x3, 0, 0, 1
    l2_load_w x1, x2, x5
    end
