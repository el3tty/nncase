# float32 weights converted to float16 (l2_datatype 1, ddr_datatype 2)
# l2_load_w_conf: rlen_decompressed = number of weight elements (the high half of the 64 bit wconf), l2_datatype, ddr_datatype (element
# size code 0: 1 byte, 1: 2 bytes, 2: 4 bytes, 3: packed 4 bit, 4: packed 6 bit), enable_decompress.
# l2_load_w copies the weights to the GLB in groups of rvalid_c_num + 1 elements, every group starts a new row of 24 elements.
# DDR_IN: f32 [1.5, -2.0, 0.10000000149011612, 65504.0, 9.99999993922529e-09, 3.141590118408203]
# OUT @0x4000: u16 [15872, 49152, 11878, 31743, 0, 16968, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405, 42405]
    li x1, 0x4000
    li x2, DDR_IN
    li x3, 6
    li x4, 6
    li x5, 5
    l2_load_w_conf x4, x3, 1, 2, 0
    l2_load_w x1, x2, x5
    end
