# PDP1 with a uint8 source and destination: dequantise (q - zero_point) * scale, pool in fp16, quantise again
# deq scale = 0.5, zero point = 0, quant scale = 1.0, zero point = 0. u8 rows have to be a multiple of 32 bytes apart.
# IN @0x4000: u8 [2, 10, 4]
# IN @0x4020: u8 [200, 8, 6]
# IN @0x4040: u8 [100, 254, 12]
# OUT @0x6000: u8 [100, 5]
# OUT @0x6020: u8 [127, 127]
    li x1, 0x6000
    li x2, 0x4000
    li x3, 1
    li x4, 3
    li x5, 2
    li x6, 32
    li x9, 0x3800                     # 0.5
    li x10, 0x3c00                    # 1.0
    ss_pack_shape x3, x3, x4, x4, s1
    ss_pack_stride x3, x4, x6, s2
    ss_pack_stride x3, x5, x6, s3
    ss_pack_shape x0, x0, x0, x0, s4
    mfu_pdp1_conf1 1, 1, s2, 1, s3
    mfu_pdp1_conf2 x5, x5, x0, x0
    mfu_pdp1_conf3 x0, x0, x0, s4
    mfu_pdp1_conf4 x5, x5, x0, 0, 0
    mfu_pdp1_conf_deq x9, x0, 1       # source type 1 = uint8
    mfu_pdp1_conf_quant x10, x0, 1    # destination type 1 = uint8
    mfu_pdp1_compute x1, x2, s1
    end
