# MaxPool with a rectangular 2x3 window and strides 1x2 on a 3x5 matrix
# PDP1 works on fp16: input 3x5, window 2x3, stride 1x2, padding t/b/l/r = 0/0/0/0, mode 1 (max)
# Stride registers: {planes, rows per plane, pitch}; the pitch has to be a multiple of 16 fp16 elements (32 bytes).
# IN @0x4000: f16 [1.0, 2.0, 3.0, 4.0, 5.0]
# IN @0x4020: f16 [6.0, 7.0, 8.0, 9.0, 10.0]
# IN @0x4040: f16 [11.0, 12.0, 13.0, 14.0, 15.0]
# OUT @0x6000: f16 [8.0, 10.0]
# OUT @0x6020: f16 [13.0, 15.0]
    li x1, 0x6000                    # destination
    li x2, 0x4000                    # source
    li x3, 1
    li x4, 3
    li x5, 5
    li x6, 16                        # pitch
    li x7, 2
    li x8, 2
    li x9, 2
    li x10, 3
    li x11, 0x0                  # pad value (fp16 bits)
    li x12, 0x3800                   # 0.5 (fp16), the scale of mode 3
    li x13, 0
    li x14, 0
    li x15, 0
    li x16, 0
    ss_pack_shape x3, x3, x4, x5, s1    # 1 x 1 x H x W
    ss_pack_stride x3, x4, x6, s2       # source: 1 plane of H rows, pitch 16
    ss_pack_stride x3, x7, x6, s3       # destination: 1 plane of 2 rows, pitch 16
    ss_pack_shape x13, x14, x15, x16, s4
    mfu_pdp1_conf1 2, 1, s2, 1, s3        # stride_w, stride_h, source, mode, destination
    mfu_pdp1_conf2 x8, x7, x0, x0        # output width, output height
    mfu_pdp1_conf3 x0, x0, x11, s4       # pad value and padding
    mfu_pdp1_conf4 x10, x9, x12, 0, 0    # window width, window height, scale
    mfu_pdp1_conf_deq x0, x0, 0          # fp16 source
    mfu_pdp1_conf_quant x0, x0, 0        # fp16 destination
    mfu_pdp1_compute x1, x2, s1
    end
