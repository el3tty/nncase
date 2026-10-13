# MaxPool of a 3x3 matrix with a 2x2 kernel, stride 1 (output 2x2)
# PDP1 works on fp16: input 3x3, window 2x2, stride 1x1, padding t/b/l/r = 0/0/0/0, mode 1 (max)
# Stride registers: {planes, rows per plane, pitch}; the pitch has to be a multiple of 16 fp16 elements (32 bytes).
# IN @0x4000: f16 [1.0, 5.0, 2.0]
# IN @0x4020: f16 [-3.0, 4.0, 0.5]
# IN @0x4040: f16 [7.0, -8.0, 6.0]
# OUT @0x6000: f16 [5.0, 5.0]
# OUT @0x6020: f16 [7.0, 6.0]
    li x1, 0x6000                    # destination
    li x2, 0x4000                    # source
    li x3, 1
    li x4, 3
    li x5, 3
    li x6, 16                        # pitch
    li x7, 2
    li x8, 2
    li x9, 2
    li x10, 2
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
    mfu_pdp1_conf1 1, 1, s2, 1, s3        # stride_w, stride_h, source, mode, destination
    mfu_pdp1_conf2 x8, x7, x0, x0        # output width, output height
    mfu_pdp1_conf3 x0, x0, x11, s4       # pad value and padding
    mfu_pdp1_conf4 x10, x9, x12, 0, 0    # window width, window height, scale
    mfu_pdp1_conf_deq x0, x0, 0          # fp16 source
    mfu_pdp1_conf_quant x0, x0, 0        # fp16 destination
    mfu_pdp1_compute x1, x2, s1
    end
