# 3x3 convolution, 2 -> 2 channels on a 5x6 image with padding 1 (output 5x6)
# Convolution through the whole PU pipeline: l2-less (the tensors are already in GLB) dm_load_l1 / dm_load_w -> PU (TCU) -> PSUM
# -> ACT0 (per channel line fit: slope_lo, slope_hi, offset_lo, offset_hi, lower clamp, upper clamp, knee; the PSUM is
# scaled by 2^-shift first) -> dm_store_of -> GLB.
# Input 2x5x6 int8, weights 2x2x3x3 int8 (one 216 byte line per output channel: byte (kx + kw*ky) * 24 + channel),
# padding t/b/l/r 1/1/1/1, stride 1x1, output 2x5x6
# IN @0x4000: i8 [-3, 3, 4, 0, 7, 1, 3, -3, 7, -8, -8, -7, 4, -1, 5, 4, -2, 1, -6, -4, 6, -3, -6, -4, 3, 4, 2, 4, 5, 6, -6, -4, -1, -1, 1, -7, -1, 6, -8, -2, 5, 1, 1, -1, 7, 5, -2, 1, -7, -8, 4, -7, 0, 0, 5, -5, 7, 2, 1, 0]
# IN @0x5000: i8 [3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# IN @0x6000: f16 [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0]
# OUT @0x7000: f16 [6.0, -25.0, 51.0, -19.0, 14.0, 7.0, -5.0, 15.0, -15.0, 101.0, 19.0, 10.0, 17.0, -43.0, -61.0, 51.0, -48.0, -41.0, 34.0, -3.0, 11.0, 13.0, 18.0, 2.0, -14.0, -19.0, -10.0, -30.0, -22.0, -21.0, -22.0, -7.0, -30.0, -44.0, -4.0, 25.0, -18.0, 15.0, -16.0, -32.0, 8.0, 17.0, 24.0, 24.0, -89.0, 48.0, 31.0, 43.0, -20.0, 19.0, -26.0, -32.0, -5.0, -7.0, -12.0, 43.0, -39.0, -8.0, 17.0, 4.0]
    li x1, 0x4000                       # input
    li x2, 0x5000                       # weights
    li x3, 0x6000                       # ACT0 parameters
    li x4, 0x7000                       # output
    li x5, 1
    li x6, 2
    li x7, 5
    li x8, 6
    li x9, 2
    li x10, 5
    li x11, 6
    li x12, 216                  # bytes per weight line
    li x13, 1
    li x14, 1
    li x15, 1
    li x16, 1
    li x17, 112
    ss_pack_shape x5, x6, x7, x8, s1    # input 2x5x6
    ss_pack_stride x6, x7, x8, s2       # input strides: planes, rows per plane, pitch
    ss_pack_shape x5, x9, x10, x11, s3  # output 2x5x6
    ss_pack_stride x9, x10, x11, s4     # output strides
    ss_pack_stride x0, x0, x9, s5       # r_iochannels: low half = number of weight lines
    ss_pack_shape x13, x14, x15, x16, s6
    li x18, 7
    li x19, 8
    ss_pack_shape x5, x6, x18, x19, s7  # pu_fetchif_conf3 wants the padded input shape
    dm_load_l1_conf 0, 0, s2, 0, 0
    dm_load_l1 0, 0, x1, x0, s1, 0
    dm_load_w_conf 0, 0, 3, 3, x12
    dm_load_w 0, 0, x2, x2, s5, 0
    pu_fetchif_conf1 0, 0, 1, 1, s2
    pu_fetchif_conf3 0, 0, x0, x5, s7
    pu_fetchif_conf4 0, 0, x0, s6
    pu_w_conf 0, 0, 3, 3
    pu_of_conf1 0, 0, x0, x0, s4
    pu_of_conf2 0, 0, x0, s3
    pu_compute_conf 0, 0, 0, 0, 1, 1, 0 # forward the result to ACT0
    dm_load_act0 0, 0, x3, x17, 0, 1
    dm_store_of_conf 0, 0, s4, 2
    dm_store_of 0, 0, x4, s3, 0
    act0_src1_conf 0, 0, 0, s3, 0
    act0_compute x0, 0, 0, 1, 2, 1
    pu_compute 0, 0
    end
