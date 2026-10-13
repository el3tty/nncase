# 1x1 convolution, 4 -> 3 channels, identity line fit, fp16 output
# 1x1 convolution whose PSUM is handed to ACT0 explicitly with pu_forward_psum instead of dest_target = 1.
# Convolution through the whole PU pipeline: l2-less (the tensors are already in GLB) dm_load_l1 / dm_load_w -> PU (TCU) -> PSUM
# -> ACT0 (per channel line fit: slope_lo, slope_hi, offset_lo, offset_hi, lower clamp, upper clamp, knee; the PSUM is
# scaled by 2^-shift first) -> dm_store_of -> GLB.
# Input 4x1x8 int8, weights 3x4x1x1 int8 (one 24 byte line per output channel: byte (kx + kw*ky) * 24 + channel),
# padding t/b/l/r 0/0/0/0, stride 1x1, output 3x1x8
# IN @0x4000: i8 [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1, -1, -2, -3, -4, 1, 2, 3, 4, 0, 1, 0, 1, 0, 1, 0, 1]
# IN @0x5000: i8 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1, 3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# IN @0x6000: f16 [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0]
# OUT @0x7000: f16 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0, 6.0, 6.0, 10.0, 12.0, 12.0, 14.0, -9.0, -11.0, -9.0, -11.0, 9.0, 13.0, 21.0, 25.0]
    li x1, 0x4000                       # input
    li x2, 0x5000                       # weights
    li x3, 0x6000                       # ACT0 parameters
    li x4, 0x7000                       # output
    li x5, 1
    li x6, 4
    li x7, 1
    li x8, 8
    li x9, 3
    li x10, 1
    li x11, 8
    li x12, 24                  # bytes per weight line
    li x13, 0
    li x14, 0
    li x15, 0
    li x16, 0
    li x17, 112
    ss_pack_shape x5, x6, x7, x8, s1    # input 4x1x8
    ss_pack_stride x6, x7, x8, s2       # input strides: planes, rows per plane, pitch
    ss_pack_shape x5, x9, x10, x11, s3  # output 3x1x8
    ss_pack_stride x9, x10, x11, s4     # output strides
    ss_pack_stride x0, x0, x9, s5       # r_iochannels: low half = number of weight lines
    ss_pack_shape x13, x14, x15, x16, s6
    li x18, 1
    li x19, 8
    ss_pack_shape x5, x6, x18, x19, s7  # pu_fetchif_conf3 wants the padded input shape
    dm_load_l1_conf 0, 0, s2, 0, 0
    dm_load_l1 0, 0, x1, x0, s1, 0
    dm_load_w_conf 0, 0, 1, 1, x12
    dm_load_w 0, 0, x2, x2, s5, 0
    pu_fetchif_conf1 0, 0, 1, 1, s2
    pu_fetchif_conf3 0, 0, x0, x5, s7
    pu_fetchif_conf4 0, 0, x0, s6
    pu_w_conf 0, 0, 1, 1
    pu_of_conf1 0, 0, x0, x0, s4
    pu_of_conf2 0, 0, x0, s3
    pu_compute_conf 0, 0, 0, 0, 0, 1, 0 # dest_target 0: the result stays in PSUM_L1
    pu_compute 0, 0                     # runs immediately, int32 sums are left in PSUM_L1
    dm_load_act0 0, 0, x3, x17, 0, 1
    dm_store_of_conf 0, 0, s4, 2
    dm_store_of 0, 0, x4, s3, 0         # no compute is waiting: the store just stays queued
    act0_src1_conf 0, 0, 0, s3, 0
    act0_compute x0, 0, 0, 1, 2, 1
    li x18, 8
    pu_forward_psum 0, 0, x0, x18       # copy 8 words per channel from PSUM_L1 offset 0 into ACT0 and run it
    end
