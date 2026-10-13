# 1x1 convolution with a different line fit per output channel (identity, ReLU6, 0.5 x + 1) and int8 output
# Convolution through the whole PU pipeline: l2-less (the tensors are already in GLB) dm_load_l1 / dm_load_w -> PU (TCU) -> PSUM
# -> ACT0 (per channel line fit: slope_lo, slope_hi, offset_lo, offset_hi, lower clamp, upper clamp, knee; the PSUM is
# scaled by 2^-shift first) -> dm_store_of -> GLB.
# Signed int8 values stored as uint8 (value + 128); the zero points (128) are subtracted by the PU (conf_deq type 1).
# Input 4x1x8, weights 3x4x1x1 (one 24 byte line per output channel: byte (kx + kw*ky) * 24 + channel),
# padding t/b/l/r 0/0/0/0, stride 1x1, output 3x1x8
# IN @0x4000: u8 [129, 130, 131, 132, 133, 134, 135, 136, 136, 135, 134, 133, 132, 131, 130, 129, 127, 126, 125, 124, 129, 130, 131, 132, 128, 129, 128, 129, 128, 129, 128, 129]
# IN @0x5000: u8 [129, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 130, 127, 131, 126, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
# IN @0x5800: u8 [128, 128, 128]
# IN @0x6000: f16 [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.5, 0.5, 1.0, 1.0, -65504.0, 65504.0, 0.0]
# OUT @0x7000: i8 [1, 2, 3, 4, 5, 6, 7, 8, 6, 6, 6, 6, 6, 6, 6, 6, -4, -4, -4, -4, 6, 8, 12, 14]
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
    li x23, 31                          # "all channels"
    li x20, 128                         # input zero point
    li x21, 1                           # weight groups
    li x22, 0x5800                      # weight zero points
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
    dm_load_w_conf_deq 0, 0, 1
    dm_load_w_conf2 0, 0, x21, x9       # rgroups, rgoc = output channels
    dm_load_w 0, 0, x2, x22, s5, 0
    pu_fetchif_conf1 0, 0, 1, 1, s2
    pu_fetchif_conf2 0, 0, x6, x0       # rgic = input channels
    pu_fetchif_conf3 0, 0, x0, x5, s7
    pu_fetchif_conf4 0, 0, x0, s6       # pad value 0 (the pad value is fed after the zero point subtraction)
    pu_fetchif_conf_deq 0, 0, x23, x20, 1
    pu_w_conf 0, 0, 1, 1
    pu_of_conf1 0, 0, x9, x0, s4
    pu_of_conf2 0, 0, x0, s3
    pu_compute_conf 0, 0, 0, 0, 1, 1, 0 # forward the result to ACT0
    dm_load_act0 0, 0, x3, x17, 0, 1
    dm_store_of_conf 0, 0, s4, 1
    dm_store_of 0, 0, x4, s3, 0
    act0_src1_conf 0, 0, 0, s3, 0
    act0_compute x0, 0, 0, 1, 1, 1
    pu_compute 0, 0
    end
