# 3x3 convolution with stride 2 and no padding (5x7 -> 2x3); the windows cover the input exactly
# Convolution through the whole PU pipeline: l2-less (the tensors are already in GLB) dm_load_l1 / dm_load_w -> PU (TCU) -> PSUM
# -> ACT0 (per channel line fit: slope_lo, slope_hi, offset_lo, offset_hi, lower clamp, upper clamp, knee; the PSUM is
# scaled by 2^-shift first) -> dm_store_of -> GLB.
# Signed int8 values stored as uint8 (value + 128); the zero points (128) are subtracted by the PU (conf_deq type 1).
# Input 2x5x7, weights 2x2x3x3 (one 216 byte line per output channel: byte (kx + kw*ky) * 24 + channel),
# padding t/b/l/r 0/0/0/0, stride 2x2, output 2x2x3
# IN @0x4000: u8[2,5,7] [
#   [[126, 128, 120, 122, 130, 127, 135]
#    [135, 127, 129, 127, 123, 132, 130]
#    [134, 134, 120, 128, 127, 127, 121]
#    [121, 133, 123, 120, 128, 133, 133]
#    [126, 124, 125, 126, 122, 132, 131]]
#   [[125, 131, 127, 133, 128, 131, 132]
#    [131, 124, 134, 124, 127, 127, 133]
#    [124, 129, 120, 132, 122, 120, 133]
#    [130, 127, 132, 121, 127, 129, 128]
#    [130, 132, 124, 120, 132, 121, 134]]
# ]
# IN @0x5000: u8[2,9,24] [
#   [[131, 126, 128*22]
#    [128*24]
#    [131, 129, 128*22]
#    [130, 127, 128*22]
#    [126, 129, 128*22]
#    [126, 125, 128*22]
#    [128, 130, 128*22]
#    [129, 128*23]
#    [125, 126, 128*22]]
#   [[127, 126, 128*22]
#    [125, 128*23]
#    [129, 131, 128*22]
#    [126, 130, 128*22]
#    [127, 129, 128*22]
#    [127, 128*23]
#    [126, 130, 128*22]
#    [125, 126, 128*22]
#    [131, 128*23]]
# ]
# IN @0x5800: u8 [128, 128]
# IN @0x6000: f16[2,7] [
#   [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0]*2
# ]
# OUT @0x7000: f16[2,2,3] [
#   [[2.0, -10.0, -8.0]
#    [-18.0, -19.0, -34.0]]
#   [[-79.0, 31.0, 9.0]
#    [-28.0, 26.0, 44.0]]
# ]
    li x1, 0x4000                       # input
    li x2, 0x5000                       # weights
    li x3, 0x6000                       # ACT0 parameters
    li x4, 0x7000                       # output
    li x5, 1
    li x6, 2
    li x7, 5
    li x8, 7
    li x9, 2
    li x10, 2
    li x11, 3
    li x12, 216                  # bytes per weight line
    li x13, 0
    li x14, 0
    li x15, 0
    li x16, 0
    li x17, 112
    li x23, 31                          # "all channels"
    li x20, 128                         # input zero point
    li x21, 1                           # weight groups
    li x22, 0x5800                      # weight zero points
    ss_pack_shape x5, x6, x7, x8, s1    # input 2x5x7
    ss_pack_stride x6, x7, x8, s2       # input strides: planes, rows per plane, pitch
    ss_pack_shape x5, x9, x10, x11, s3  # output 2x2x3
    ss_pack_stride x9, x10, x11, s4     # output strides
    ss_pack_stride x0, x0, x9, s5       # r_iochannels: low half = number of weight lines
    ss_pack_shape x13, x14, x15, x16, s6
    li x18, 5
    li x19, 7
    ss_pack_shape x5, x6, x18, x19, s7  # pu_fetchif_conf3 wants the padded input shape
    dm_load_l1_conf 0, 0, s2, 0, 0
    dm_load_l1 0, 0, x1, x0, s1, 0
    dm_load_w_conf 0, 0, 3, 3, x12
    dm_load_w_conf_deq 0, 0, 1
    dm_load_w_conf2 0, 0, x21, x9       # rgroups, rgoc = output channels
    dm_load_w 0, 0, x2, x22, s5, 0
    pu_fetchif_conf1 0, 0, 2, 2, s2
    pu_fetchif_conf2 0, 0, x6, x0       # rgic = input channels
    pu_fetchif_conf3 0, 0, x0, x5, s7
    pu_fetchif_conf4 0, 0, x0, s6       # pad value 0 (the pad value is fed after the zero point subtraction)
    pu_fetchif_conf_deq 0, 0, x23, x20, 1
    pu_w_conf 0, 0, 3, 3
    pu_of_conf1 0, 0, x9, x0, s4
    pu_of_conf2 0, 0, x0, s3
    pu_compute_conf 0, 0, 0, 0, 1, 1, 0 # forward the result to ACT0
    dm_load_act0 0, 0, x3, x17, 0, 1
    dm_store_of_conf 0, 0, s4, 2   # the board stores 2 byte elements for type 1, so int8 uses the 1 byte mode
    dm_store_of 0, 0, x4, s3, 0
    act0_src1_conf 0, 0, 0, s3, 0
    act0_compute x0, 0, 0, 1, 2, 1
    pu_compute 0, 0
    end
