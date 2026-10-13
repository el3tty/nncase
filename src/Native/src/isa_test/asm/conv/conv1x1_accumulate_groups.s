# Accumulating convolution: 6 input channels are processed as two groups of 3. The first pu_compute writes the int32 PSUM
# (dest_target 0), the second one runs with load_psum = 1 so that its products are added to the PSUM, and forwards the sum to ACT0.
# Signed int8 values stored as uint8 (value + 128); the zero points (128) are subtracted by the PU.
# IN @0x4000: u8 [120, 127, 136, 120, 119, 133, 129, 129, 131, 137, 120, 122, 130, 124, 125, 128, 135, 136, 127, 132, 120, 132, 131, 137]
# IN @0x4400: u8 [127, 123, 122, 119, 129, 135, 123, 134, 124, 127, 123, 127, 119, 120, 121, 132, 131, 132, 126, 137, 134, 135, 133, 125]
# IN @0x5000: u8 [126, 126, 126, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 127, 124, 125, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 132, 124, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
# IN @0x5400: u8 [127, 126, 125, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 124, 126, 124, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 131, 124, 126, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
# IN @0x5800: u8 [128, 128, 128]
# IN @0x6000: f16 [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0]
# OUT @0x7000: f16 [-4.0, -37.0, 24.0, 4.0, 29.0, -22.0, 2.0, -25.0, -25.0, -53.0, 69.0, 22.0, 15.0, -41.0, 16.0, -48.0, -9.0, -15.0, -22.0, -81.0, 67.0, 7.0, -21.0, -28.0]
    li x1, 0x4000
    li x2, 0x4400
    li x3, 0x5000
    li x4, 0x5400
    li x5, 0x6000
    li x6, 0x7000
    li x7, 1
    li x8, 3
    li x9, 8
    li x11, 3
    li x12, 24
    li x17, 112
    li x13, 31                          # "all channels"
    li x14, 128                         # input zero point
    li x15, 0x5800                      # weight zero points
    ss_pack_shape x7, x8, x7, x9, s1    # one group: 3 x 1 x 8
    ss_pack_stride x8, x7, x9, s2
    ss_pack_shape x7, x11, x7, x9, s3   # output 3 x 1 x 8
    ss_pack_stride x11, x7, x9, s4
    ss_pack_stride x0, x0, x11, s5
    ss_pack_shape x0, x0, x0, x0, s6
    dm_load_l1_conf 0, 0, s2, 0, 0
    dm_load_w_conf 0, 0, 1, 1, x12
    dm_load_w_conf_deq 0, 0, 1
    dm_load_w_conf2 0, 0, x7, x11       # rgroups, rgoc
    pu_fetchif_conf1 0, 0, 1, 1, s2
    pu_fetchif_conf2 0, 0, x8, x0       # rgic = channels of a group
    pu_fetchif_conf3 0, 0, x0, x7, s1
    pu_fetchif_conf4 0, 0, x0, s6
    pu_fetchif_conf_deq 0, 0, x13, x14, 1
    pu_w_conf 0, 0, 1, 1
    pu_of_conf1 0, 0, x11, x0, s4
    pu_of_conf2 0, 0, x0, s3
    # group 0: overwrite the PSUM, keep the result there
    dm_load_l1 0, 0, x1, x0, s1, 0
    dm_load_w 0, 0, x3, x15, s5, 0
    pu_compute_conf 0, 0, 0, 0, 0, 1, 0
    pu_compute 0, 0
    # group 1: accumulate and forward to ACT0
    dm_load_l1 0, 0, x2, x0, s1, 0
    dm_load_w 0, 0, x4, x15, s5, 0
    pu_compute_conf 0, 0, 1, 0, 1, 1, 0
    dm_load_act0 0, 0, x5, x17, 0, 1
    dm_store_of_conf 0, 0, s4, 2
    dm_store_of 0, 0, x6, s3, 0
    act0_src1_conf 0, 0, 0, s3, 0
    act0_compute x0, 0, 0, 1, 2, 1
    pu_compute 0, 0
    end
