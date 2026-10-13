# Accumulating convolution: 6 input channels are processed as two groups of 3. The first pu_compute writes the int32 PSUM
# (dest_target 0), the second one runs with load_psum = 1 so that its products are added to the PSUM, and forwards the sum to ACT0.
# IN @0x4000: i8 [-8, -1, 8, -8, -9, 5, 1, 1, 3, 9, -8, -6, 2, -4, -3, 0, 7, 8, -1, 4, -8, 4, 3, 9]
# IN @0x4400: i8 [-1, -5, -6, -9, 1, 7, -5, 6, -4, -1, -5, -1, -9, -8, -7, 4, 3, 4, -2, 9, 6, 7, 5, -3]
# IN @0x5000: i8 [-2, -2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -4, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# IN @0x5400: i8 [-1, -2, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -4, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
    ss_pack_shape x7, x8, x7, x9, s1    # one group: 3 x 1 x 8
    ss_pack_stride x8, x7, x9, s2
    ss_pack_shape x7, x11, x7, x9, s3   # output 3 x 1 x 8
    ss_pack_stride x11, x7, x9, s4
    ss_pack_stride x0, x0, x11, s5
    ss_pack_shape x0, x0, x0, x0, s6
    dm_load_l1_conf 0, 0, s2, 0, 0
    dm_load_w_conf 0, 0, 1, 1, x12
    pu_fetchif_conf1 0, 0, 1, 1, s2
    pu_fetchif_conf3 0, 0, x0, x7, s1
    pu_fetchif_conf4 0, 0, x0, s6
    pu_w_conf 0, 0, 1, 1
    pu_of_conf1 0, 0, x0, x0, s4
    pu_of_conf2 0, 0, x0, s3
    # group 0: overwrite the PSUM, keep the result there
    dm_load_l1 0, 0, x1, x0, s1, 0
    dm_load_w 0, 0, x3, x3, s5, 0
    pu_compute_conf 0, 0, 0, 0, 0, 1, 0
    pu_compute 0, 0
    # group 1: accumulate and forward to ACT0
    dm_load_l1 0, 0, x2, x0, s1, 0
    dm_load_w 0, 0, x4, x4, s5, 0
    pu_compute_conf 0, 0, 1, 0, 1, 1, 0
    dm_load_act0 0, 0, x5, x17, 0, 1
    dm_store_of_conf 0, 0, s4, 2
    dm_store_of 0, 0, x6, s3, 0
    act0_src1_conf 0, 0, 0, s3, 0
    act0_compute x0, 0, 0, 1, 2, 1
    pu_compute 0, 0
    end
