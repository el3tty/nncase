# 1x1 convolution on uint8 data with input and weight zero points
# Unsigned input and weights: pu_fetchif_conf_deq (type 1) subtracts the input zero point (128) from every channel (ric = 31 means
# all channels) and dm_load_w_conf_deq (type 1) subtracts a per output channel weight zero point (a byte table at raddr_bw).
# IN @0x4000: u8 [10, 20, 30, 40, 250, 0, 5, 128, 128, 127, 129, 100, 100, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 255, 254, 253, 0, 1, 2, 3, 4]
# IN @0x5000: u8 [130, 128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 129, 129, 129, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 128, 140, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# IN @0x5800: u8 [128, 127, 130]
# IN @0x6000: f16 [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0]
# OUT @0x7000: f16 [-236.0, -216.0, -196.0, -176.0, 244.0, -256.0, -246.0, 0.0, -236.0, -218.0, -194.0, -736.0, -312.0, -1002.0, -986.0, -734.0, -3900.0, -3958.0, -4022.0, 3536.0, 1416.0, 4090.0, 4018.0, 2766.0]
    li x1, 0x4000
    li x2, 0x5000
    li x3, 0x6000
    li x4, 0x7000
    li x5, 1
    li x6, 4
    li x7, 1
    li x8, 8
    li x9, 3
    li x10, 1
    li x11, 8
    li x12, 24
    li x17, 112
    li x18, 0x5800                      # weight zero points
    li x19, 31                          # "all channels"
    li x20, 128
    ss_pack_shape x5, x6, x7, x8, s1
    ss_pack_stride x6, x7, x8, s2
    ss_pack_shape x5, x9, x10, x11, s3
    ss_pack_stride x9, x10, x11, s4
    ss_pack_stride x0, x0, x9, s5
    ss_pack_shape x0, x0, x0, x0, s6
    dm_load_l1_conf 0, 0, s2, 0, 0
    dm_load_l1 0, 0, x1, x0, s1, 0
    dm_load_w_conf 0, 0, 1, 1, x12
    dm_load_w_conf_deq 0, 0, 1
    dm_load_w_conf2 0, 0, x5, x9
    dm_load_w 0, 0, x2, x18, s5, 0
    pu_fetchif_conf1 0, 0, 1, 1, s2
    pu_fetchif_conf2 0, 0, x6, x0
    pu_fetchif_conf3 0, 0, x0, x5, s1
    pu_fetchif_conf4 0, 0, x0, s6
    pu_fetchif_conf_deq 0, 0, x19, x20, 1
    pu_w_conf 0, 0, 1, 1
    pu_of_conf1 0, 0, x9, x0, s4
    pu_of_conf2 0, 0, x0, s3
    pu_compute_conf 0, 0, 0, 0, 1, 1, 0
    dm_load_act0 0, 0, x3, x17, 0, 1
    dm_store_of_conf 0, 0, s4, 2
    dm_store_of 0, 0, x4, s3, 0
    act0_src1_conf 0, 0, 0, s3, 0
    act0_compute x0, 0, 0, 1, 2, 1
    pu_compute 0, 0
    end
