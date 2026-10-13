# PDP0 max pooling, 3x3 window, stride 1, padding 1; pad value -128 so the border never wins
# PDP0 reads its input as int8 bytes from PSUM_L1, so the program first runs an identity 1x1 convolution whose ACT0 stage writes the
# image to PSUM_L1 (target 0) and then starts PDP0 (max pooling, mode 2): window 3x3, stride 1x1, padding t/b/l/r 1/1/1/1.
# The PDP0 result goes through ACT0 (engine 1) and dm_store_of (src_channel 1) to GLB. PDP0 runs when the store is queued.
# The hardware only handles unsigned data: the image is stored as value + 128 (zero point 128) and the identity convolution of the first
# stage (zero points 0) writes it to PSUM_L1 as uint8. PDP0 subtracts the zero point 128 again.
# IN @0x4000: u8 [129, 133, 130, 128, 125, 132, 134, 127, 135, 120, 130, 130, 128, 131, 124, 137, 127, 126, 125, 124, 133, 134, 135, 136, 119, 138, 117, 140, 141, 114, 143, 112]
# IN @0x5000: u8 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# IN @0x5C00: u8 [0, 0]
# IN @0x6000: f16 [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0, 1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0]
# OUT @0x7000: f16 [5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 6.0, 6.0, 7.0, 7.0, 9.0, 9.0, 7.0, 7.0, 9.0, 9.0, 6.0, 7.0, 8.0, 8.0, 10.0, 10.0, 12.0, 12.0, 13.0, 15.0, 15.0, 15.0, 13.0, 15.0, 15.0, 15.0]
    li x1, 0x4000
    li x2, 0x5000
    li x3, 0x6000
    li x4, 0x7000
    li x5, 1
    li x6, 2
    li x7, 4
    li x8, 4
    li x12, 24
    li x17, 112
    li x20, 2048                        # PSUM_L1 byte offset of the intermediate image
    li x23, 31                          # "all channels"
    li x24, 128                         # input zero point
    li x25, 1                           # weight groups
    li x26, 0x5C00                      # weight zero points
    ss_pack_shape x5, x6, x7, x8, s1
    ss_pack_stride x6, x7, x8, s2
    ss_pack_stride x0, x0, x6, s5       # one weight line per channel
    dm_load_l1_conf 0, 0, s2, 0, 0
    dm_load_l1 0, 0, x1, x0, s1, 0
    dm_load_w_conf 0, 0, 1, 1, x12
    dm_load_w_conf_deq 0, 0, 1
    dm_load_w_conf2 0, 0, x25, x6       # rgroups, rgoc
    dm_load_w 0, 0, x2, x26, s5, 0
    pu_fetchif_conf1 0, 0, 1, 1, s2
    pu_fetchif_conf2 0, 0, x6, x0       # rgic
    pu_fetchif_conf3 0, 0, x0, x5, s1
    ss_pack_shape x0, x0, x0, x0, s6
    pu_fetchif_conf4 0, 0, x0, s6
    pu_fetchif_conf_deq 0, 0, x23, x0, 1
    pu_w_conf 0, 0, 1, 1
    pu_of_conf1 0, 0, x6, x0, s2
    pu_of_conf2 0, 0, x0, s1
    pu_compute_conf 0, 0, 0, 0, 1, 1, 0
    dm_load_act0 0, 0, x3, x17, 0, 1
    act0_src1_conf 0, 0, 0, s1, 0
    act0_compute x20, 0, 0, 0, 0, 1     # target 0: uint8 into PSUM_L1 only, no store needed
    pu_compute 0, 0
    # --- PDP0 ---
    li x9, 4
    li x10, 4
    li x11, -128
    li x13, 1
    li x14, 1
    li x15, 1
    li x16, 1
    li x18, 6
    li x19, 6
    ss_pack_shape x5, x6, x18, x19, s3  # padded input shape
    ss_pack_shape x5, x6, x9, x10, s4   # output shape
    ss_pack_stride x6, x9, x10, s7      # output strides
    ss_pack_shape x13, x14, x15, x16, s6
    pu_pdp0_mode_conf 0, 0, 2
    pu_pdp0_fetchif_conf1 0, 0, 1, 1
    pu_pdp0_fetchif_conf2 0, 0, x6, x0   # rgic
    pu_pdp0_fetchif_conf3 0, 0, s3
    pu_pdp0_fetchif_conf4 0, 0, x11, s6
    pu_pdp0_conf_deq 0, 0, x24, 1       # unsigned input, zero point 128
    pu_pdp0_w_conf 0, 0, 3, 3
    pu_pdp0_of_conf 0, 0, s7, s4
    dm_load_act0 0, 0, x3, x17, 1, 1
    dm_store_of_conf 0, 0, s7, 2
    act0_src1_conf 0, 0, 1, s4, 0
    act0_compute x0, 0, 1, 1, 2, 1
    pu_pdp0_compute 0, x20              # queue the compute (it needs the store)
    dm_store_of 0, 0, x4, s4, 1         # queues the store and starts PDP0
    end
