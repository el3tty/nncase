# Accumulating convolution whose PSUM is handed to ACT0 with pu_forward_psum: 6 input channels are processed as two groups
# of 3. The first pu_compute writes the int32 PSUM (dest_target 0), the second one runs with load_psum = 1 and also keeps the
# sum in the PSUM (dest_target 0); pu_forward_psum then copies it into ACT0, which activates it and feeds the store.
# This is the order the compiler uses for transposed convolutions. On the board pu_forward_psum only produces output when the
# pu_compute before it accumulated (load_psum 1); after a single compute that overwrites or clears the PSUM it does nothing
# (asm/sim_only/conv1x1_forward_psum.s). Same data and result as conv1x1_accumulate_groups.s.
# Signed int8 values stored as uint8 (value + 128); the zero points (128) are subtracted by the PU.
# IN @0x4000: u8[3,1,8] [
#   [[120, 127, 136, 120, 119, 133, 129, 129]]
#   [[131, 137, 120, 122, 130, 124, 125, 128]]
#   [[135, 136, 127, 132, 120, 132, 131, 137]]
# ]
# IN @0x4400: u8[3,1,8] [
#   [[127, 123, 122, 119, 129, 135, 123, 134]]
#   [[124, 127, 123, 127, 119, 120, 121, 132]]
#   [[131, 132, 126, 137, 134, 135, 133, 125]]
# ]
# IN @0x5000: u8[3,24] [
#   [126, 126, 126, 128*21]
#   [127, 124, 125, 128*21]
#   [128, 132, 124, 128*21]
# ]
# IN @0x5400: u8[3,24] [
#   [127, 126, 125, 128*21]
#   [124, 126, 124, 128*21]
#   [131, 124, 126, 128*21]
# ]
# IN @0x5800: u8 [128, 128, 128]
# IN @0x6000: f16[3,7] [
#   [1.0, 1.0, 0.0, 0.0, -65504.0, 65504.0, 0.0]*3
# ]
# OUT @0x7000: f16[3,1,8] [
#   [[-4.0, -37.0, 24.0, 4.0, 29.0, -22.0, 2.0, -25.0]]
#   [[-25.0, -53.0, 69.0, 22.0, 15.0, -41.0, 16.0, -48.0]]
#   [[-9.0, -15.0, -22.0, -81.0, 67.0, 7.0, -21.0, -28.0]]
# ]
    # Addresses of the tensors in GLB (see the IN / OUT lines above).
    li x1, 0x4000                       # input, channels 0..2
    li x2, 0x4400                       # input, channels 3..5
    li x3, 0x5000                       # weights for channels 0..2
    li x4, 0x5400                       # weights for channels 3..5
    li x5, 0x6000                       # ACT0 parameters (7 fp16 per output channel)
    li x6, 0x7000                       # output (fp16)
    # Shape numbers and constants.
    li x7, 1
    li x8, 3                            # input channels per pass
    li x9, 8                            # pixels per row (W)
    li x11, 3                           # output channels
    li x12, 24                          # bytes per weight line: 1x1 kernel x 24 input channel slots of the PE array
    li x17, 112                         # length operand of dm_load_act0 (not used by the simulator)
    li x13, 31                          # "all channels" (all 24 zero point slots)
    li x14, 128                         # input zero point
    li x15, 0x5800                      # weight zero point table (one byte per output channel)

    # ss_pack_shape rN, rC, rH, rW, sd: packs four dimensions into the shape register sd.
    # s1 = {N 1, C 3, H 1, W 8}: the input of one pass (3 channels x 1 row x 8 pixels).
    ss_pack_shape x7, x8, x7, x9, s1
    # ss_pack_stride r0, r1, r2, sd: packs three numbers into sd. s2 = {3, 1, 8}: the dimensions of the input tile, which
    # dm_load_l1_conf and pu_fetchif_conf1 take as the layout of the input.
    ss_pack_stride x8, x7, x9, s2
    # s3 = {1, 3, 1, 8}: the output, 3 channels x 1 row x 8 pixels.
    ss_pack_shape x7, x11, x7, x9, s3
    # s4 = {3, 1, 8}: layout of the output (used by pu_of_conf1 and dm_store_of_conf).
    ss_pack_stride x11, x7, x9, s4
    # s5 = {0, 0, 3}: its low half is the number of weight lines for dm_load_w, one line per output channel = 3.
    ss_pack_stride x0, x0, x11, s5
    # s6 = all zero: no padding (used by pu_fetchif_conf4).
    ss_pack_shape x0, x0, x0, x0, s6

    # ---- Configuration. The registers keep these settings; nothing is loaded or computed yet. ----

    # dm_load_l1_conf tcu, pu, rstride_s, datatype, l1_type
    # Layout of the input tile that dm_load_l1 copies into the input buffer: dimensions from s2 (3 x 1 x 8),
    # datatype 0 = 8-bit elements, l1_type 0 (not used by the simulator).
    dm_load_l1_conf 0, 0, s2, 0, 0
    # dm_load_w_conf tcu, pu, kernel_h, kernel_w, rstride_oc
    # Weight layout for dm_load_w: 1x1 kernel, x12 = 24 bytes per weight line (one line per output channel).
    dm_load_w_conf 0, 0, 1, 1, x12
    # dm_load_w_conf_deq tcu, pu, quant_type
    # Weights are unsigned 8-bit with zero points (the table given to dm_load_w).
    dm_load_w_conf_deq 0, 0, 1
    # dm_load_w_conf2 tcu, pu, rgroups, rgoc
    # rgroups = 1 (x7), rgoc = 3 (x11) output channels. The simulator only decodes it, the board needs it.
    dm_load_w_conf2 0, 0, x7, x11       # rgroups, rgoc
    # pu_fetchif_conf1 tcu, pu, stride_w, stride_h, rstride_s
    # How the PU reads the input: stride 1 in width and height, tile layout s2.
    pu_fetchif_conf1 0, 0, 1, 1, s2
    # pu_fetchif_conf2 tcu, pu, rgic, rgic_last
    # Input channels per pass: rgic = 3 (x8), rgic_last = 0 (x0). The simulator only decodes it, the board needs it.
    pu_fetchif_conf2 0, 0, x8, x0       # rgic = channels of a group
    # pu_fetchif_conf3 tcu, pu, raddr_s, rgroups, rshape
    # Start address in the input buffer 0 (x0), rgroups = 1 (x7), input shape s1 (1 x 3 x 1 x 8).
    pu_fetchif_conf3 0, 0, x0, x7, s1
    # pu_fetchif_conf4 tcu, pu, rpad_value, sspad
    # Pad value 0 (x0), padding amounts s6 (all zero): no padding.
    pu_fetchif_conf4 0, 0, x0, s6
    # pu_fetchif_conf_deq tcu, pu, ric, rbx, quant_type
    # Input zero point: for all 24 slots (x13 = 31) the zero point is 128 (x14); quant_type 1 = unsigned input with zero point.
    # The PU subtracts it from every input byte.
    pu_fetchif_conf_deq 0, 0, x13, x14, 1
    # pu_w_conf tcu, pu, kernel_h, kernel_w
    # Kernel size seen by the PU: 1x1.
    pu_w_conf 0, 0, 1, 1
    # pu_of_conf1 tcu, pu, rgoc, rgoc_last, rstride_d
    # Output side of the PU: 3 output channels (x11), rgoc_last = 0 (x0), output layout s4.
    pu_of_conf1 0, 0, x11, x0, s4
    # pu_of_conf2 tcu, pu, raddr_d, rshape_d
    # The PU writes its result at address 0 (x0) of the PSUM, output shape s3 (1 x 3 x 1 x 8).
    pu_of_conf2 0, 0, x0, s3

    # ---- Pass 0 (input channels 0..2): overwrite the PSUM and keep the result there ----

    # dm_load_l1 tcu, pu, raddr_s, rhtoc_window, rshape, l1_type
    # Copies the input tile from GLB address x1 (0x4000) into the input buffer now, using the layout set by
    # dm_load_l1_conf. rhtoc_window = x0 (none), tile shape s1, l1_type 0.
    dm_load_l1 0, 0, x1, x0, s1, 0
    # dm_load_w tcu, pu, raddr_s, raddr_bw, r_iochannels, dest_type
    # Queues the weights at GLB x3 (0x5000) together with the zero point table at x15 (0x5800). The number of weight
    # lines (3, one per output channel) is the low half of s5. dest_type 0. The weights are read when pu_compute runs.
    dm_load_w 0, 0, x3, x15, s5, 0
    # pu_compute_conf tcu, pu, load_psum, clr_psum, dest_target, release_if, mode
    # load_psum 0: start from zero, do not add to the PSUM. clr_psum 0. dest_target 0: the result stays in the PSUM
    # (does not go to ACT0). release_if 1: the input buffer can be reused afterwards. mode 0.
    pu_compute_conf 0, 0, 0, 0, 0, 1, 0
    # pu_compute tcu, of_shift_mode
    # Runs the convolution of the queued input and weights. of_shift_mode 0.
    pu_compute 0, 0

    # ---- Pass 1 (input channels 3..5): add to the PSUM, then forward the sum to ACT0 and store ----

    # Same two loads as in pass 0, now with the second half of the input (x2 = 0x4400) and weights (x4 = 0x5400).
    dm_load_l1 0, 0, x2, x0, s1, 0
    dm_load_w 0, 0, x4, x15, s5, 0
    # pu_compute_conf tcu, pu, load_psum, clr_psum, dest_target, release_if, mode
    # load_psum 1: the products are added to the PSUM kept from pass 0. dest_target 0: the sum stays in the PSUM, it is
    # handed to ACT0 by pu_forward_psum below.
    pu_compute_conf 0, 0, 1, 0, 0, 1, 0
    # dm_load_act0 tcu, pu, raddr_s, rlen, dest_channel, is_by_channel
    # Queues the ACT0 parameters from GLB x5 (0x6000). rlen = x17 (ignored by the simulator). dest_channel 0: for the
    # PU (1 would be PDP0). is_by_channel 1: one row of 7 fp16 parameters per output channel.
    dm_load_act0 0, 0, x5, x17, 0, 1
    # dm_store_of_conf tcu, pu, rstride_d, datatype
    # Layout of the output store: s4 (3 x 1 x 8), datatype 2 = fp16.
    dm_store_of_conf 0, 0, s4, 2
    # dm_store_of tcu, pu, raddr_d, rshape, src_channel
    # Queues the store of the output tile to GLB x6 (0x7000), shape s3, source 0 = the PU. It is done when ACT0 runs on
    # the forwarded PSUM.
    dm_store_of 0, 0, x6, s3, 0
    # act0_src1_conf tcu, pu, channel, rshape [, shift]
    # Starts an ACT0 job: channel 0 = on the PU result, tile shape s3, shift 0 (the PSUM is not scaled by 2^-shift).
    act0_src1_conf 0, 0, 0, s3, 0
    # act0_compute raddr_d, tcu, channel, target, dest_datatype, is_by_channel
    # Completes the ACT0 job: destination offset x0, channel 0 (PU), target 1 (send to the DM store), dest_datatype 2
    # (fp16), is_by_channel 1 (per-channel parameters from dm_load_act0).
    act0_compute x0, 0, 0, 1, 2, 1
    # Runs pass 1: convolution + accumulation with the PSUM; the result stays in the PSUM.
    pu_compute 0, 0
    # pu_forward_psum tcu, pu, raddr, rlen
    # Copies rlen = 8 words per output channel row from PSUM offset x0 into ACT0 and runs the ACT0 job above, which sends
    # the activated tile to the queued store.
    li x18, 8
    pu_forward_psum 0, 0, x0, x18
    end
