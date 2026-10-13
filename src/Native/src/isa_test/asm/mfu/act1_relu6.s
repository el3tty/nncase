# mfu_act1_compute: dst = line_fit(src1 op src2), element-wise. Registers: stride regs {planes, rows, pitch}, shapes {n,c,h,w}.
# The fit table (act1 mode, 7 fp16 per parameter set) is: knee, slope(x<knee), slope(x>=knee), intercept(x<knee),
# intercept(x>=knee), lower clamp, upper clamp.
# one source (src2 is not configured, so it contributes nothing) and a ReLU6 fit: slope 0 below the knee 0, slope 1 above, clamp [0, 6]
# IN @0x4000: f16 [1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 0.5, 8.0]
# IN @0x6000: f16 [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0]
# OUT @0x7000: f16 [1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 0.5, 6.0]
    li x1, 0x7000                      # destination
    li x2, 0x4000                      # src1
    li x3, 0x5000                      # src2
    li x4, 0x6000                      # fit table
    li x5, 1
    li x6, 2
    li x7, 4
    li x8, 8
    li x9, 1
    ss_pack_shape x5, x9, x6, x7, s1    # n=1 c=1 h=2 w=4
    ss_pack_stride x9, x6, x7, s2       # dense strides
    mfu_act1_conf_stride s2, s2, s2
    mfu_act1_conf_src1 x8, x5, x5, 0, 0      # slice = all elements, no repeats
    mfu_act1_conf_src2 x5, s1, 0, 0          # rleft_repeats, shape, src1, from GLB
    mfu_act1_conf_dest x8, s1
    mfu_act1_conf_quant 0
    mfu_act1_conf 0, 0, 0
    mfu_act1_compute x1, x2, x3, x4
    end
