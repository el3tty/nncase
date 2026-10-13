# mode word 0x201 (l2_datatype = 1, ddr_datatype = 2): float32 in DDR -> float16 in GLB
# the expected values are rounded to nearest even; 1e-8 underflows to 0 and 100000 overflows to +inf
# DDR_IN: f32 [1.5, -2.0, 0.10000000149011612, 65504.0, 9.99999993922529e-09, 100000.0, 3.1415927410125732, 5.999999848427251e-05, -0.0, 0.33333298563957214]
# OUT @0x4000: f16 [1.5, -2.0, 0.0999755859375, 65504.0, 0.0, inf, 3.140625, 6.002187728881836e-05, -0.0, 0.333251953125]
    li x1, 0x100000
    li x2, 0x4000
    li x3, 1
    li x4, 10
    ss_pack_shape x3, x3, x3, x4, s1
    ss_pack_stride x3, x3, x4, s2
    l2_load_conf s2, s2, 1, 2
    l2_load x2, x1, s1
    end
