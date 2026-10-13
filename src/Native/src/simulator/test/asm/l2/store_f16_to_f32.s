# l2_store with mode 0x201: float16 in GLB -> float32 in DDR (exact widening, subnormals included)
# IN @0x4000: f16 [1.0, -2.5, 0.0999755859375, 65504.0, 6.097555160522461e-05, 5.960464477539063e-08]
# DDR_OUT: f32 [1.0, -2.5, 0.0999755859375, 65504.0, 6.097555160522461e-05, 5.960464477539063e-08]
    li x1, 0x200000
    li x2, 0x4000
    li x3, 1
    li x4, 6
    ss_pack_shape x3, x3, x3, x4, s1
    ss_pack_stride x3, x3, x4, s2
    l2_store_conf s2, s2, 1, 2
    l2_store x1, x2, s1
    end
