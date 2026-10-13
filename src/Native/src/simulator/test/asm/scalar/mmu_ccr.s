# mmu_setid rewrites the bank nibble [31:28] of a register. mmu_conf, the ccr_* instructions and the decode-only engine
# configuration instructions have no observable effect: the program has to run through them and the registers stay intact.
# OUT: [0x52345678, 0x0BCDEF01, 0x12345678]
    li   x10, 0x1000
    li   x1, 0x12345678
    li   x2, 0xBCDEF01
    li   x3, 0x12345678
    mmu_setid x1, 5
    mmu_setid x2, 0
    mmu_conf x0, x3, 2              # rstart, rdepth, mmu id
    ccr_decl x3
    ccr_set 3, 2
    ccr_clr 3
    dm_conf_broadcast 0, 1, 0, 1
    dm_load_w_conf2 0, 0, x3, x3
    pu_fetchif_conf2 0, 0, x3, x0
    pu_pdp0_fetchif_conf2 0, 0, x3, x0
    sw   x10, x1, 0
    sw   x10, x2, 4
    sw   x10, x3, 8
    end  x0
