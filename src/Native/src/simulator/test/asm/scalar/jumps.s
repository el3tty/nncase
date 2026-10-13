# auipc, jal and jalr. pc values are DDR offsets: the test app places the code at 0x1000, so the instruction at code offset N has pc 0x1000 + N.
# The expected pc values were read off the machine code listing (-v): auipc at code offset 0x04, second auipc at 0x0c,
# first jal at 0x14 (link 0x1018), jalr at 0x28 (link 0x102c).
# OUT: [0x1004, 0x200c, 0x1018, 0x102c, 7, 5]
    li   x10, 0x1000
    auipc x1, 0
    sw   x10, x1, 0
    auipc x2, 1
    sw   x10, x2, 4
    jal  x3, over
    li   x4, 99
    sw   x10, x4, 20        # skipped: would overwrite the marker
over:
    sw   x10, x3, 8
    auipc x5, 0
    jalr x6, x5, 12         # target = pc of the auipc + 12: skips the next instruction
    sw   x10, x10, 20       # skipped
    sw   x10, x6, 12
    jal  x0, tail           # rd = x0: no link
    sw   x10, x10, 16       # skipped
tail:
    li   x7, 7
    sw   x10, x7, 16
    li   x7, 5
    sw   x10, x7, 20
    end  x0
