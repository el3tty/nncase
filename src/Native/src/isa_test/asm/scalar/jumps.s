# auipc, jal and jalr. The absolute pc depends on where the code is loaded (DDR offset in the simulator, a physical
# address on the device), so every value is stored relative to the pc of the first auipc (x1 = code offset 0x04).
# Code offsets: auipc x2 at 0x08, auipc x13 at 0x0c, jal at 0x20 (link 0x24), second auipc at 0x34, jalr at 0x38 (link 0x3c).
# OUT: [0x1004, 8, 0x20, 0x38, 7, 5]
    li   x10, 0x1000
    auipc x1, 0
    auipc x2, 1
    auipc x13, 0
    sub  x11, x2, x1        # (0x08 + 0x1000) - 0x04
    sw   x10, x11, 0
    sub  x11, x13, x1       # 0x0c - 0x04
    sw   x10, x11, 4
    jal  x3, over
    li   x4, 99
    sw   x10, x4, 20        # skipped: would overwrite the marker
over:
    sub  x11, x3, x1        # link 0x24 - 0x04
    sw   x10, x11, 8
    auipc x5, 0
    jalr x6, x5, 12         # target = pc of the auipc + 12: skips the next instruction
    sw   x10, x10, 20       # skipped
    sub  x11, x6, x1        # link 0x3c - 0x04
    sw   x10, x11, 12
    jal  x0, tail           # rd = x0: no link
    sw   x10, x10, 16       # skipped
tail:
    li   x7, 7
    sw   x10, x7, 16
    li   x7, 5
    sw   x10, x7, 20
    end  x0
