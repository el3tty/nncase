# Register-register ALU: add sub mul div divu rem remu, and addi/lui/li
# IN: [-7, 3]
# OUT: [-4, -10, -21, -2, 1431655763, -1, 0, -5, 1048576, 305441741]
    lw   x1, x0, 0          # -7
    lw   x2, x0, 4          # 3
    li   x10, 0x1000
    add  x3, x1, x2         # -4
    sw   x10, x3, 0
    sub  x3, x1, x2         # -10
    sw   x10, x3, 4
    mul  x3, x1, x2         # -21
    sw   x10, x3, 8
    div  x3, x1, x2         # -2 (truncates towards zero)
    sw   x10, x3, 12
    divu x3, x1, x2         # 0xFFFFFFF9 / 3 = 1431655763
    sw   x10, x3, 16
    rem  x3, x1, x2         # -1
    sw   x10, x3, 20
    remu x3, x1, x2         # 0xFFFFFFF9 % 3 = 0
    sw   x10, x3, 24
    addi x3, x1, 2          # -5
    sw   x10, x3, 28
    lui  x3, 0x100          # 0x100000
    sw   x10, x3, 32
    li   x3, 0x1234ABCD     # lui + addi with a negative low part
    sw   x10, x3, 36
    end  x0
