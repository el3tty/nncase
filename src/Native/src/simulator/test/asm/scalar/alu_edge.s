# ALU edge cases: wrap-around of add / sub / mul, rounding of signed division and the sign of the remainder (they follow the
# dividend), unsigned operations on 0xFFFFFFFF, writes to x0 are discarded, addi sign extension, lui with the top bit set.
# Division by zero or INT_MIN / -1 would crash the host, so they are not tested.
# IN: [-2147483648, -7, -3, -2, -1, 1, 2, 7, 10, 1431655766, 2147483647]
# OUT: [-2147483648, 2147483647, 1, 0, -2, -3, -3, -1, 1, 2147483647, 5, -5, 0, -4096]
    li   x10, 0x1000
    lw   x1, x0, 40
    lw   x2, x0, 20
    add  x3, x1, x2          # 2147483647 add 1
    sw   x10, x3, 0
    lw   x1, x0, 0
    lw   x2, x0, 20
    sub  x3, x1, x2          # -2147483648 sub 1
    sw   x10, x3, 4
    lw   x1, x0, 40
    lw   x2, x0, 40
    mul  x3, x1, x2          # 2147483647 mul 2147483647
    sw   x10, x3, 8
    lw   x1, x0, 0
    lw   x2, x0, 24
    mul  x3, x1, x2          # -2147483648 mul 2
    sw   x10, x3, 12
    lw   x1, x0, 8
    lw   x2, x0, 36
    mul  x3, x1, x2          # -3 mul 1431655766
    sw   x10, x3, 16
    lw   x1, x0, 4
    lw   x2, x0, 24
    div  x3, x1, x2          # -7 div 2
    sw   x10, x3, 20
    lw   x1, x0, 28
    lw   x2, x0, 12
    div  x3, x1, x2          # 7 div -2
    sw   x10, x3, 24
    lw   x1, x0, 4
    lw   x2, x0, 24
    rem  x3, x1, x2          # -7 rem 2
    sw   x10, x3, 28
    lw   x1, x0, 28
    lw   x2, x0, 12
    rem  x3, x1, x2          # 7 rem -2
    sw   x10, x3, 32
    lw   x1, x0, 16
    lw   x2, x0, 24
    divu x3, x1, x2          # -1 divu 2
    sw   x10, x3, 36
    lw   x1, x0, 16
    lw   x2, x0, 32
    remu x3, x1, x2          # -1 remu 10
    sw   x10, x3, 40
    addi x4, x0, -5          # sign extended 12 bit immediate
    sw   x10, x4, 44
    addi x0, x4, 100         # x0 is hard wired to zero
    sw   x10, x0, 48
    lui  x5, 0xFFFFF
    sw   x10, x5, 52
    end  x0
