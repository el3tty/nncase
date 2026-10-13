# Multiplies two numbers: out[0] = in[0] * in[1]
# IN: [3, 3]
# OUT: [9]
    lw   x5, x0, 0          # a (input area: GLB 0x0000)
    lw   x6, x0, 4          # b
    mul  x7, x5, x6
    li   x8, 0x1000         # output area
    sw   x8, x7, 0          # sw base, source, offset
    end  x0
