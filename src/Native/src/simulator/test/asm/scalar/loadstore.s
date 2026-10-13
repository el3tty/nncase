# Loads and stores of all widths (GLB only)
# IN: u8 [128, 255, 1, 2]
# OUT: [-128, 128, -128, 65408, 33685376, 205, 43981]
    li   x10, 0x1000
    lb   x1, x0, 0          # 0x80 sign extended
    sw   x10, x1, 0
    lbu  x1, x0, 0          # 0x80 zero extended
    sw   x10, x1, 4
    lh   x1, x0, 0          # 0xFF80 sign extended
    sw   x10, x1, 8
    lhu  x1, x0, 0          # 0xFF80 zero extended
    sw   x10, x1, 12
    lw   x1, x0, 0          # 0x0201FF80
    sw   x10, x1, 16
    sw   x10, x0, 20        # clear the words before the narrow stores
    sw   x10, x0, 24
    li   x2, 0x1234ABCD
    sb   x10, x2, 20        # only the low byte: 0xCD
    sh   x10, x2, 24        # only the low half word: 0xABCD
    end  x0
