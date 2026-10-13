# fence and fence_i have no effect: the code after them still runs (a K230 board does not stop at fence_i)
# OUT: [2]
    li   x10, 0x1000
    li   x1, 1
    sw   x10, x1, 0
    fence
    fence_i
    li   x1, 2
    sw   x10, x1, 0
    end  x0
