# fence has no effect and fence_i terminates the program like end does (the code after it never runs)
# OUT: [1]
    li   x10, 0x1000
    li   x1, 1
    sw   x10, x1, 0
    fence
    fence_i
    li   x1, 2
    sw   x10, x1, 0
    end  x0
