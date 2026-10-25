# 3x3 matrix multiplication C = A * B with scalar loops (row major, i32)
# IN: i32[3,3] [
#   [1, 2, 3]
#   [4, 5, 6]
#   [7, 8, 9]
# ]
# IN: i32[3,3] [
#   [9, 8, 7]
#   [6, 5, 4]
#   [3, 2, 1]
# ]
# OUT: i32[3,3] [
#   [30, 24, 18]
#   [84, 69, 54]
#   [138, 114, 90]
# ]
    li   x9, 3              # matrix size
    li   x11, 4             # element size
    li   x10, 0x1000        # C pointer
    li   x1, 0              # i
loop_i:
    li   x2, 0              # j
loop_j:
    li   x3, 0              # k
    li   x4, 0              # acc
loop_k:
    mul  x5, x1, x9
    add  x5, x5, x3         # i*3 + k
    mul  x5, x5, x11
    lw   x6, x5, 0          # A[i][k]  (A at GLB 0)
    mul  x7, x3, x9
    add  x7, x7, x2         # k*3 + j
    mul  x7, x7, x11
    lw   x8, x7, 36         # B[k][j]  (B at GLB 36)
    mul  x6, x6, x8
    add  x4, x4, x6
    addi x3, x3, 1
    blt  x3, x9, loop_k
    sw   x10, x4, 0         # C[i][j]
    addi x10, x10, 4
    addi x2, x2, 1
    blt  x2, x9, loop_j
    addi x1, x1, 1
    blt  x1, x9, loop_i
    end  x0
