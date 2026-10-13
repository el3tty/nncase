# All six conditional branches, with operands (-1, 1), (5, 5) and (1, -1): the signed and unsigned compares disagree
# for the first and the last pair. Every result is 1 when the branch was taken, 0 otherwise.
# IN: [-1, 1, 5]
# OUT: [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    li   x10, 0x1000
    li   x9, 0
    lw   x1, x0, 0
    lw   x2, x0, 4
    lw   x3, x0, 8
    li   x5, 0
    beq  x1, x2, taken1
    j    done1
taken1:
    li   x5, 1
done1:
    sw   x10, x5, 0
    li   x5, 0
    bne  x1, x2, taken2
    j    done2
taken2:
    li   x5, 1
done2:
    sw   x10, x5, 4
    li   x5, 0
    blt  x1, x2, taken3
    j    done3
taken3:
    li   x5, 1
done3:
    sw   x10, x5, 8
    li   x5, 0
    bltu x1, x2, taken4
    j    done4
taken4:
    li   x5, 1
done4:
    sw   x10, x5, 12
    li   x5, 0
    bge  x1, x2, taken5
    j    done5
taken5:
    li   x5, 1
done5:
    sw   x10, x5, 16
    li   x5, 0
    bgeu x1, x2, taken6
    j    done6
taken6:
    li   x5, 1
done6:
    sw   x10, x5, 20
    li   x5, 0
    beq  x3, x3, taken7
    j    done7
taken7:
    li   x5, 1
done7:
    sw   x10, x5, 24
    li   x5, 0
    bne  x3, x3, taken8
    j    done8
taken8:
    li   x5, 1
done8:
    sw   x10, x5, 28
    li   x5, 0
    blt  x3, x3, taken9
    j    done9
taken9:
    li   x5, 1
done9:
    sw   x10, x5, 32
    li   x5, 0
    bltu x3, x3, taken10
    j    done10
taken10:
    li   x5, 1
done10:
    sw   x10, x5, 36
    li   x5, 0
    bge  x3, x3, taken11
    j    done11
taken11:
    li   x5, 1
done11:
    sw   x10, x5, 40
    li   x5, 0
    bgeu x3, x3, taken12
    j    done12
taken12:
    li   x5, 1
done12:
    sw   x10, x5, 44
    li   x5, 0
    beq  x2, x1, taken13
    j    done13
taken13:
    li   x5, 1
done13:
    sw   x10, x5, 48
    li   x5, 0
    bne  x2, x1, taken14
    j    done14
taken14:
    li   x5, 1
done14:
    sw   x10, x5, 52
    li   x5, 0
    blt  x2, x1, taken15
    j    done15
taken15:
    li   x5, 1
done15:
    sw   x10, x5, 56
    li   x5, 0
    bltu x2, x1, taken16
    j    done16
taken16:
    li   x5, 1
done16:
    sw   x10, x5, 60
    li   x5, 0
    bge  x2, x1, taken17
    j    done17
taken17:
    li   x5, 1
done17:
    sw   x10, x5, 64
    li   x5, 0
    bgeu x2, x1, taken18
    j    done18
taken18:
    li   x5, 1
done18:
    sw   x10, x5, 68
    end  x0
