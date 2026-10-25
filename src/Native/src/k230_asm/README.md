# k230_asm

Assembler for the K230 NPU instruction set (encodings: `../simulator/README.md`).

* `k230_asm` (static library, `assembler.h`): `assemble(source, symbols)` returns the machine code, the byte offset
  and the source line of every instruction. Used by the ISA tests (`../isa_test`).
* `k230-as` (tool): `k230-as [-o out.bin] [-l listing.txt] [-D NAME=VALUE]... <input.s | ->` writes the raw
  little-endian instruction stream (the text of a K230 function, as it is in a kmodel) and / or a listing (byte offset,
  machine code, source line). Exit code 0 ok, 1 assembly error (`file: line N: message`), 2 usage / file error.

## Syntax

```asm
loop:                          # labels may share a line with an instruction
    lw    x5, x0, 4            # operands in compiler-dump order: lw rd, rs, offset / sw base, src, offset
    addi  x5, x5, -1
    blt   x0, x5, loop         # branch / jal targets: label or byte offset
    li    x8, 0x1000           # pseudo: lui + addi as needed; also nop, mv, j
    li    x9, DDR_IN+16        # symbols (-D or the caller of assemble()) are accepted by li only
    dm_store_of_conf 0, 0, ss2, fp16       # enum fields: name or number
    ss_pack_shape x13, x14, x13, x24, ss3  # shape registers: s0..s7 or ss0..ss7
    end   x0
```

Numbers are decimal, `0x` hex or `0b` binary with an optional `-`. A signed immediate may also be written as its raw
field value (0..2^width-1). Fields the compiler dumps by name (`psum`, `fp16`, `pu_mode_deconv`, `NCHW`, ...) accept the
names of the compiler enums (`modules/Nncase.Modules.K230/Nncase.TIR.Instructions`). The instruction layouts are the
table in `assembler.cpp`; mnemonics are looked up with `StringSwitch` (`string_switch.h`, the interface of
`llvm::StringSwitch`).

## Compiler dumps

A compiler dump (`compile.text.asm`, written by `K230DeSerializerVisitor`) prints every instruction as a block:

```
pc offset: 96
inst addi
    opcode : arithm_imm
    rd : x19
    rs : x0
    imm : 3
```

Written one instruction per line, without `pc offset`, `opcode` and the `binding_*` lines, with the remaining values in
order (`addi x19, x0, 3`), it assembles to the same bytes as the kmodel. This was checked for every K230 function of the
`tests/importer/onnx_/basic` suite.
