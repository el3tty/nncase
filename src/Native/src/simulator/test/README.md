# Simulator ISA tests

`k230_isa_test <program.s>` assembles one program, sets up the memory, runs the simulator and checks the output. The
simulator is not linked into the test app: as in the host runtime it is loaded as a library when `K230_SIMULATOR_DLL=ON`
(the default in `src/Native/src/CMakeLists.txt`) and started as an executable otherwise. Set the environment variable
`K230_SIMULATOR` to the library / executable to use a specific one.
`cmake --build . --target run_isa_tests` runs every `asm/*.s`.

Exit code: 0 pass, 1 output mismatch, 2 usage / assembly / test description error, 3 simulator failure (for example a
missing `end` or an unsupported instruction). `-v` prints the machine code.

## Test file

```asm
# IN:  [3, 3]          input tensor(s), one line each; optional element type: i8 u8 i16 u16 i32 (default) u32 f32
# OUT: [9]             expected output tensor(s)
    lw   x5, x0, 0
    lw   x6, x0, 4
    mul  x7, x5, x6
    li   x8, 0x1000
    sw   x8, x7, 0
    end  x0
```

Memory seen by the program (the "ctrl" shared memory is GLB bank 0, GLB addresses start at 0):

| Address | Content |
|---|---|
| `0x0000` | input tensors, back to back, each aligned to 4 bytes |
| `0x1000` | output tensors, same packing; pre-filled with `0xA5` so that a missing store shows up |

The code is placed at DDR offset `0x1000` and has to finish with `end`. Matrices are flat row-major lists.

## Assembler

Syntax and encodings follow `../isa/README.md`; operands are written in compiler-dump order (`lw rd, rs, offset`,
`sw base, src, offset`, `blt rs1, rs2, label`). Labels, `#` comments, and the pseudo instructions `nop`, `mv`, `li`, `j` are
supported. Mnemonics are looked up with `StringSwitch` (`string_switch.h`, same interface as `llvm::StringSwitch`); the
instruction layouts are the table in `assembler.cpp`.
