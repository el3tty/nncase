# K230 ISA tests

`k230_isa_test [-v] <program.s | directory>` assembles one program, sets up the memory, runs it and
checks the output. With a directory every `*.s` below it is run (sorted, each in its own process, because the simulator
keeps global state and a hanging program must not take the others down) and a summary is printed.

The same tests run on two platforms. Everything platform specific is behind one function,
`int run_test(TestCase &, const RunOptions &)` (`test_runner.h`), implemented once per platform:

| Platform | File | Selected by |
|---|---|---|
| C model | `platform_simulator.cpp` | the normal host build |
| K230 board | `platform_k230.cpp` | the runtime (cross) build, `BUILDING_RUNTIME` |

Platform independent: `main.cpp` (command line, directory runs), `test_case.*` (test description), `test_runner.*`
(memory layout, filling and checking the memory, assembling), `assembler.*`.

**Simulator.** It is not linked into the test app: as in the host runtime it is loaded as a library when
`K230_SIMULATOR_DLL=ON` (the default in `src/Native/src/CMakeLists.txt`) and started as an executable otherwise. Set the
environment variable `K230_SIMULATOR` to the library / executable to use a specific one.
`cmake --build . --target run_isa_tests` runs every test below `asm/`.

**K230 board.** Build with the runtime toolchain and run as root on the board (`k230_isa_test asm/`). The app does not
use the nncase runtime. It opens `/dev/mmz` and `/dev/k230-gnne`, maps the 4 MiB L2 memory (physical `0x80000000`) as
GLB bank 0 and the GNNE registers (`0x80400000`), allocates the DDR tensors and the program with libmmz (`kd_mpi_sys_mmz_alloc`, freed after
every test) and starts the program with `gnne_enable()` (`runtime/k230/gnne.c`, linked in unchanged). The L2 mapping is device memory: a misaligned access to it
faults (SIGSEGV), and the library `memcpy` does exactly that, so the test app reads and writes the GLB only byte by byte.
A test fails when
the NPU does not signal the end within 10 s or reports an exception. Nothing is set up in the NPU before the program
starts.

Exit code: 0 pass, 1 output mismatch, 2 usage / assembly / test description error, 3 simulator / device failure (for
example a missing `end`, an unsupported instruction, a timeout). `-v` prints the machine code.

## Test file

```asm
# IN:  [3, 3]          input tensor(s), one line each; optional element type: i8 u8 i16 u16 i32 (default) u32 f32 f16
# OUT: [9]             expected output tensor(s)
    lw   x5, x0, 0
    lw   x6, x0, 4
    mul  x7, x5, x6
    li   x8, 0x1000
    sw   x8, x7, 0
    end  x0
```

Memory seen by the program. GLB (bank 0, addresses start at 0):

| Address | Content |
|---|---|
| `0x0000` | input tensors, back to back, each aligned to 4 bytes |
| `0x1000` | output tensors, same packing; pre-filled with `0xA5` so that a missing store shows up |

DDR tensors (`DDR_IN`, `DDR_OUT`, below) are located by the test app, because their addresses are DDR offsets in the
simulator but physical addresses on the board. The program gets them through two symbols that `li` accepts, optionally
with an offset: `li x1, DDR_IN`, `li x2, DDR_OUT+16`. Inside a region the tensors are back to back, each aligned to 16
bytes. The code is placed anywhere the platform likes (do not depend on its address: `jumps.s` stores pc differences)
and has to finish with `end`. Matrices are flat row-major lists.

Engine tests need data at other places, so every tensor line can name its own address:

```asm
# IN  @0x4000: u8 [1, 2, 3]        GLB address (bank 0) the input is copied to before the run
# OUT @0x5000: f16 [1.5, -2]       GLB address that is compared after the run (pre-filled with 0xA5)
# DDR_IN:  f32 [1.0, 2.0]          DDR tensors for l2_load / l2_store (addresses: symbols DDR_IN / DDR_OUT)
# DDR_OUT: u8 [1, 2, 3]
```

Without `@` the tensors are laid out automatically: `IN` from GLB `0`, `OUT` from GLB `0x1000`, `DDR_IN` / `DDR_OUT` back to
back in their own region (16-byte aligned; the simulator puts them at DDR offsets `0x100000` / `0x200000`). `@` is only
available for GLB tensors. `f16` values are rounded to nearest even; the
comparison is always bit exact.

## Tests

| Directory | What it checks |
|---|---|
| `asm/scalar` | ALU (incl. edge cases), `mul`, loads / stores, branches, `jal` / `jalr`, `fence_i`, `mmu_*` / `ccr_*` no-ops, 3x3 matrix multiplication |
| `asm/l2` | `l2_load` / `l2_store`: crops, planes and batches, 16 bit elements, float32 <-> float16; `l2_load_w` weight loads (8/16 bit, f32->f16, packed 4/6 bit, compressed) |
| `asm/mfu` | `mfu_memset`, `mfu_memcpy`, `mfu_transpose`, PDP1 pooling (min / max / average / sum, padding, u8 quantisation) and ACT1 (add, mul, line fits incl. 16 segments and shifts, broadcast, per channel) |
| `asm/sim_only` | tests that pass only on the simulator (`pu_forward_psum`; the board does nothing for it). Do not run them on the board |
| `asm/conv` | DM -> PU -> ACT0 -> DM store convolutions (1x1, 3x3, padding, stride, zero points, output types) accumulation over input channel groups, `pu_forward_psum`, and PDP0 pooling / depthwise |
| `asm/ai2d` | AI2D (`extrw` / `extraw` then `ai2d_compute` + `fence`), GLB to GLB: copy with pitch, crop / destination offset, nearest and bilinear resize, constant padding, three planes |

The expected values of the engine tests were computed independently with numpy (convolutions, pooling, fp16 rounding), not
taken from the simulator. Most programs are short enough to read as a description of how the unit is driven.

## Assembler

Syntax and encodings follow `../simulator/README.md`; operands are written in compiler-dump order (`lw rd, rs, offset`,
`sw base, src, offset`, `blt rs1, rs2, label`). Labels, `#` comments, and the pseudo instructions `nop`, `mv`, `li`, `j` are
supported. Mnemonics are looked up with `StringSwitch` (`string_switch.h`, same interface as `llvm::StringSwitch`); the
instruction layouts are the table in `assembler.cpp`.

Hardware notes for the `asm/conv` tests (found on a K230 board): the PU only convolves unsigned data, so signed values are stored as value + 128 with zero point 128 (`*_conf_deq` type 1); `dm_load_w_conf2`, `pu_fetchif_conf2` and `pu_of_conf1` need real `rgroups` / `rgic` / `rgoc` values or the convolution does not run; the CCR instructions must not be used without a producer (`ccr_clr` stalls the program). On the board the 3x3 stride-2 convolution only works when the windows fit the padded input exactly, i.e. `(size - kernel) % stride == 0` in both directions; otherwise the output row pitch is larger than the output shape. Signed int8 output must be stored with `dm_store_of_conf` type 0 (type 1 is a 16-bit mode).

### AI2D on the board

* The engine runs on `ai2d_compute`, not on the write to register 35: `extrw` / `extraw` only queue register writes.
  Write all 36 registers in ascending order (pointers 0..7 with `extraw`, the others with `extrw`), then `ai2d_compute` and `fence`.
  The registers keep their values between operations, so every register the operation does not use has to be written as 0
  (marked `unused, must be 0` in the tests), like the compiler does.
* In the NPU instruction stream the AI2D only works on GLB (`src_ind = dst_ind = 0`); pointers are GLB offsets
  (`offset + (mmu id << 28)`). DDR pointers written with `extraw` are truncated to 22 bits.
* The AI2D timeout register (+0xC0) has to be set to 0x1388 before the run (done by `platform_k230.cpp`).
