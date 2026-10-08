# nncase K230 simulator - reconstructed C++ (from IDA/Hex-Rays dumps)

Layout (include root is this folder: `g++ -I .`; build with `./build.sh [out]` or CMake):
- `main.cpp`, `globals.h/.cpp` (free C++/C routines), `CMakeLists.txt`, `build.sh`
- `isa/`: `kinstruction.h/.cpp` (base class), `pu_common.h` (shared PU helpers) and one merged `insn_<group>.h/.cpp` per instruction family
  (alu, mem, branch, system, dm, l2, pu, pupdp0, mfu_act1, mfu_pdp1, mfu_misc, act0, ai2d); each class keeps a `// ---- ClassName ----` banner
- `engines/`: one `.h/.cpp` pair per hardware/engine class (ai2d act0 mfu meshnet conv2d pdp0 pdp1 tcu mne l2load l2store dm checkpoint tilehelper memaccessor shared_memory simulator mfu_const)
- `math/numeric_types.h/.cpp`: FP16 / FP24 / BF16 types and reduce_element
- Pre-regroup copy: `work/results_before_regroup`.

Status: every .cpp except main.cpp passes `g++ -std=c++17 -fsyntax-only -fpermissive -w`. Everything compiles and links (see build.sh).

Build notes
- The original used the old libstdc++ string ABI; the object layouts (e.g. KInstruction.flag at +48) assume `-D_GLIBCXX_USE_CXX11_ABI=0`.
- main.cpp does not build yet: it needs `nncase/runtime/k230/shared_memory.h` (use engines/shared_memory.h), its `extern int g_gp_reg[32]` must be `uint32_t`, and it uses the Imm9A/Imm9B/Conf41/Conf43/Conf45/Imm9C instruction types which are not in the dump.
- Verified against the full disassembly (sim.s / the .sc ELF): the earlier UNCERTAIN markers were resolved (register fields 5 bit, 12-bit immediates, signed Blt/Bge, Jalr rs1, DmLoadL1::operation, clamp constants, tables, ...). Decoders that were inlined into main (Add, Sub, Mul, Div, Divu, Rem, Addi, Lui, Auipc, Jal, MmuSetid, 16-bit variants) were reconstructed from the asm; mfu_const.h/.cpp hold the bf16 tables dumped from .rodata.
- main.cpp still uses invented instruction names (RetInstruction, Imm9A/B/C, Conf41/43/45); in the binary these are End, Fence, FenceI, CcrDecl, CcrSet, CcrClr (and Intr) - parsers for them are not added yet.
- Singleton storage that appears as loose `unk_54xxxx` globals only partly aliases the lifted objects (see TODO(layout) comments).
- Remaining `UNCERTAIN` markers (few) are cases where the assembly itself leaves the meaning open; search for `UNCERTAIN` to find places where the decompilation was ambiguous (compare results, lost float math, etc.).
- Sources: work/gen.py (generator), work/gen_baseline (faithful pre-lift output).
