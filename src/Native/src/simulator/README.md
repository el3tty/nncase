# K230 NPU instruction set (as implemented by the C-model simulator)

This document describes the instruction set of the Kendryte K230 KPU (NPU) as it is executed by the
reconstructed `nncase.simulator.k230` C-model in this folder. It is **reverse-engineered**: every statement
comes from the decoders and `operation()` bodies in `insn_*.cpp`, the dispatch in `../main.cpp` and the engine
classes in `../engines/`. There is no vendor specification behind it. Where the simulator decodes a field but never
looks at it, or where a behaviour was inferred from names only, the text says so (see [Open questions](#open-questions)).

Terminology: *asm* is the textual syntax printed by the nncase compiler dumps (`ids.txt`, TIR/`*.il` dumps). The
operand names in the tables are the compiler's operand names; the C++ classes use the same names with a trailing `_`.

Contents

1. [Machine model](#1-machine-model)
2. [Instruction word and dispatch](#2-instruction-word-and-dispatch)
3. [Opcode map](#3-opcode-map)
4. [Scalar core](#4-scalar-core)
5. [Control: end, fence, interrupt, CCR](#5-control-end-fence-interrupt-ccr)
6. [MMU, shape and stride registers](#6-mmu-shape-and-stride-registers)
7. [AI2D register writes](#7-ai2d-register-writes)
8. [L2 transfers (DDR <-> GLB)](#8-l2-transfers-ddr---glb)
9. [DM: data movement into/out of the PU](#9-dm-data-movement-intoout-of-the-pu)
10. [PU: convolution engine](#10-pu-convolution-engine)
11. [PU-PDP0: depthwise / pooling engine](#11-pu-pdp0-depthwise--pooling-engine)
12. [ACT0: activation after the PU](#12-act0-activation-after-the-pu)
13. [MFU: memcpy, memset, transpose, PDP1, ACT1](#13-mfu-memcpy-memset-transpose-pdp1-act1)
14. [Execution model of the asynchronous engines](#14-execution-model-of-the-asynchronous-engines)
15. [Typical instruction sequences](#15-typical-instruction-sequences)
16. [Source map](#16-source-map)
17. [Open questions](#open-questions)

---

## 1. Machine model

| State | Size | Notes |
|---|---|---|
| General purpose registers `x0..x31` | 32 x 32 bit | `_G.gp_reg`. `x0` is hard-wired to zero (`set_g_gp_reg` ignores writes). Register operands are 5 bits. |
| Shape / stride registers `s0..s7` | 8 x 64 bit | `_G.shape_reg`. Operands are 3 bits. Written only by `ss_pack_shape` / `ss_pack_stride`. |
| PC | 32 bit | Byte offset of the instruction **inside the DDR image**, relative to `_G.DDR`. `Simulator::start_pc_` is the entry offset; the trace uses `pc_rel = pc - start_pc`. |
| DDR | 2 GiB | Shared memory object mapped at `_G.DDR`. Holds code, constants (weights) and the model input/output. |
| GLB (L2, "global buffer") | 16 banks x 4 MiB address windows | `_G.GLB[0..15]`. Bank 0 is the shared-memory mapping, banks 1..15 are zero-filled host memory. |
| MMU table | 16 x {start, depth} | `_G.MMU_MMUItem[2*bank]`, `[2*bank+1]`, in 32-byte lines. Written by `mmu_conf`. |
| PSUM L1 | 128 KiB | `_G.PSUM_L1`, partial sums of the PU. |
| IF L1 | 24 KiB | `_G.IF_L1`, input feature buffer loaded by `dm_load_l1`. |
| ACT0 PSUM buffers | 2 x 128 KiB | Inside the `Act0` singleton; buffer 0 belongs to the PU (Conv2D), buffer 1 to PDP0. |
| AI2D register file | 36 words | Reached through `extrw` / `extraw`. |

The PU/PDP0/MFU/L2 units are singletons in `../engines/`; instructions mostly *write configuration registers* of these
units and a few *trigger* instructions run them.

### 1.1 Addresses

* **GLB address** (used by L2 load/store, DM, MFU and by scalar `lw/sw/...`):

  ```
  31      28 27                              0
  +---------+--------------------------------+
  |  bank   |        byte offset in bank      |
  +---------+--------------------------------+
  ```

  The bank selects `_G.GLB[bank]`. The 28-bit offset is a byte offset into that bank. `mmu_setid` rewrites the bank
  nibble of a register. The "MMU address" printed in traces is `offset + 32 * MMU_MMUItem[2*bank]` (physical GLB line
  position); it is **not** used to index memory in the simulator, it only appears in the trace dumps.
* **DDR address**: plain 32-bit byte offset from `_G.DDR` (`l2_load` source, `l2_store` destination, branch targets).

### 1.2 Shape and stride registers

`ss_pack_shape` builds a 4-dimensional shape (all fields are 16 bit):

```
 63      48 47      32 31      16 15       0
 +---------+----------+----------+----------+
 |    N    |    C     |    H     |    W     |      shape_reg[rss]
 +---------+----------+----------+----------+
```

`ss_pack_stride` builds a 3-dimensional stride (element or byte units depending on the consumer):

```
 63      48 47      32 31      16 15       0
 +---------+----------+----------+----------+
 |    0    | N stride | C stride | H stride |      shape_reg[rss]
 +---------+----------+----------+----------+
```

The packing is plain shift/or with **no masking**: values wider than 16 bits spill into the neighbouring field. When a
configuration instruction consumes a stride register it calls the three fields `dim0 = [47:32]`, `dim1 = [31:16]`,
`dim2 = [15:0]`; consumers of a shape register call `dim0 = [63:48]` ... `dim3 = [15:0]` or, in the PU/PDP0 classes,
`shape_word(i)` with `i = 0` the lowest halfword (W).

Padding is also passed in a shape register (`sspad`): halfword 3 = top, 2 = bottom, 1 = left, 0 = right
(`PuFetchifConf4`, `PuPdp0FetchifConf4`).

---

## 2. Instruction word and dispatch

Instructions are 32-bit (most) or 16-bit ("short") little-endian words. The simulator fetches `*(uint32_t*)pc`, takes
`opcode = raw & 0x7F` and dispatches in `../main.cpp`. After each instruction `pc += 4` (or `+= 2`).

```
 31                                                  7 6        0
 +-----------------------------------------------------+---------+
 |                    operands / selectors               | opcode  |
 +-----------------------------------------------------+---------+
```

* Operands are packed in *assembly order* starting at bit 7: a register operand takes 5 bits, a shape register 3 bits,
  `tcu_id` 3 bits, `pu_id` 3 bits, flags 1 bit, other immediates as listed per instruction.
* Several opcodes are *families* selected by an extra field that is not an assembly operand:

| Family opcode | Selector | Bits |
|---|---|---|
| `alu` (0x0C) | `alu_op` | `[21:17]` |
| `addi`, `load`, `store`, `branch`, `jalr` | `funct3` | `[19:17]` |
| `dm_conf` (0x50), `pu_conf` (0x5A), `pu_pdp0_conf` (0x5E) | `conf_id` | `[16:13]` |
| `mfu_conf` (0x62) | `mfu_conf_id` | `[11:7]` |

* 32-bit NPU instructions that start with `tcu_id`/`pu_id` have them at `[9:7]` and `[12:10]`; the K230 has a single
  TCU and a single PU, so the compiler always emits 0 for both and the simulator never uses them to select hardware.
* Unused/reserved bits are decoded into `reserved_` members and ignored. They are not checked, so malformed words are
  silently accepted.
* An unknown opcode or selector prints `unsupported instruction in Cmodel, Skip!`, the pc and opcode, and throws
  `std::runtime_error("Invaild Opcode")` (the typo is the original's).

The trace of each decoded instruction records `bit_offset` (running count of executed bits) in `Simulator::trace_log_`.

---

## 3. Opcode map

`16` = 16-bit encoding (pc += 2). Opcodes not listed raise "unsupported instruction".

| Opcode | Mnemonic / family | Size | Section |
|---|---|---|---|
| 0x01 | `intr` | 16 | 5 |
| 0x02 | `lui` | 32 | 4 |
| 0x03 | `end` | 16 | 5 |
| 0x04 | `auipc` | 32 | 4 |
| 0x05 | `fence` | 16 | 5 |
| 0x06 | `lw/lh/lhu/lb/lbu` (funct3) | 32 | 4 |
| 0x07 | `fence_i` | 16 | 5 |
| 0x08 | `sw/sh/sb` (funct3) | 32 | 4 |
| 0x0C | `add/sub/mul/div/divu/rem/remu` (funct5) | 32 | 4 |
| 0x0E | `addi` (funct3 must be 0) | 32 | 4 |
| 0x10 | `beq/bne/blt/bltu/bge/bgeu` (funct3) | 32 | 4 |
| 0x12 | `jal` | 32 | 4 |
| 0x14 | `jalr` | 32 | 4 |
| 0x16 | `extrw` | 32 | 7 |
| 0x18 | `extraw` | 32 | 7 |
| 0x40 | `ss_pack_shape` | 32 | 6 |
| 0x41 | `ccr_decl` | 16 | 5 |
| 0x42 | `ss_pack_stride` | 32 | 6 |
| 0x43 | `ccr_set` | 16 | 5 |
| 0x44 | `mmu_conf` | 32 | 6 |
| 0x45 | `ccr_clr` | 16 | 5 |
| 0x46 | `l2_load_conf` | 32 | 8 |
| 0x48 | `l2_load_w_conf` | 32 | 8 |
| 0x49 | `mmu_setid` | 16 | 6 |
| 0x4A | `l2_store_conf` | 32 | 8 |
| 0x4B | `dm_conf_broadcast` | 16 | 9 |
| 0x4C | `l2_load` | 32 | 8 |
| 0x4D | `pu_compute` | 16 | 10 |
| 0x4E | `l2_store` | 32 | 8 |
| 0x4F | `pu_pdp0_compute` | 16 | 11 |
| 0x50 | `dm_conf` family: `dm_load_l1_conf` (0), `dm_load_w_conf` (1), `dm_load_w_conf_deq` (2), `dm_store_of_conf` (4), `dm_load_w_conf2` (5) | 32 | 9 |
| 0x51 | `ai2d_compute` | 16 | 7 |
| 0x52 | `dm_load_l1` | 32 | 9 |
| 0x54 | `dm_load_w` | 32 | 9 |
| 0x56 | `dm_load_act0` | 32 | 9 |
| 0x57 | `l2_load_w` (odd opcode, but 32-bit) | 32 | 8 |
| 0x58 | `dm_store_of` | 32 | 9 |
| 0x5A | `pu_conf` family, ids 0..8 | 32 | 10 |
| 0x5C | `pu_forward_psum` | 32 | 10 |
| 0x5E | `pu_pdp0_conf` family, ids 0..7 | 32 | 11 |
| 0x60 | `act0_src1_conf` | 32 | 12 |
| 0x62 | `mfu_conf` family, ids 0..14 (5 unused) | 32 | 13 |
| 0x64 | `mfu_memcpy` | 32 | 13 |
| 0x66 | `mfu_memset` | 32 | 13 |
| 0x68 | `mfu_transpose` | 32 | 13 |
| 0x6A | `mfu_pdp1_compute` | 32 | 13 |
| 0x72 | `mfu_act1_compute` | 32 | 13 |
| 0x74 | `act0_compute` | 32 | 12 |

---

## 4. Scalar core

A small 32-bit RISC-V-like integer core. It is used to compute addresses, loop counters and parameters; there are no
floating point or atomic instructions. The scalar instructions are executed **at decode time** (`parser_operation`),
the branches in `operation()` right after.

Note the register field positions differ from RISC-V: `rd` is `[11:7]`, `rs1` is `[16:12]`, and the function code
sits *between* `rs1` and `rs2`/`imm`.

### 4.1 Register-register ALU (opcode 0x0C)

```
 31    27 26   22 21    17 16   12 11    7 6      0
 +-------+-------+--------+-------+-------+--------+
 | rsvd  |  rs2  | funct5 |  rs1  |  rd   | 0x0C   |
 +-------+-------+--------+-------+-------+--------+
```

| funct5 | asm | Operation |
|---|---|---|
| 0 | `add rd, rs1, rs2` | `rd = rs1 + rs2` |
| 1 | `sub rd, rs1, rs2` | `rd = rs1 - rs2` |
| 2 | `mul rd, rs1, rs2` | low 32 bits of the product |
| 3 | `div rd, rs1, rs2` | signed division |
| 4 | `divu rd, rs1, rs2` | unsigned division |
| 5 | `rem rd, rs1, rs2` | signed remainder |
| 6 | `remu rd, rs1, rs2` | unsigned remainder |

Division and remainder are plain host `idiv`/`div`: division by zero (and `INT_MIN / -1`) **faults the host process**
in the original simulator; there is no RISC-V style defined result. Other `funct5` values are unsupported.

### 4.2 Immediates

| asm | Layout | Operation |
|---|---|---|
| `addi rd, rs, imm` | opcode 0x0E; `rd[11:7] rs[16:12] funct3[19:17]=0 imm[31:20]` | `rd = rs + sext12(imm)`. `funct3 != 0` is unsupported. |
| `lui rd, imm` | opcode 0x02; `rd[11:7] imm[31:12]` | `rd = imm << 12` |
| `auipc rd, imm` | opcode 0x04; `rd[11:7] imm[31:12]` | `rd = pc + (imm << 12)`, where `pc` is the DDR-relative address of the instruction |

The compiler materialises 32-bit constants as `lui` + `addi`.

### 4.3 Loads and stores (GLB only)

```
load : rd[11:7]  rs[16:12]  funct3[19:17]  imm[31:20]      (opcode 0x06)
store: base[11:7] src[16:12] funct3[19:17] imm[31:20]      (opcode 0x08)
```

| Opcode | funct3 | Instruction | Access |
|---|---|---|---|
| 0x06 | 0 | `lw rd, rs, offset` | 32-bit |
| 0x06 | 1 | `lh` | 16-bit, sign-extended |
| 0x06 | 2 | `lhu` | 16-bit, zero-extended |
| 0x06 | 3 | `lb` | 8-bit, sign-extended |
| 0x06 | 4 | `lbu` | 8-bit, zero-extended |
| 0x08 | 0 | `sw` | 32-bit |
| 0x08 | 1 | `sh` | 16-bit |
| 0x08 | 2 | `sb` | 8-bit |

The effective address is `reg + sext12(imm)`, interpreted as a GLB address (`bank = addr >> 28`,
`offset = addr & 0x0FFFFFFF`). Loads/stores **cannot reach DDR**; use `l2_load`/`l2_store` for that. The store form
puts the *base* register in the `rd` position `[11:7]` and the *data* register in `[16:12]`. There is no bounds check.

In compiler dumps only `lw` and `addi/lui` appear (the dumps print `lw rd, rs, offset`).

### 4.4 Branches and jumps

```
branch: rs1[11:7] rs2[16:12] funct3[19:17] imm[31:20]     (opcode 0x10)
jal   : rd[11:7]  imm[31:12]                               (opcode 0x12)
jalr  : rd[11:7]  rs[16:12]  funct3[19:17]=0 imm[31:20]    (opcode 0x14)
```

| funct3 | Instruction | Condition |
|---|---|---|
| 0 | `beq` | `rs1 == rs2` |
| 1 | `bne` | `rs1 != rs2` |
| 2 | `blt` | signed `<` |
| 3 | `bltu` | unsigned `<` |
| 4 | `bge` | signed `>=` |
| 5 | `bgeu` | unsigned `>=` |

* Branch target = `pc + sext12(imm)` and `jal` target = `pc + sext20(imm)`. The offset is in **bytes** and is
  *not* shifted left by one as in RISC-V.
* `jal` writes `rd = pc + 4` and always jumps; `jalr` writes `rd = pc + 4` and jumps to `rs + sext12(imm)` (the low bit
  is **not** cleared). `pc` values are DDR-relative.
* The comparison operands are sampled when the instruction is decoded. A taken branch replaces the program counter
  with `_G.DDR + next_pc`.

---

## 5. Control: end, fence, interrupt, CCR

All are 16-bit.

| Opcode | asm | Layout | Effect in the simulator |
|---|---|---|---|
| 0x03 | `end rs` | `rs[11:7]`, `[15:12]` rsvd | Terminates the program (the compiler always uses `x0`). Runs the end-of-program queue check (see section 14) and returns 0. |
| 0x05 | `fence` | `[15:7]` rsvd | No effect (decoded only). |
| 0x07 | `fence_i` | `[15:7]` rsvd | No effect (decoded only). The original simulator ends the program here, but a K230 board runs past it (checked with `isa_test`), so it is a no-op in this reimplementation. Not present in compiler dumps. |
| 0x01 | `intr` | `rs[11:7]`, `[15:12]` rsvd | Throws `std::runtime_error("INTR!")`. |
| 0x41 | `ccr_decl rnum` | `rnum[11:7]` | Decoded only; samples `gp_reg[rnum]`. |
| 0x43 | `ccr_set ccr, value` | `ccr[11:7] value[15:12]` | No effect. Compiler ranges: `ccr` 0..13, `value` 1..2. |
| 0x45 | `ccr_clr ccr` | `ccr[11:7]` | No effect. |

The original has empty `operation()` bodies for the CCR instructions. The C-model executes a single in-order stream and
models each unit synchronously, so they are no-ops here. Their hardware purpose (presumably synchronisation between
units) is not known.

---

## 6. MMU, shape and stride registers

| Opcode | asm | Layout | Effect |
|---|---|---|---|
| 0x44 | `mmu_conf rstart, rdepth, mmu_id` | `rstart[11:7] rdepth[16:12] mmu_id[20:17]` | Stores `gp_reg[rstart]` and `gp_reg[rdepth]` as the segment start/depth of bank `mmu_id` in `_G.MMU_MMUItem`, `_G.glb_start/depth` and the AI2D object. Compiler range of `mmu_id`: 0..5. |
| 0x49 (16-bit) | `mmu_setid rd, id` | `rd[11:7] id[15:12]` | `rd = (rd & 0x0FFFFFFF) | (id << 28)` — rewrites the bank nibble of an address held in a register. |
| 0x40 | `ss_pack_shape rn, rc, rh, rw, rss` | `rn[11:7] rc[16:12] rh[21:17] rw[26:22] rss[29:27]` | `shape_reg[rss] = rn<<48 | rc<<32 | rh<<16 | rw` |
| 0x42 | `ss_pack_stride rn, rc, rh, rss` | `rn[11:7] rc[16:12] rh[21:17] rss[29:27]` | `shape_reg[rss] = rn<<32 | rc<<16 | rh` |

The MMU values only influence the *trace* addresses (`+ 32 * start`); the data itself is addressed through
`bank/offset`. Their purpose on the hardware is to place logical GLB banks in physical L2 memory.

---

## 7. AI2D register writes

The AI2D engine (image resize / crop / pad / affine / format conversion) is programmed by writing its register file
through two 32-bit instructions. In the simulator the processing is done synchronously (`AI2D::ai2d_proc`).

```
extrw / extraw: reg_addr[16:7] (10 bit)  rs[21:17]  imm[31:22] (decoded, unused)
```

* `extrw reg_addr, rs` writes `gp_reg[rs]` into the AI2D register at **byte** address `reg_addr` (`reg_addr >> 2` is the
  register index, 0..35 are defined).
* `extraw` is the restricted variant that handles only byte addresses 0..31, i.e. the eight address registers
  (`src_ch_ptr[0..3]`, `dst_ch_ptr[0..3]`).
* Writing the register at byte address **140** (index 35, `calc_enable`/`dst_x`/`dst_y`) starts the engine.
* `ai2d_compute` (0x51, 16-bit) has no operation body.

AI2D register map (word index = byte address / 4):

| Index | Contents |
|---|---|
| 0..3 | source channel pointers 0..3 |
| 4..7 | destination channel pointers 0..3 |
| 8, 9 | source width/layout (two halfwords each) |
| 10, 11 | destination width/layout |
| 12, 13, 14, 15, 32, 33 | affine matrix (raw words) |
| 18 | formats and mode: `dst_format[31:28] src_format[27:24] bound_ind[23:20] shift[19:12] pad_mod[11:10] interpolation[9:8] cord_round[7:6] dst_channel[5:3] channel_cfg[2:0]` |
| 19 | `bound_smooth[16]`, `bound_val[15:0]` |
| 20..24, 26 | YUV->RGB coefficients (12-bit fields), word 24 also holds `const_pad_ch[0]` |
| 25 | `is_signed[27] cmd_id[26] dst_ind[25] src_ind[24]` and `const_pad_ch[3:1]` |
| 28, 29 | padding bottom/top, right/left (10-bit fields) |
| 30 | source height/width (13-bit fields) |
| 31 | `csc_en[31] intr_mask[30]`, destination height/width |
| 34 | `src_x`, `src_y` |
| 35 | `calc_enable[31]`, `dst_y`, `dst_x`: **start** |

Indices 16, 17, 27 have no register behind them.

---

## 8. L2 transfers (DDR <-> GLB)

The L2 units copy tensors between DDR and GLB. A transfer is configured with a `*_conf` instruction (strides and data
types) and started with the transfer instruction (addresses and shape). Both execute synchronously.

### 8.1 `l2_load_conf` — configure DDR->GLB tile copy (0x46)

`l2_load_conf rstride_d, rstride_s, l2_datatype, ddr_datatype`

```
rstride_d[9:7]  rstride_s[12:10]  l2_datatype[14:13]  ddr_datatype[17:15]
```

* `rstride_d` — stride register for the GLB (destination) tensor, `rstride_s` — for the DDR (source) tensor. Each
  gives `{dim0 = [47:32], dim1 = [31:16], pitch = [15:0]}` (elements).
* `ddr_datatype:l2_datatype` is the 16-bit mode word `Mode() = l2 | ddr << 8`:
  `0x201` converts **float32 (DDR) -> float16 (GLB)**; otherwise a non-zero `l2_datatype` copies 16-bit elements and
  zero copies 8-bit elements.

### 8.2 `l2_load` — DDR->GLB tile copy (0x4C)

`l2_load raddr_d, raddr_s, rshape` — `raddr_d[11:7] raddr_s[16:12] rshape[19:17]`

* `gp_reg[raddr_d]` is the GLB destination (bank | offset), `gp_reg[raddr_s]` is the DDR **offset**.
* `shape_reg[rshape]` = `{batch[63:48], plane[47:32], row_count[31:16], row_len[15:0]}`.
* Copies `batch x plane x row_count` rows of `row_len` elements, applying the strides of the previous `l2_load_conf`.

### 8.3 `l2_load_w_conf` / `l2_load_w` — weight loads (0x48 / 0x57)

`l2_load_w_conf rlen_compressed, rlen_decompressed, l2_datatype, ddr_datatype, enable_decompress`

```
rlen_compressed[11:7]  rlen_decompressed[16:12]  l2_datatype[18:17]  ddr_datatype[21:19]  enable_decompress[22]
```

`wconf = (gp_reg[rlen_decompressed] << 32) | gp_reg[rlen_compressed]`; the high half is the number of weight elements.
`enable_decompress` selects the compressed weight format (16-byte blocks with a presence bitmap). The weight element
size code in `ddr_datatype` is: 0 = 1 byte, 1 = 2 bytes, 2 = 4 bytes, 3 = packed 4-bit, 4 = packed 6-bit, 5 = 2 bytes;
`l2_datatype:ddr_datatype = 1:2` converts float32 to float16.

`l2_load_w raddr_d, raddr_s, rvalid_c_num` — `raddr_d[11:7] raddr_s[16:12] rvalid_c_num[21:17]`: copy (and optionally
decompress) the weights from DDR offset `gp_reg[raddr_s]` to the GLB address `gp_reg[raddr_d]`;
`gp_reg[rvalid_c_num]` is the number of valid output channels per destination row group.
The scatter works on the bytes of the GLB data, whatever the element size: groups of `rvalid_c_num + 1` bytes, every group
starts a new 24-byte row (with 16-bit weights an odd group splits an element between two rows). Verified on a board.

### 8.4 `l2_store_conf` / `l2_store` — GLB->DDR (0x4A / 0x4E)

Mirror image of the load: `l2_store_conf rstride_d, rstride_s, l2_datatype, ddr_datatype` has the same layout as
`l2_load_conf` but `rstride_d` is the DDR stride and `rstride_s` the GLB stride;
`l2_store raddr_d, raddr_s, rshape` writes `gp_reg[raddr_d]` (DDR offset) from `gp_reg[raddr_s]` (GLB address).

---

## 9. DM: data movement into/out of the PU

The DM ("data mover") moves tiles between GLB and the engines' private buffers and queues the descriptors that the
engines consume. `dm_conf` instructions fill the DM singleton, the movers snapshot it into a descriptor.
`dest_type` / `dest_channel` / `src_channel` choose between the PU (convolution, `0`) and PDP0 (`1`).

All DM instructions start with `tcu_id[9:7] pu_id[12:10]`.

### 9.1 Configuration (opcode 0x50, selector `conf_id[16:13]`)

| conf_id | asm | Fields after `conf_id` | Effect |
|---|---|---|---|
| 0 | `dm_load_l1_conf tcu_id, pu_id, rstride_s, datatype, l1_type` | `rstride_s[19:17] datatype[21:20] l1_type[23:22]` | L1 load tile strides (3 dims) and element mode. `l1_type` is decoded but unused. |
| 1 | `dm_load_w_conf tcu_id, pu_id, kernel_h, kernel_w, rstride_oc` | `kernel_h[21:17] kernel_w[26:22] rstride_oc[31:27]` | Kernel size and the output-channel stride register (`loadw_len`). |
| 2 | `dm_load_w_conf_deq tcu_id, pu_id, quant_type` | `quant_type[18:17]` | Weight dequantisation mode. |
| 4 | `dm_store_of_conf tcu_id, pu_id, rstride_d, datatype` | `rstride_d[19:17] datatype[21:20]` | Output-feature store strides and element mode. |
| 5 | `dm_load_w_conf2 tcu_id, pu_id, rgroups, rgoc` | `rgroups[21:17] rgoc[26:22]` | Decoded only (the original's `operation` is empty): group counts. |
| 0x4B (16-bit) | `dm_conf_broadcast` | `tcu_id[9:7] broadcast_if[10] broadcast_w[11] psum_cascade[12]` | Decoded only; no consumer. Meaning of the names is a guess. |

`conf_id` 3 and 6..15 are unsupported.

### 9.2 Movers

| Opcode | asm | Fields | Effect |
|---|---|---|---|
| 0x52 | `dm_load_l1 tcu_id, pu_id, raddr_s, rhtoc_window, rshape, l1_type` | `raddr_s[17:13] rhtoc_window[22:18] rshape[25:23] l1_type[27:26]` | Loads the feature tile at GLB address `gp_reg[raddr_s]` into `IF_L1` **immediately**, and pushes a descriptor onto `Conv2D::if_queue_`. `gp_reg[rhtoc_window]` is the H-to-C window / layout selector. |
| 0x54 | `dm_load_w tcu_id, pu_id, raddr_s, raddr_bw, r_iochannels, dest_type` | `raddr_s[17:13] raddr_bw[22:18] r_iochannels[25:23] dest_type[27:26]` | Queues a weight descriptor (`raddr_s` = weights, `raddr_bw` = bias/zero-points, low half of `shape_reg[r_iochannels]` = line count) on `Conv2D::weight_queue_` or `PDP0::weight_queue_`. |
| 0x56 | `dm_load_act0 tcu_id, pu_id, raddr_s, rlen, dest_channel, is_by_channel` | `raddr_s[17:13] rlen[22:18] dest_channel[23] is_by_channel[24]` | Queues the ACT0 parameter table (line-fit segment parameters, fp16) from GLB on `Conv2D::act0_param_queue_` or `PDP0::load_act0_queue_`. `rlen` is read but not used. |
| 0x58 | `dm_store_of tcu_id, pu_id, raddr_d, rshape, src_channel` | `raddr_d[17:13] rshape[20:18] src_channel[21]` | Queues the output-feature store descriptor (GLB destination, 4-D shape) on `Conv2D::store_queue_` or `PDP0::store_of_queue_`, then lets a **waiting** compute run (see section 14). |

---

## 10. PU: convolution engine

The PU computes a convolution tile into `PSUM_L1` (or straight into ACT0). Everything except `pu_compute` and
`pu_forward_psum` only writes the `PuCompute` configuration snapshot inside `Conv2D` (`cfg_`).

### 10.1 `pu_conf` family (opcode 0x5A, `tcu_id[9:7] pu_id[12:10] conf_id[16:13]`)

| conf_id | asm | Fields | Effect |
|---|---|---|---|
| 0 | `pu_fetchif_conf1 tcu_id, pu_id, stride_w, stride_h, rstride_s` | `stride_w[21:17] stride_h[26:22]` (immediates), `rstride_s[29:27]` | Convolution strides and the 3-D IF stride register. |
| 1 | `pu_fetchif_conf2 tcu_id, pu_id, rgic, rgic_last` | `rgic[21:17] rgic_last[26:22]` | Input channel groups / channels of the last group. Decoded only (empty operation). |
| 2 | `pu_fetchif_conf3 tcu_id, pu_id, raddr_s, rgroups, rshape` | `raddr_s[21:17] rgroups[26:22] rshape[29:27]` | IF base address, number of groups and the input shape `{N, C, H, W}`. |
| 3 | `pu_fetchif_conf4 tcu_id, pu_id, rpad_value, sspad` | `rpad_value[21:17]`, `sspad[29:27]` (bits 26:22 are unused) | Pad value (low byte only) and padding amounts (top/bottom/left/right from the halfwords of `sspad`). |
| 4 | `pu_fetchif_conf_deq tcu_id, pu_id, ric, rbx, quant_type` | `ric[21:17] rbx[26:22] quant_type[28:27]` | IF dequantisation: `if_zero_points[gp_reg[ric]] = gp_reg[rbx]` (low byte), or all 24 slots when `gp_reg[ric] == 31`. `quant_type == 1` means unsigned input with zero points. |
| 5 | `pu_w_conf tcu_id, pu_id, kernel_h, kernel_w` | `kernel_h[21:17] kernel_w[26:22]` (immediates) | Kernel size. |
| 6 | `pu_of_conf1 tcu_id, pu_id, rgoc, rgoc_last, rstride_d` | `rgoc[21:17] rgoc_last[26:22] rstride_d[29:27]` | Output stride (3-D). `rgoc*` are decoded, not used. |
| 7 | `pu_of_conf2 tcu_id, pu_id, raddr_d, rshape_d` | `raddr_d[21:17]`, `rshape_d[29:27]` | PSUM base byte offset and output shape. |
| 8 | `pu_compute_conf tcu_id, pu_id, load_psum, clr_psum, dest_target, release_if, mode` | `load_psum[17] clr_psum[18] dest_target[19] release_if[20] mode[21]` | Compute flags: accumulate into the existing PSUM; clear PSUM first; `dest_target` 1 = forward the result to ACT0, 0 = keep in PSUM; release the IF buffer; PU mode. |

`conf_id` 9..15 are unsupported.

### 10.2 Triggers

| Opcode | asm | Fields | Effect |
|---|---|---|---|
| 0x4D (16-bit) | `pu_compute tcu_id, of_shift_mode` | `tcu_id[9:7] of_shift_mode[11:10]` | Stores the output shift mode (0 none, 1 = `<< 4`, otherwise `>> 4`), appends a snapshot of the configuration to `compute_queue_` and runs the compute under the conditions in section 14. |
| 0x5C | `pu_forward_psum tcu_id, pu_id, raddr, rlen` | `raddr[17:13] rlen[22:18]` | Copies `rlen` words per channel row from `PSUM_L1` (byte offset `gp_reg[raddr]`) into the ACT0 PSUM buffer, then runs ACT0 on it (`Conv2D::Activate`). Emitted by the compiler after the last pass of a transposed convolution. On a K230 board it only does this after an accumulating `pu_compute` (`load_psum` 1); after a compute with `load_psum` 0 (with or without `clr_psum`) the PU stays busy on it and the store queued for ACT0 never runs, although the program still reaches `end`. The simulator does not model this and always forwards (`isa_test/asm/conv/conv1x1_accumulate_forward_psum.s` uses the order that works on the board). |


---

## 11. PU-PDP0: depthwise / pooling engine

PDP0 handles depthwise convolution and pooling. Registers are word indices 16384+ in the original, kept as named fields
in `Pdp0Compute`. Layout of the fields is the same as for the PU.

| Opcode / conf_id | asm | Fields | Effect |
|---|---|---|---|
| 0x5E / 0 | `pu_pdp0_mode_conf` | `mode[19:17]` | Mode: 0 = depthwise convolution, 1 and 2 = pooling kinds. |
| 0x5E / 1 | `pu_pdp0_fetchif_conf1` | `stride_w[21:17] stride_h[26:22]` | Strides. |
| 0x5E / 2 | `pu_pdp0_fetchif_conf2` | `rgic[21:17] rgic_last[26:22]` | No effect. |
| 0x5E / 3 | `pu_pdp0_fetchif_conf3` | `rshape[29:27]` | Input shape `{N, C, H, W}` (before padding is removed). |
| 0x5E / 4 | `pu_pdp0_fetchif_conf4` | `rpad_value[21:17] sspad[24:22]` | Pad value and padding amounts. |
| 0x5E / 5 | `pu_pdp0_conf_deq` | `rbx[21:17] quant_type[23:22]` | Input zero point; `quant_type` is the PDP0 `unsigned_flag`. |
| 0x5E / 6 | `pu_pdp0_w_conf` | `kernel_h[21:17] kernel_w[26:22]` | Window size. |
| 0x5E / 7 | `pu_pdp0_of_conf` | `rstride_d[19:17] rshape_d[22:20]` | Output stride (3-D) and output shape (4-D). |
| 0x4F (16-bit) | `pu_pdp0_compute` | `tcu_id[9:7] raddr_s[14:10]` | Stores `gp_reg[raddr_s]` as the PSUM offset, queues a PDP0 compute descriptor and runs it when a `dm_store_of` is already queued. |

`conf_id` 8..15 are unsupported. None of the PDP0 instructions appear in the nncase compiler dumps analysed so far;
their asm syntax is therefore inferred from the C++ names.

---

## 12. ACT0: activation after the PU

ACT0 applies a piecewise-linear ("line fit") activation with per-channel parameters to a PSUM tile, scales it
by `2^-shift`, converts it and routes the result to the PSUM and/or the DM store.

| Opcode | asm | Fields | Effect |
|---|---|---|---|
| 0x60 | `act0_src1_conf tcu_id, pu_id, channel, rshape` | `tcu_id[9:7] pu_id[12:10] variant[15:13] channel[16] rshape[19:17] shift[24:20]` | Defines the activation tile `{N, C, H, W}`, the PSUM shift (`shift`, not an assembly operand in the dumps) and the engine (`channel` 0 = PU, 1 = PDP0), and creates the *current* `Act0Compute` descriptor of that engine. `variant` is decoded but unused. |
| 0x74 | `act0_compute raddr_d, tcu_id, channel, target, dest_datatype, is_by_channel` | `raddr_d[16:12] tcu_id[19:17] channel[20] target[22:21] dest_datatype[24:23] is_by_channel[25]` | Completes the current descriptor (output byte offset in PSUM L1 `gp_reg[raddr_d]`, route, datatype, per-channel flag) and pushes it onto `act0_queue_` of the PU or PDP0. |

* `target` (output route): 0 = write to `PSUM_L1` only, 1 = send to the DM store only, other values = both.
* `dest_datatype`: 0 = uint8 (saturated 0..255), 1 = int8 (saturated -127..127), 2 = fp16 bits, 3 = int16
  (saturated -32767..32767).
* `is_by_channel` = 1: one parameter row per channel (taken from `dm_load_act0`); 0: one shared row.

---

## 13. MFU: memcpy, memset, transpose, PDP1, ACT1

The MFU is a second, GLB-to-GLB engine that is independent from the PU. All of its instructions execute synchronously.
Its configuration instructions share opcode 0x62 with the selector `mfu_conf_id[11:7]`.

### 13.1 `mfu_conf` family (opcode 0x62)

| id | asm | Fields after `mfu_conf_id[11:7]` | Effect |
|---|---|---|---|
| 0 | `mfu_transpose_conf rstride_d, rstride_s, l2_datatype, permute` | `rstride_d[14:12] rstride_s[17:15] l2_datatype[19:18] permute[24:20]` | Source/destination stride registers, element size (`1` = 16-bit) and the dimension permutation (0..23). |
| 1 | `mfu_pdp1_conf1 stride_w, stride_h, rstride_s, funct2, rstride_d` | `stride_w[16:12] stride_h[21:17] rstride_s[24:22] pool_mode[26:25] rstride_d[29:27]` | PDP1 strides (immediates), source/destination stride registers and pooling function: 0 min, 1 max, 2 average (sum / count), 3 sum scaled by `avg_scale`. |
| 2 | `mfu_pdp1_conf2 rcount_w, rcount_h, rpe_h, rpe_last_h` | `rcount_w[16:12] rcount_h[21:17] rpe_h[26:22] rpe_last_h[31:27]` | Output window counts (the `rpe_*` registers are read but not used). |
| 3 | `mfu_pdp1_conf3 rpe_channels, rpe_last_channels, rpad_value, sspad` | `rpe_channels[16:12] rpe_last_channels[21:17] rpad_value[26:22] sspad[29:27]` | Channels per pass, fp16 pad value and padding amounts. |
| 4 | `mfu_pdp1_conf4 rwindow_w, rwindow_h, rscale, enable_h2c, enable_bw` | `rwindow_w[16:12] rwindow_h[21:17] rscale[26:22] enable_h2c[27] enable_bw[28]` | Pool window, average scale (bf16/fp16 bits) and the H-to-C and BW options. |
| 6 | `mfu_pdp1_conf_deq rscale, rbias, quant_type` | `rscale[16:12] rbias[21:17] cfg[28:22]` | PDP1 input dequantisation: scale, zero point, source data type (`cfg[1:0]` of the field) and flag. |
| 7 | `mfu_pdp1_conf_quant rscale, rbias, quant_type` | same layout | PDP1 output quantisation: scale, zero point, destination type and flag. |
| 8 | `mfu_act1_conf_stride rstride_s1, rstride_s2, rstride_d1` | `rstride_s1[14:12] rstride_s2[17:15] rstride_d1[20:18]` | Packs of `{c, h, w}` strides of ACT1 source 1, source 2 and destination (`rh | rc << 16 | rn << 32`). |
| 9 | `mfu_act1_conf_src1 rslice, rright_repeats, rslice_repeats, sid, slice_loc` | `rslice[16:12] rright_repeats[21:17] rslice_repeats[26:22] sid[27] slice_loc[28]` | Broadcast/repeat configuration of source `sid` (0 = src1, 1 = src2). |
| 10 | `mfu_act1_conf_src2 rleft_repeats, rshape, sid, source_type` | `rleft_repeats[16:12] rshape[19:17] sid[20] source_type[21]` | Third repeat factor, shape and whether the source is in L2 or PSUM. |
| 11 | `mfu_act1_conf_dest rlen, rshape` | `rlen[16:12] rshape[19:17]` | Destination length and shape. |
| 12 | `mfu_act1_conf_deq rscale, rbias, quant_type, sid` | `rscale[16:12] rbias[21:17] quant_type[23:22] sid[24] shift[29:25]` | Dequantisation of source `sid`. |
| 13 | `mfu_act1_conf_quant quant_type` | `cfg[18:12]` | Destination type (`[13:12]`) and fit shift (`[18:14]`). |
| 14 | `mfu_act1_conf funct4, is_by_channel, is_16_segments` | `alu_op[15:12] is_by_channel[16] is_16_segments[17]` | ACT1 ALU operation, per-channel parameters and 16-segment line fit. |

`mfu_conf_id` 5 and 15..31 are unsupported (5 may be a missing PDP1 configuration).

### 13.2 Copy, fill, transpose

| Opcode | asm | Fields | Effect |
|---|---|---|---|
| 0x64 | `mfu_memcpy` | `raddr_d[11:7] raddr_s[16:12] rstride_d[19:17] rstride_s[22:20] rshape[25:23]` | GLB->GLB strided copy. Not in the compiler dumps. |
| 0x66 | `mfu_memset` | `raddr_d[11:7] rv[16:12] rstride[19:17] rshape[22:20] l2_datatype[24:23]` | Fills a strided GLB tile with the low halfword of `gp_reg[rv]`; `l2_datatype` 1 = 16-bit elements. Not in the compiler dumps. |
| 0x68 | `mfu_transpose raddr_d, raddr_s, rshape` | `raddr_d[11:7] raddr_s[16:12] rshape[19:17]` | Permutes the 4-D tile as configured by `mfu_transpose_conf`. |

### 13.3 PDP1: pooling / reduction (opcode 0x6A)

`mfu_pdp1_compute raddr_d, raddr_s, rshape` — `raddr_d[11:7] raddr_s[16:12] rshape[19:17]`. Takes the shape
`{dim1_count[63:48], dim2_count[47:32], shape_h[31:16], shape_w[15:0]}` and runs `PDP1::PdpRedCompute` with the
configuration set by `mfu_pdp1_conf1..4/deq/quant`. Data are fp16 inside; the dequant/quant stages convert from/to
8- or 16-bit integers.

### 13.4 ACT1: element-wise binary/unary operations (opcode 0x72)

`mfu_act1_compute raddr_d1, raddr_s1, raddr_s2, raddr_arg` — `raddr_d1[11:7] raddr_s1[16:12] raddr_s2[21:17]
raddr_arg[26:22]`. Reads source 1 and source 2 (broadcast according to the repeat factors), applies the ACT1 ALU
operation and the 16-segment line fit whose parameters are at `gp_reg[raddr_arg]`, and writes the result to
`gp_reg[raddr_d1]`. The three tensor layouts are set by `mfu_act1_conf_stride`: **the stride registers must be
configured before every compute that uses a different layout** (an earlier simulator version did not latch these
registers, which made every output row equal to the first).

---

## 14. Execution model of the asynchronous engines

The scalar instructions and the MFU run immediately. The PU and PDP0 are decoupled: configuration and data
movement instructions *enqueue descriptors*; the compute runs when all inputs have been queued.

`Conv2D` queues (PDP0 has the analogous `weight`, `store_of`, `load_act0`, `act0`, `compute` queues):

| Queue | Filled by | Consumed by |
|---|---|---|
| `if_queue_` | `dm_load_l1` | convolution (IF tile) |
| `weight_queue_` | `dm_load_w` (`dest_type = 0`) | convolution |
| `act0_param_queue_` | `dm_load_act0` | ACT0 (parameter table) |
| `act0_queue_` | `act0_compute` | ACT0 |
| `store_queue_` | `dm_store_of` | DM store of the ACT0 result |
| `compute_queue_` | `pu_compute` | `Conv2D::Compute` |

Compute trigger rules (derived from `PuComputeInstruction` / `DmStoreOfInstruction`):

* `pu_compute` always queues a snapshot of the configuration.
  * With `pu_compute_conf.dest_target != 1` (result stays in PSUM) it runs **immediately**.
  * With `dest_target == 1` (result forwarded to ACT0) it runs now only if a `dm_store_of` is already queued, or the
    oldest ACT0 compute does not route to the DM store (`target == 0`). Otherwise the compute waits.
* `dm_store_of` (PU) enqueues the store and then runs the compute **only if one is waiting** in `compute_queue_`. If no
  compute is waiting the store remains queued for a later `pu_compute`/`act0_compute`.
* `pu_pdp0_compute` runs when `store_of_queue_` is non-empty; `dm_store_of` with `src_channel = 1` runs the PDP0 whenever
  the queue is not empty.
* `act0_compute` does not trigger anything by itself.
* `pu_forward_psum` copies the PSUM into ACT0 and runs ACT0, and with it the queued DM store, right away, whatever the
  previous `pu_compute` was. A board only does so after an accumulating `pu_compute` (see the `pu_forward_psum` row in
  the PU table).

### 14.1 End-of-program invariant

When the stream reaches `end` **all engine queues must be empty**. A leftover descriptor means a load or
compute was configured but never consumed, i.e. the instruction stream is malformed (compiler bug). `main.cpp` throws
`std::runtime_error("Engine queues are not empty at the end of the program (compiler bug?): conv.store_queue=1, ...")`
and lists every non-empty queue.

---

## 15. Typical instruction sequences

These sequences are an **inferred, schematic order** of how the instruction groups work together (from the engine queue
logic and the operand names), not a transcript of the compiler output. Exact ordering of the configuration instructions
varies; check `dumps-*/gencode/` for real streams. Registers are loaded with `lui`/`addi` just before use.

### 15.1 Convolution tile

```
mmu_conf / mmu_setid                      ; map the GLB banks used by this tile
l2_load_conf ...   ; l2_load ...          ; input feature map DDR -> GLB
l2_load_w_conf ... ; l2_load_w ...        ; weights DDR -> GLB (optionally decompressed)

ss_pack_shape / ss_pack_stride            ; build shapes and strides (s0..s7)
dm_load_l1_conf ...; dm_load_l1 ...       ; IF tile -> IF_L1, descriptor -> if_queue
dm_load_w_conf  ...; dm_load_w  ...       ; weights/bias -> weight_queue
pu_fetchif_conf1..4, pu_fetchif_conf_deq  ; strides, input shape, padding, zero points
pu_w_conf, pu_of_conf1, pu_of_conf2       ; kernel size, output strides and shape
pu_compute_conf ...                       ; psum flags, forward to ACT0?
act0_src1_conf ...; dm_load_act0 ...      ; activation tile and line-fit parameters
act0_compute ...                          ; route and datatype of the result
dm_store_of_conf ...; dm_store_of ...     ; output store descriptor (-> GLB)
pu_compute ...                            ; (the compute runs here or at dm_store_of)

l2_store_conf ...; l2_store ...           ; result GLB -> DDR
```

Large convolutions are split into input-channel groups that accumulate in PSUM (`load_psum`, see
`isa_test/asm/conv/conv1x1_accumulate_groups.s`); the scalar branches are available to loop over tiles.

Transposed convolutions (`pu_compute_conf` mode `pu_mode_deconv`) never send a pass straight to ACT0: the compiler runs
one `pu_compute` per kernel position (and input-channel group), the first with `clr_psum` 1 and the others with
`load_psum` 1, all with `dest_target` 0, and hands the PSUM to ACT0 and the store with `pu_forward_psum` after the last
one. On a board the forward needs that last pass to accumulate, so a deconvolution with a single pass (1x1 kernel)
writes no output there; the compiler keeps 1x1 transposed convolutions on the CPU for that reason.

### 15.2 Element-wise / resize / pooling tiles

```
l2_load ...
mfu_act1_conf_stride / _src1 / _src2 / _dest / _deq / _quant / mfu_act1_conf
mfu_act1_compute ...
l2_store ...
```

```
mfu_pdp1_conf1..4 [, _deq, _quant]
mfu_pdp1_compute ...
```

### 15.3 Program skeleton

```
<mmu_conf, setup>
<tile loop with loads, compute, stores>
end x0
```

---

## 16. Source map

| Topic | Header | Implementation |
|---|---|---|
| Base class, decode helper | `kinstruction.h` | `kinstruction.cpp` |
| Scalar ALU, addi/lui/auipc | `insn_alu.h` | `insn_alu.cpp` |
| Loads and stores | `insn_mem.h` | `insn_mem.cpp` |
| Branches, jal, jalr | `insn_branch.h` | `insn_branch.cpp` |
| end/fence/intr/ccr, MMU, shape/stride, AI2D writes | `insn_system.h` | `insn_system.cpp` |
| AI2D compute | `insn_ai2d.h` | `insn_ai2d.cpp` |
| L2 load/store | `insn_l2.h` | `insn_l2.cpp` -> `../engines/l2load.*`, `l2store.*` |
| DM | `insn_dm.h` | `insn_dm.cpp` -> `../engines/dm.*` |
| PU | `insn_pu.h`, `pu_common.h` | `insn_pu.cpp` -> `../engines/conv2d.*`, `tcu.*`, `mne.*` |
| PU-PDP0 | `insn_pupdp0.h` | `insn_pupdp0.cpp` -> `../engines/pdp0.*` |
| ACT0 | `insn_act0.h` | `insn_act0.cpp` -> `../engines/act0.*` |
| MFU memcpy/memset/transpose | `insn_mfu_misc.h` | `insn_mfu_misc.cpp` -> `../engines/mfu.*` |
| MFU PDP1 | `insn_mfu_pdp1.h` | `insn_mfu_pdp1.cpp` -> `../engines/pdp1.*` |
| MFU ACT1 | `insn_mfu_act1.h` | `insn_mfu_act1.cpp` -> `../engines/meshnet.*` |
| Fetch/decode/execute loop | — | `../main.cpp` |

Each instruction class has: the decoded fields as members, a `Simulator::InstParser<I, 16|32>` specialisation (decode;
for scalar instructions also execute), `get_next_pc()` (`pc + 4` or `pc + 2`) and `operation()` (execute / configure).

---

## Open questions

* **Unused fields.** Many decoded fields are never read by `operation()`: `tcu_id`, `pu_id`, `l1_type`, `rlen` of
  `dm_load_act0`, `rgoc`/`rgoc_last`, `rpe_*`, `variant` of `act0_src1_conf`, and the reserved bits. Whether the real
  hardware uses them is unknown; the compiler only ever emits 0 for `tcu_id`/`pu_id`.
* **CCR and fences.** `ccr_*` and `fence` are no-ops here; their hardware synchronisation semantics are not modelled.
* **`pu_fetchif_conf4`.** The decoder reads `sspad` from `[29:27]` although the operand order would suggest `[26:22]`.
* **MFU ACT1 source/dest fields.** `mfu_act1_conf_src1/src2/dest` store the register *values* called
  `rslice_repeats`, `rleft_repeats` and `rlen` as source/destination addresses in some paths. The mapping between
  operand names and roles is of low-medium confidence.
* **Instructions not in the dumps.** `auipc`, the `lb/lbu/lh/lhu/sb/sh/sw` family, the branches, `jal/jalr`, `fence`,
  `fence_i`, `intr`, `mmu_setid`, `mfu_memcpy`, `mfu_memset`, `ai2d_compute`, `extrw/extraw`,
  `dm_conf_broadcast` and all `pu_pdp0_*` were verified only against the simulator binary, not against compiler output.
  Their assembly syntax in this document is by analogy.
* **Board behaviour not modelled.** `pu_forward_psum` after a `pu_compute` that did not accumulate writes nothing on
  a board (the PU waits on it) but works here, so the simulator accepts programs that fail on the device.
* **Missing PDP1 conf id.** `mfu_conf_id = 5` is rejected; it might be a missing `mfu_pdp1_conf5`.
* **Undefined integer results.** `div/rem` by zero crashes the host in the original; real hardware behaviour is unknown.
* **Timing.** The model is purely functional. Cycle counts, DMA overlap, GLB bank conflicts and the ordering of
  the MFU against the PU are not modelled.
