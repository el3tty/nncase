# K230 AI2D Device Specification

Reverse-engineered from the nncase K230 runtime sources (`ai2d_builder.cpp/.h`, `gnne_tile_utils.cpp/.h`) and
the `ai2d_test` test tool. This is **not** a vendor document.

**Evidence tags used throughout**

| Tag | Meaning |
|---|---|
| **[C]** | Stated by the driver source code |
| **[I]** | Inferred from the source (what the hardware must be doing for the code to make sense) |
| **[O]** | Observed in an output image produced on a K230 board |
| **[?]** | Unknown / unconfirmed |

The register bit map in section 5 was extracted mechanically: the real `ai2d_config::get_addr_value()` was compiled,
each field set to all ones in a zeroed config, and the changed register bits recorded. It therefore matches the
source exactly; whether the source matches the silicon is **[I]**.

---

## 1. Purpose

AI2D is a small 2-D image pre-processing engine on the K230 SoC. It sits beside the neural-network accelerator and
prepares camera or bitmap data for it, so the CPU does not have to: it reads an image from DRAM, applies a
geometric transform and a format conversion, and writes the result back to DRAM.

Typical use: take a camera frame (YUV420 or planar/packed RGB), crop it, resize or warp it to the model input size,
add letterbox padding, and convert it to planar RGB (`NCHW`), the layout the NN runtime consumes.

The engine is **tile based**. It processes the output in rectangular tiles. For each tile it loads the needed source window
from DRAM into an internal SRAM, computes the tile, and stores it to the output buffer. Software drives it by writing
a register file, and is notified of completion by an interrupt. **[C][I]**

---

## 2. System integration

### 2.1 Address map **[C]**

| Item | Value |
|---|---|
| Register page (physical) | `0x80400000` |
| Register window offset in the page | `0x0C00` (window runs `0xC00`..`0xFFF`, 1 KiB) |
| Register window base (physical) | `0x80400C00` |
| Mapped size | `0x1000` (one page) |
| Interrupt device node (Linux) | `/dev/k230-ai2d` (used only to wait; `poll()`) |
| Register access (Linux) | `/dev/mem`, opened `O_RDWR \| O_SYNC`, `mmap` of the page |

Offsets inside the window:

| Offset | Absolute | Size | Name | Section |
|---|---|---|---|---|
| `0x00`..`0x8F` | `0x80400C00`..`0x80400C8F` | 36 x 32 bit | Configuration registers `R0`..`R35` | 5 |
| `0x90` | `0x80400C90` | 16 bit (read) | Interrupt number (status) | 6 |
| `0xA0` | `0x80400CA0` | 64 bit (write) | CPU interrupt clear, writes `1` | 6 |
| `0xA8`, `0xAC` | `0x80400CA8`, `0x80400CAC` | 32 bit | Meaning unknown **[?]**. The driver writes `0` to both, in `ai2d_clear_cpu_intr()`, immediately after writing `1` to `0xA0` (interrupt clear) | 6 |
| `0xC0` | `0x80400CC0` | 32 bit | Timeout value (driver writes `0x1388` = 5000; unit unconfirmed **[?]**) | 6 |
| `0xC4`..`0xCC` | | 3 x 32 bit | Cleared to 0 together with the timeout | 6 |

### 2.2 Memory model

- **Image buffers live in DRAM** and must be physically contiguous: the driver writes the *physical* address of the
  tensor into the pointer registers. Only the low **32 bits** of the address are used. The memory comes from a
  contiguous physical-memory allocation made through `/dev/mmz`; the nncase runtime uses it when the tensor buffer is
  allocated with the `hrt::pool_shared` flag. **[C]**
- **Caches are not coherent** with the engine. The driver writes back the input and invalidates the output before
  starting (`sync_write_back`, `sync_invalidate`); callers should invalidate the output again before the CPU
  reads it. **[C][I]**
- **Internal SRAM.** The driver models a 256 x 256 element budget (`sram_len = 256`, `sram_size = 65536`) for the
  *source window* of one tile. Output tiles are chosen so that the source window they need fits. **[C]**
- Source and destination may each be in DDR or in the "global buffer" (`ai2d_data_loc::ddr = 1`, `glb = 0`);
  encoded in `src_ind` / `dst_ind`. The driver defaults to DDR. **[C]**

### 2.3 Data flow

Registers `R0`-`R3` hold the DRAM addresses of the **input** (source) planes and `R4`-`R7` those of the **output**
(destination) planes; see the full register map in [section 5.3](#53-register-map).

```
        DRAM: source planes
        pointers R0-R3, row pitch R8-R9
                   │
                   ▼
        ┌────────────────────────┐
        │ source window in SRAM  │   window origin/size: R34 / R30
        └────────────────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │ crop offset            │
        └────────────────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │ resize or affine       │   nearest | bilinear
        └────────────────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │ YUV → RGB (if csc_en)  │
        └────────────────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │ pad border             │
        └────────────────────────┘
                   │
                   ▼
        ┌────────────────────────┐
        │ output tile            │   tile size / origin: R31 / R35
        └────────────────────────┘
                   │
                   ▼
        DRAM: destination planes
        pointers R4-R7, row pitch R10-R11
```

The sequence is repeated for every tile, and each pass handles up to 4 channel planes.

---

## 3. Functional description

### 3.1 Processing order **[C][I]**

For each output tile the engine applies, in this order:

1. **Crop**: a window of the source. It is realised by the source-window origin (`src_x`, `src_y`, absolute) rather
   than by a separate stage.
2. **Geometric mapping**: *either* resize *or* affine. They are mutually exclusive: the driver throws
   `"We don't affine and resize simultaneously."` The mapping is a 2x3 matrix, output coordinates to input coordinates.
3. **Sampling**: nearest or bilinear.
4. **Format / colour conversion**: YUV to RGB when the source is YUV and the destination is planar.
5. **Pad**: constant, copy or mirror border around the output area.
6. **Shift** (RAW16 data only): bit shift of the samples.

### 3.2 Tile model **[C]**

- The output tensor is split into tiles: `N` in chunks of 1, channels in chunks of 4 (2 for RAW16), then `H` x `W` tiles.
  The padding-free interior is what gets tiled; tile coordinates are **0-based inside that interior**.
- For a **resize** tile the source window is derived from the two opposite output corners; for an **affine** tile it
  is the bounding box of all four mapped corners, clamped to the image (at least 1 x 1). With YUV420 the origin
  and size are forced even.
- The tile size comes from a search that grows the output tile until its source window no longer fits the SRAM
  budget (`resize_sram_search`, `affine_sram_search`).
- The first tile of every (n, c) block writes all 36 registers; later tiles rewrite only `R28`..`R35`.

### 3.3 Coordinate convention

Source and destination coordinates are pixel indices; the register matrix maps **output to input**
(see 5.4). The tool in this repository uses the OpenCV convention: `(0,0)` is the centre of the top-left pixel,
`+y` points down, and a positive rotation is clockwise on screen. **[C]** for the register meaning; the convention
is the tool's.

---

## 4. Supported operations

### 4.1 Formats and layouts **[C]**

| Code | `ai2d_format` | Layout in DRAM | Planes (`dst_channel`) | Row pitch registers |
|---|---|---|---|---|
| 0 | `YUV420_NV12` | Y plane, then interleaved UV; buffer is `3H/2` rows | 2 | W for all planes |
| 1 | `YUV420_NV21` | Y plane, then interleaved VU | 2 | W |
| 2 | `YUV420_I420` | Y, then U, then V planes (chroma rows W/2) | 3 | ch0 = W, ch1 = ch2 = W/2 |
| 3 | `NCHW_FMT` | Planar: one full plane per channel (RRR.. GGG.. BBB..) | = channel count | W |
| 4 | `RGB_packed` | Interleaved RGBRGB.. (NHWC), one plane | 1 | ch0 = 3W |
| 5 | `RAW16` | 16-bit samples, four planes | 4 | 2W (bytes) for all four |

The tensor shape given to the driver is NCHW. For YUV the shape's `H` is the **stored** height (`3H/2`) and the driver
derives the logical height as `2H/3` and the channel count as 3. For `RGB_packed` the shape is NHWC.

### 4.2 Format conversion matrix **[C]**

The driver sets `csc_en`, `dst_channel` and the format codes from the (source, destination) pair:

| Source | Destination | `csc_en` | Result |
|---|---|---|---|
| NV12, NV21, I420 | `NCHW_FMT` | **1** | YUV to planar RGB (`dst_channel = 3`) |
| NV12, NV21, I420 | any other format | 0 | Geometry only, no conversion (`dst_channel` 2 or 3) |
| `RGB_packed` | `NCHW_FMT` | 0 | De-interleave to planar (`dst_channel = 3`) |
| `RGB_packed` | any other format | 0 | One interleaved plane out |
| `NCHW_FMT`, `RAW16` | `NCHW_FMT` | 0 | Planar to planar |
| `NCHW_FMT`, `RAW16` | any other format | 0 | Format bits copied from the destination |

Exercised by the `ai2d_test` tool: `NCHW → NCHW` and `NV12/NV21/I420 → NCHW`. The other combinations are
accepted by the driver but their hardware behaviour is **[?]**. There is no RGB-to-YUV conversion: only YUV-to-RGB
coefficient registers exist.

### 4.3 Crop **[C]**

Parameters: `start_x`, `start_y`, `width`, `height` inside the input image. Checks: `0 ≤ start_x < in_w`,
`start_x + width ≤ in_w` (same for y), otherwise `"Crop param(x) error."`. Without crop the window is the whole
image. The affine matrix and resize scale are relative to the **crop window**; the register `src_x`/`src_y` holds the
absolute position (crop start added). For YUV sources the window should be even-aligned.

### 4.4 Resize **[C]**

`scale_w = in_w / out_w`, `scale_h = in_h / out_h` (of the crop window and the padding-free output). The
output-to-input map is diagonal: `M0 = scale_w`, `M4 = scale_h`, `M2`/`M5` = bias.

| Interpolation | Mode | Scale | Bias | `interpolation` | `cord_round` |
|---|---|---|---|---|---|
| bilinear (`tf_`/`cv2_bilinear`) | `half_pixel`, or any `cv2_bilinear` | in/out | `0.5*scale - 0.5` | 1 | 0 |
| bilinear | `align_corner` | (in-1)/(out-1) | 0 | 1 | 0 |
| bilinear | `none` | in/out | 0 | 1 | 0 |
| nearest | `half_pixel` | in/out | `0.5*scale` | 0 | 2 |
| nearest | `align_corner` | (in-1)/(out-1) | 0 | 0 | 0 |
| nearest | `none` | in/out | 0 | 0 | 2 |

### 4.5 Affine **[C]**

The caller supplies the **forward** 2x3 matrix `{a, b, tx, c, d, ty}` (input to output). The driver inverts it and the
registers hold the **output-to-input** map (5.4). Parameters: interpolation, `cord_round` (2 bits), `bound_ind`
(4 bits), `bound_val` (16 bits), `bound_smooth` (1 bit).

- The interpolation bit is taken from the **resize parameter's** `interp_method` (`tf_/cv2_bilinear` = 1, else 0),
  not from the affine parameter. A caller must set both consistently.
- `bound_val` is the value the engine writes for output samples whose source position lies outside the
  source window: grey `127,127,127` appears in the corners of a rotated image when `bound_val = 127`. **[O]**
- The meaning of `bound_ind` and `bound_smooth` is **[?]**; the upstream callers pass `0` and `1`.
- `cord_round` selects how the mapped coordinate is rounded for sampling; the upstream callers use 0 for bilinear and
  2 for nearest. The exact semantics are **[?]**.
- The inverse of the matrix must exist (`det != 0`).

### 4.6 Pad **[C]**

Four-dimensional `paddings` (before/after for N, C, H, W); only H and W are used. The output tensor includes
the border: `out_h = interior_h + pad_t + pad_b`, likewise for width. If every pad is 0 the driver clears the pad flag.

| `pad_mod` | Mode |
|---|---|
| 0 | `constant`: per-channel values in `const_pad_ch0..3` (RAW16: 16-bit values) |
| 1 | `copy` (replicate the edge) |
| 2 | `mirror` |

Per tile, `pad_t/b/l/r` are non-zero only for tiles that touch the matching edge of the output.
`dst_x`/`dst_y` carry `pad_before + tile_start` for tiles that start inside the interior; for a **first** tile
(start 0) they are 0 and the hardware applies `pad_l`/`pad_t` itself. **[I]**
The driver prints `"left padding is not supported when width <= 32!, may cause hardware panic!"` for left padding
on a tile at most 32 wide.

### 4.7 Shift **[C]**

`shift_val` is written to `shift` (8 bits; negative values + 32). Allowed only with a `RAW16` source and an
`NCHW_FMT` or `RAW16` destination, otherwise `"Only Raw16(src) support shift."`

### 4.8 YUV-to-RGB conversion **[C][I]**

Enabled by `csc_en = 1` (YUV source, planar destination). Twelve 12-bit two's-complement coefficients in `R20`..`R24`
and `R26`. Read as three rows of four, `[Y, U, V, bias]` per output channel, with the first three scaled by 1/256:

| Row | coef | Default | Reads as |
|---|---|---|---|
| R | 0..3 | 256, 0, 292, -146 | `Y + 1.141*V - 146` |
| G | 4..7 | 256, -101, -149, 125 | `Y - 0.395*U - 0.582*V + 125` |
| B | 8..11 | 256, 520, 0, -260 | `Y + 2.031*U - 260` |

The bias terms equal `-coefficient * 128 / 256`, which confirms the layout. The defaults are not exactly any standard
matrix (see section 9, item 1); the driver never overrides them.

### 4.9 Channel handling

- Up to **4 planes per pass** (`R0..R3`, `R4..R7`).
- `NCHW → NCHW` with different channel counts: one input plane is broadcast to all output channels
  (`broadcast_in_channel`; the plane stride becomes 0). **[C]**
- `channel` and `dst_channel` are 3-bit fields.

### 4.10 Data types **[C]**

Pixel data is 8-bit (`uint8`) or 16-bit (`RAW16`). `sign` is 0 for unsigned 8/16-bit sources and 1 for other types.
Plane offsets use the element size of `src_type` / `dst_type` (1, 2 or 4 bytes).

### 4.11 Limits

| Limit | Value | Source |
|---|---|---|
| Source window per tile | ≤ 256 x 256 = 65536 elements | SRAM budget **[C]** |
| Affine source window | bounding box + 3 pixels in each direction must fit | `try_allocate_affine_sram` **[C]** |
| Source window width/height | 16 bits each | `R30` **[C]** |
| Destination tile width | 16 bits | `R31` **[C]** |
| Destination tile height | 14 bits (16383) | `R31` **[C]** |
| Destination `y` origin | 13 bits (8191) | `R35` **[C]** |
| Destination `x` origin, source `x`/`y` | 16 bits | `R34`, `R35` **[C]** |
| Planes per pass | 4 (2 for RAW16) | `c_step` **[C]** |
| Register bytes per interrupt batch | ≤ 1024 | `build_schedule` **[C]** |
| Physical buffer address | 32 bits | register width **[C]** |

### 4.12 Not supported

Rotation by an arbitrary angle is not a separate operation: it is an affine matrix. Resize and affine in one pass
and RGB-to-YUV conversion are not available. Whether cropping a YUV source to an odd offset works is **[?]**
(the driver only enforces evenness for the tile windows it computes).

---

## 5. Register file

### 5.1 Organisation

36 consecutive 32-bit registers `R0`..`R35` (144 bytes) at `0x80400C00`. `R(i)` is at byte offset `4*i`. **[C]**
Bit positions below are exact relative to `get_addr_value()`. "Static" registers are rewritten only on the first tile
of each (n, c) block; "per tile" registers (`R28`..`R35`) are rewritten for every tile.

Reserved fields are always written as 0.

### 5.2 Write protocol **[C]**

- The driver writes the registers in **groups of four**, in ascending order: `R(i)`, `R(i+1)`, `R(i+2)`, `R(i+3)`.
- A full update is 9 groups (144 bytes, `R0..R35`); a partial update is 2 groups (32 bytes, `R28..R35`).
- For the pointer registers the driver adds the physical address of the input tensor (`R0`..`R3` group) or of the
  output tensor (`R4`..`R7` group) at write time; the `config_` values hold offsets only.
- **Start trigger [I]:** every config carries `ai2d_calc_enable = 1` (`R35[31]`). The last group of every update contains
  `R35`, so writing it is the probable start of the tile. The driver contains no other write that could start the
  engine, and `/dev/k230-ai2d` is used only to wait.
- **Command queue [I]:** a batch of several tiles is written to the same register addresses *before* the driver waits
  once, and the batch size is capped at 1024 bytes, which equals the size of the register window (`0xC00`..`0xFFF`).
  This indicates the window accepts several queued register updates while an earlier tile is still running
  (the device most likely starts a tile when `R35` is written, without waiting for the end of the batch);
  only the last tile of a batch raises an interrupt (section 6).

### 5.3 Register map

Columns: register index, offset from `0x80400C00`, absolute address, bit range, field, meaning, reset value of the
whole register (the `ai2d_config` constructor default).

| Reg | Offset | Address | Bits | Field | Meaning | Default |
|---|---|---|---|---|---|---|
| 0 | 0x000 | 0x80400C00 | [31:0] | `src_ch0_ptr` | Source plane 0: DRAM physical address of the plane (input buffer base + channel/plane offset) | 0x00000000 |
| 1 | 0x004 | 0x80400C04 | [31:0] | `src_ch1_ptr` | Source plane 1 (plane 0 + one plane) | 0x00000000 |
| 2 | 0x008 | 0x80400C08 | [31:0] | `src_ch2_ptr` | Source plane 2 | 0x00000000 |
| 3 | 0x00C | 0x80400C0C | [31:0] | `src_ch3_ptr` | Source plane 3 | 0x00000000 |
| 4 | 0x010 | 0x80400C10 | [31:0] | `dst_ch0_ptr` | Destination plane 0: DRAM physical address (output buffer base + offset) | 0x00000000 |
| 5 | 0x014 | 0x80400C14 | [31:0] | `dst_ch1_ptr` | Destination plane 1 | 0x00000000 |
| 6 | 0x018 | 0x80400C18 | [31:0] | `dst_ch2_ptr` | Destination plane 2 | 0x00000000 |
| 7 | 0x01C | 0x80400C1C | [31:0] | `dst_ch3_ptr` | Destination plane 3 | 0x00000000 |
| 8 | 0x020 | 0x80400C20 | [31:16] | `src_ch1_width_layout` | Source plane 1 row pitch in bytes | 0x00000000 |
|  |  |  | [15:0] | `src_ch0_width_layout` | Source plane 0 row pitch in bytes |  |
| 9 | 0x024 | 0x80400C24 | [31:16] | `src_ch3_width_layout` | Source plane 3 row pitch in bytes | 0x00000000 |
|  |  |  | [15:0] | `src_ch2_width_layout` | Source plane 2 row pitch in bytes |  |
| 10 | 0x028 | 0x80400C28 | [31:16] | `dst_ch1_width_layout` | Destination plane 1 row pitch in bytes | 0x00000000 |
|  |  |  | [15:0] | `dst_ch0_width_layout` | Destination plane 0 row pitch in bytes |  |
| 11 | 0x02C | 0x80400C2C | [31:16] | `dst_ch3_width_layout` | Destination plane 3 row pitch in bytes | 0x00000000 |
|  |  |  | [15:0] | `dst_ch2_width_layout` | Destination plane 2 row pitch in bytes |  |
| 12 | 0x030 | 0x80400C30 | [31:0] | `M0` | float32 bits of 1024 x a (x_in = a*x_out + b*y_out + e) | 0x44800000 |
| 13 | 0x034 | 0x80400C34 | [31:0] | `M1` | float32 bits of 1024 x b | 0x00000000 |
| 14 | 0x038 | 0x80400C38 | [31:0] | `M3` | float32 bits of 1024 x c (y_in = c*x_out + d*y_out + f) | 0x00000000 |
| 15 | 0x03C | 0x80400C3C | [31:0] | `M4` | float32 bits of 1024 x d | 0x44800000 |
| 16 | 0x040 | 0x80400C40 | [31:0] | `reserved0` |  | 0x00000000 |
| 17 | 0x044 | 0x80400C44 | [31:0] | `reserved1` |  | 0x00000000 |
| 18 | 0x048 | 0x80400C48 | [31:28] | `dst_format` | Destination format code 0..5 (section 5.1) | 0x00000000 |
|  |  |  | [27:24] | `src_format` | Source format code 0..5 (section 5.1) |  |
|  |  |  | [23:20] | `bound_ind` | Affine boundary-handling selector (semantics unconfirmed) |  |
|  |  |  | [19:12] | `shift` | Right/left shift amount for RAW16 data; negative values stored as value + 32 |  |
|  |  |  | [11:10] | `pad_mod` | 0 constant, 1 copy (edge replicate), 2 mirror |  |
|  |  |  | [9:8] | `interpolation` | 0 nearest, 1 bilinear |  |
|  |  |  | [7:6] | `cord_round` | Coordinate rounding mode (see 5.3/5.4) |  |
|  |  |  | [5:3] | `dst_channel` | Number of destination planes (NV12/NV21 2, I420 3, RGB_packed 1, planar = channel) |  |
|  |  |  | [2:0] | `channel` | Number of source channels/planes handled in this pass (NCHW/RAW16: refined per tile) |  |
| 19 | 0x04C | 0x80400C4C | [31:17] | `reserved2` |  | 0x00000000 |
|  |  |  | [16] | `bound_smooth` | Affine boundary smoothing enable (semantics unconfirmed) |  |
|  |  |  | [15:0] | `bound_val` | Affine: value used for samples that fall outside the source |  |
| 20 | 0x050 | 0x80400C50 | [31:24] | `reserved3` |  | 0x00000100 |
|  |  |  | [23:12] | `yuv2rgb_coef1` | YUV-to-RGB coefficient 1 (12-bit two's complement) |  |
|  |  |  | [11:0] | `yuv2rgb_coef0` | YUV-to-RGB coefficient 0 (12-bit two's complement) |  |
| 21 | 0x054 | 0x80400C54 | [31:24] | `reserved4` |  | 0x00F6E124 |
|  |  |  | [23:12] | `yuv2rgb_coef3` | YUV-to-RGB coefficient 3 (12-bit two's complement) |  |
|  |  |  | [11:0] | `yuv2rgb_coef2` | YUV-to-RGB coefficient 2 (12-bit two's complement) |  |
| 22 | 0x058 | 0x80400C58 | [31:24] | `reserved5` |  | 0x00F9B100 |
|  |  |  | [23:12] | `yuv2rgb_coef5` | YUV-to-RGB coefficient 5 (12-bit two's complement) |  |
|  |  |  | [11:0] | `yuv2rgb_coef4` | YUV-to-RGB coefficient 4 (12-bit two's complement) |  |
| 23 | 0x05C | 0x80400C5C | [31:24] | `reserved6` |  | 0x0007DF6B |
|  |  |  | [23:12] | `yuv2rgb_coef7` | YUV-to-RGB coefficient 7 (12-bit two's complement) |  |
|  |  |  | [11:0] | `yuv2rgb_coef6` | YUV-to-RGB coefficient 6 (12-bit two's complement) |  |
| 24 | 0x060 | 0x80400C60 | [31:24] | `const_pad_ch0` | Constant pad value, channel 0 | 0x00EFC000 |
|  |  |  | [23:12] | `yuv2rgb_coef11` | YUV-to-RGB coefficient 11 (12-bit two's complement) |  |
|  |  |  | [11:0] | `yuv2rgb_coef10` | YUV-to-RGB coefficient 10 (12-bit two's complement) |  |
| 25 | 0x064 | 0x80400C64 | [31:28] | `reserved7` |  | 0x00000000 |
|  |  |  | [27] | `sign` | 0 for unsigned 8/16-bit source data, 1 otherwise |  |
|  |  |  | [26] | `cmd_id` | Command id (always 0 in this driver) |  |
|  |  |  | [25] | `dst_ind` | 1 = destination in DDR, 0 = global buffer |  |
|  |  |  | [24] | `src_ind` | 1 = source in DDR, 0 = source in the global buffer (ai2d_data_loc) |  |
|  |  |  | [23:16] | `const_pad_ch3` | Constant pad value, channel 3 |  |
|  |  |  | [15:8] | `const_pad_ch2` | Constant pad value, channel 2 |  |
|  |  |  | [7:0] | `const_pad_ch1` | Constant pad value, channel 1 |  |
| 26 | 0x068 | 0x80400C68 | [31:24] | `reserved8` |  | 0x00208100 |
|  |  |  | [23:12] | `yuv2rgb_coef9` | YUV-to-RGB coefficient 9 (12-bit two's complement) |  |
|  |  |  | [11:0] | `yuv2rgb_coef8` | YUV-to-RGB coefficient 8 (12-bit two's complement) |  |
| 27 | 0x06C | 0x80400C6C | [31:0] | `reserved9` |  | 0x00000000 |
| 28 | 0x070 | 0x80400C70 | [31:16] | `pad_b` | Rows of padding below this tile | 0x00000000 |
|  |  |  | [15:0] | `pad_t` | Rows of padding above this tile |  |
| 29 | 0x074 | 0x80400C74 | [31:16] | `pad_r` | Columns of padding right of this tile | 0x00000000 |
|  |  |  | [15:0] | `pad_l` | Columns of padding left of this tile |  |
| 30 | 0x078 | 0x80400C78 | [31:16] | `src_height_shape` | Height of the source window read for this tile | 0x00000000 |
|  |  |  | [15:0] | `src_width_shape` | Width of the source window read for this tile |  |
| 31 | 0x07C | 0x80400C7C | [31] | `csc_en` | 1 = colour-space conversion (YUV to RGB) enabled | 0x40000000 |
|  |  |  | [30] | `intr_mask` | 1 = do not raise an interrupt after this tile, 0 = raise one |  |
|  |  |  | [29:16] | `dst_height_shape` | Height of the destination tile (14 bits) |  |
|  |  |  | [15:0] | `dst_width_shape` | Width of the destination tile |  |
| 32 | 0x080 | 0x80400C80 | [31:0] | `M2` | float32 bits of 1024 x e (x offset of the output-to-input map, re-origined per tile) | 0x00000000 |
| 33 | 0x084 | 0x80400C84 | [31:0] | `M5` | float32 bits of 1024 x f | 0x00000000 |
| 34 | 0x088 | 0x80400C88 | [31:16] | `src_y` | Y of the source window origin | 0x00000000 |
|  |  |  | [15:0] | `src_x` | X of the source window origin (absolute, crop included) |  |
| 35 | 0x08C | 0x80400C8C | [31] | `ai2d_calc_enable` | Calculation enable; always 1 in this driver, probable start bit | 0x80000000 |
|  |  |  | [30:29] | `reserved10` |  |  |
|  |  |  | [28:16] | `dst_y` | Y of the destination tile origin (13 bits) |  |
|  |  |  | [15:0] | `dst_x` | X of the destination tile origin in the output buffer |  |

### 5.4 Encodings

**Matrix registers (`R12`, `R13`, `R14`, `R15`, `R32`, `R33`).** Each holds the bit pattern of an IEEE-754 `float32`
equal to **1024 x** the coefficient. Dividing by 1024 gives the output-to-input map **[C]**:

```
x_in = M0 * x_out + M1 * y_out + M2
y_in = M3 * x_out + M4 * y_out + M5
```

The reset value is the identity: `M0 = M4 = 1024.0f` (`0x44800000`), the others 0.
Resize writes only `M0`, `M4`, `M2`, `M5`. Affine writes all six, from the inverse of the caller's forward matrix.
For every tile the driver re-origins `M2`/`M5` so the mapping is relative to that tile's source window.

**`interpolation` / `cord_round`** are described in 4.4 and 4.5.

**Format code** (`src_format`, `dst_format`): 0 NV12, 1 NV21, 2 I420, 3 NCHW, 4 RGB_packed, 5 RAW16. For a
YUV source with a planar destination the driver forces `dst_format = 3`, `dst_channel = 3`, `csc_en = 1`.

**Pointer registers.** For planar data: plane `k` is at `base + k * plane_bytes + channel_tile_offset`, with
`plane_bytes = H * W * element_size` (logical `H`). For I420 the third plane starts a quarter plane after the second.
`ch3` is `ch2 + plane_bytes` (even for formats with fewer planes).

### 5.5 Registers per tile

| Written for every tile | Why |
|---|---|
| `R28`, `R29` | Pad flags of the edges this tile touches |
| `R30` | Source window size |
| `R31` | Destination tile size, `intr_mask`, `csc_en` |
| `R32`, `R33` | `M2`, `M5` re-origined to the tile |
| `R34` | Source window origin (`src_x`, `src_y`) |
| `R35` | Destination origin (`dst_x`, `dst_y`) and `ai2d_calc_enable` |

---

## 6. Interrupts and completion

| Item | Detail |
|---|---|
| Completion event | Interrupt from the engine; on Linux the kernel driver wakes `poll(/dev/k230-ai2d, POLLIN)` **[C][I]** |
| Per-tile mask | `R31[30]` `intr_mask`: `1` = no interrupt after this tile, `0` = raise an interrupt. Reset value 1 **[C]** |
| Status | 16-bit **interrupt number** read at `0x80400C90` after the wake-up **[C]** |
| Codes | `0` = `AI2D_INTR_OK`, `1` = `AI2D_INTR_TIME_OUT`, `2` = `AI2D_INTR_EXCEPTION` **[C]** |
| Timeout | Register at `0x80400CC0` set to `0x1388` (5000) by the driver; unit unknown **[?]**; hardware raises code 1 when exceeded **[I]** |
| Clear | `ai2d_clear_cpu_intr()` writes `1` (64 bit) to `0xA0` and `0` to `0xA8`, `0xAC`. In the Linux driver it is not called after the wake-up; the kernel side presumably does it **[?]** |
| Bare-metal | The non-Linux build uses a `volatile bool ai2d_done_` set from the interrupt handler instead of `poll()` **[C]** |

**Batching rule [C].** The driver does not wait for the engine after every tile. It groups consecutive tiles into
*batches*, writes the register updates of the whole batch back to back, and then waits for **one** interrupt. The terms:

- **Register group.** The unit of writing: four consecutive registers (16 bytes), written in ascending order.
- **Full update (144 bytes).** Nine groups covering `R0`..`R35`. It is used for the first tile of every (n, c) block, i.e.
  whenever the channel or batch index changes, because the static registers (formats, pointers, strides, matrices, pad
  constants, ...) must then be reloaded.
- **Partial update (32 bytes).** Two groups covering `R28`..`R35`. It is used for every other tile of the block, since only
  the per-tile registers change (section 5.5).
- **Batch.** The tiles whose register updates are written between two interrupts. In the driver it is the interval
  between two consecutive entries of `split_pos_` (the list of positions in the register-group list at which an
  interrupt is expected). Every tile except the last of a batch has `intr_mask = 1` (no interrupt); the last tile of
  a batch has `intr_mask = 0` (raise an interrupt when this tile has finished).
- **Batch size limit (1024 bytes).** The total number of register bytes in one batch is capped at 1024, the size of the
  register window (`0xC00`..`0xFFF`). It is probably the capacity of the engine's queue of pending register updates **[I]**;
  the driver treats it as a hard limit.

While building the schedule the driver keeps a counter `batch_bytes`, the register bytes queued since the last interrupt.
For each tile, in processing order:

1. Add the size of this tile's update (144 for a full update, 32 for a partial one) to `batch_bytes`.
2. Work out `next_bytes`, the size of the *next* tile's update: 32 if the next tile is in the same (n, c) block;
   144 if this is the last tile of the block and another channel segment follows; 0 if this is the very last tile.
3. If `batch_bytes + next_bytes > 1024`, the next tile would not fit: set `intr_mask = 0` on **this** tile, close the batch
   (record a split position) and reset `batch_bytes` to 0.
4. Otherwise set `intr_mask = 1`, except on the very last tile of the whole job, which always gets `intr_mask = 0`.

Consequences:

| Situation | Largest batch |
|---|---|
| Batch starts with a full update | 1 full + 27 partial = 28 tiles, 1008 bytes (a 29th tile would make 1040) |
| Batch of partial updates only | 32 tiles, exactly 1024 bytes |
| Job smaller than 1024 bytes in total | One batch, one interrupt, at the very last tile |

**What a batch looks like in memory.** The addresses below are device registers (MMIO), not DRAM. The CPU writes them through
an uncached mapping, so every store reaches the device immediately.

How the device most likely handles a batch **[I]**: it starts processing a tile when the write to `R35` arrives
(`R35` carries `ai2d_calc_enable`, and it is the last register written for every tile). It does not wait for the end of
the batch. The CPU keeps writing the next tiles' registers while the device is still busy, so the device must queue those
writes, and apply them in order once the tile in progress has finished; the 1024-byte batch limit is probably the capacity
of this queue. The `intr_mask` bit (`R31[30]`) only controls whether the device raises an interrupt when that tile completes:
tiles with `intr_mask = 1` complete silently and the device moves on to the next queued update, and the tile with
`intr_mask = 0` ends the batch and wakes the driver, which has been waiting in `poll()` since it wrote the batch.

Take a batch of three tiles: the first tile of an (n, c) block, then two more tiles of the same block. `invoke()` stores
every register group at its own register address, so a partial update always lands on `R28`..`R35` (`0x80400C70`..`0x80400C8F`),
never after the previous update **[C]**:

```
 address    0x80400C00                                      0x80400C70       0x80400C8F
            │                                               │                │
 tile 1     R0 R1 R2 ......................................  R28 ........... R35     full update, 144 B
 tile 2                                                      R28 ........... R35     partial update, 32 B
 tile 3                                                      R28 ........... R35     partial update, 32 B
                                                                                      (only tile 3 has intr_mask = 0)
```

The three updates reuse the same addresses; the device keeps them apart by the order in which the writes arrive, not by
address. Each tile starts when its `R35` write (`0x80400C8C`) arrives. The window holds the interrupt number at
`0x80400C90`, so nothing is ever written after `0x80400C8F`.

A batch may contain a full update in the middle (when the channel segment changes), in which case the count of tiles
is smaller. Because the rule looks only one tile ahead, a batch never exceeds 1024 bytes.
The reconstructed driver does not count the 144 bytes of the next tile when this is the last tile of the last channel
segment but not of the last `N` segment; this is carried over from the original code and is **[?]**
(it can only matter for `N > 1`, which the tool does not use).

**Software handling [C].** `invoke()` writes all register groups of one batch (adding the physical base address to the pointer
groups at `R0`..`R3` and `R4`..`R7`), calls `poll()` once on `/dev/k230-ai2d` with no timeout, reads the
interrupt number and then:

- `1` (timeout): prints `ai2d timeout!`, error `timed_out`.
- `2` (exception): prints `ai2d exception!`, error `invalid_argument`.
- otherwise the next batch is started.

The `poll()` itself has no timeout (`-1`); the only timeout is the hardware's. Causes of an exception interrupt are **[?]**
(an invalid configuration is the obvious candidate).

---

## 7. Programming sequence

1. Allocate input and output tensors in physically contiguous memory; write the input pixels.
2. Build the configuration: formats, crop, pad, resize or affine, shift (`ai2d_builder` constructor, `check_config`).
3. `build_schedule()`: compute the static registers, search the tile size, and for each tile compute the dynamic
   registers (source window, destination tile, pad flags, `M2`/`M5` bias, `intr_mask`) into a list of register groups.
4. `invoke(input, output)`:
   1. write back the input cache, invalidate the output cache;
   2. for each batch: write its register groups (adding the physical base to the pointer registers), then wait;
   3. check the interrupt number.
5. Invalidate the output cache again and read the result.

---

## 8. Observed behaviour

These come from output images produced on a K230 board by the reconstructed driver, not from vendor documentation.

| # | Observation | Tag |
|---|---|---|
| 1 | Out-of-source samples of a rotated image read `127,127,127` when `bound_val = 127` | **[O]** |
| 2 | Crop + resize + rotate (no pad) produces the expected image | **[O]** |
| 3 | When a tile was programmed with a destination origin too far right/down, the image overran the right and bottom edges: the overflow wrapped onto the left columns and into the first rows of the next channel plane. So writes are **not clipped** to the destination width/height | **[O][I]** |

---

## 9. Known gaps and open questions

| # | Topic | Detail |
|---|---|---|
| 1 | YUV coefficients | The defaults do not match BT.601 full range (`R-V 1.402, B-U 1.772, G -0.344 -0.714`). The U terms resemble BT.601 limited range (`2.018`, `-0.391`) but the V terms are about 0.71 times too small and the Y gain is 1.0 instead of 1.164. Either the real device uses a different matrix or two coefficients are mis-reconstructed. A tool that encodes with BT.601 will show a colour cast on the board |
| 2 | `bound_ind`, `bound_smooth`, `cord_round` | Exact meanings unknown; values used are those of upstream callers |
| 3 | Timeout register | Unit of `0x1388` unknown |
| 4 | Interrupt clearing | Who writes the clear registers on Linux is not visible in the driver |
| 5 | Format pairs other than the two tested | `RGB_packed` input, packed or YUV output and `RAW16` have never been exercised |
| 6 | Exception causes | Which invalid configurations raise interrupt code 2 is unknown |
| 7 | Earlier grey output | An all-grey result was seen with an earlier version of the reconstruction; its cause was not isolated here |

---

## Appendix A. Source map

| Topic | Where |
|---|---|
| Register struct, bit packing (`get_addr_value`) | `gnne_tile_utils.h`, `struct ai2d_config` |
| Static register setup (formats, interpolation, pad, widths) | `gnne_tile_utils.cpp`, `update_static_param` |
| Per-tile registers (windows, pad flags, pointers) | `gnne_tile_utils.cpp`, `update_dynamic_param` |
| Matrix setup and inversion | `update_M_param`, `inv_M`, `M_mul_add` |
| Tile size search | `resize_sram_search`, `affine_sram_search`, `try_allocate_*` |
| Register group packing | `update_regs` |
| Device access, batching, interrupts | `ai2d_builder.cpp` (constructor, `invoke`, `build_schedule`), `ai2d_builder.h` |
| Test tool | `ai2d_test.cpp` (`Plan`) |
