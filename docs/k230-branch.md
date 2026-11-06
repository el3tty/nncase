# The k230 branch of nncase

[nncase](https://github.com/kendryte/nncase) is a compiler for neural networks. It turns a trained model into a
`.kmodel` file that an AI accelerator can run. The Kendryte K230 is a small RISC-V chip with such an accelerator, an NPU
(neural processing unit), which this project also calls the KPU.

This branch makes the whole K230 path buildable from this repository: the compiler plugin, the board runtime, a simulator
of the NPU, and tools to test them. Upstream's Python packages `nncase` and `nncase-kpu` do not contain these changes,
so `pip install nncase` will not reproduce the results below.

The branch is 174 commits on top of upstream commit `99431d9db`. It is a personal fork and is not supported by Canaan.

* [What the branch adds](#what-the-branch-adds)
* [Where the K230 code came from](#where-the-k230-code-came-from)
* [Results: Lite-Mono on the board](#results-lite-mono-on-the-board)
* [Building](#building)
* [Getting the runtime onto the board](#getting-the-runtime-onto-the-board)
* [Further reading](#further-reading)

## What the branch adds

### Compiler

* The K230 target as source, in [`modules/Nncase.Modules.K230`](../modules/Nncase.Modules.K230), loaded by the
  command-line compiler `Nncase.Cli`.
* Models that were already quantized (converted to 8-bit numbers) can be compiled as they are. ONNX models with
  QuantizeLinear / DequantizeLinear pairs compile for the K230 without a calibration dataset, and onnxruntime QOperator
  models import too.
* Quantization controls: export and reuse a quant scheme (`--export-quant-scheme`, `--quant-scheme`,
  `--quant-scheme-strict-mode`), and resize or colour options for calibration images (`--calib-image-resize`,
  `--calib-bgr`).
* `CombineQuantizePad`, a graph rule that moves quantization above a Pad, so the data stays 8-bit through it.
* Correctness fixes for the K230, among them float16 reductions, half-pixel resizes on the AI2D image unit and 1x1
  transposed convolutions.
* Builds on Windows and Linux, on .NET 9.

### Runtime

* The K230 runtime as source, in [`src/Native/src/runtime/k230`](../src/Native/src/runtime/k230): it drives the NPU
  through `/dev/k230-gnne` and allocates memory through `/dev/mmz`.
* Python bindings for the board (`nncaseruntime`), cross-compiled for riscv64.
* Faster CPU kernels for Pad, Broadcast and GetItem. Some operations of a model run on the CPU cores rather than the NPU,
  and these three were the largest part of Lite-Mono's time.
* Reliability fixes: a 10 s limit on waiting for the NPU, with a state dump on a time-out, device memory freed at exit,
  a memory leak, and failed allocations reported.

### Simulator

* A C++ simulator of the NPU in [`src/Native/src/simulator`](../src/Native/src/simulator), so a `.kmodel` can run on a PC
  without a board. `nncaseruntime` on the PC uses it.
* A written description of the instruction set in its
  [README](../src/Native/src/simulator/README.md), including cases where the simulator and the board differ.
* Optional debug dumps, read with [`tools/ckpt_dump.py`](../tools/ckpt_dump.py).

### Tools and tests

* `k230-as`, an assembler for the NPU instruction set.
* `k230_isa_test`, 70 small assembly programs that run on the simulator and on the board, to compare the two.
* [`tests/run_k230_tests.py`](../tests/run_k230_tests.py), which runs the Python operator tests on the simulator, on a
  board over ssh, or against stored assembly output. See [`tests/ReadMe.txt`](../tests/ReadMe.txt).
* `ai2d_test`, a test tool for the AI2D image unit.

## Where the K230 code came from

Getting Lite-Mono from 527 ms to 61 ms took knowing exactly how the compiler and the KPU behave: which operations stay on
the NPU, how they get tiled, and what forces a fallback to the CPU. Little of that is documented, and upstream's repository
has no source for the K230 plugin, runtime or simulator, which Canaan ships as binaries. The sources here were recovered
from those binaries, so that the stack could be read, checked against a real board, and changed:

* **Compiler plugin:** decompiled from `Nncase.Modules.K230.dll` in the `nncase-kpu` Python package, then cleaned up.
* **Simulator:** reverse engineered from `nncase.simulator.k230.sc` in the same package, then refactored.
* **Board runtime:** reverse engineered from `libnncase.rt_modules.k230.a` and `libfunctional_k230.a` in
  `nncase_k230_v2.11.0_runtime_linux.tgz`, an asset of upstream's
  [v2.11.0 release](https://github.com/kendryte/nncase/releases/tag/v2.11.0). The K230 Linux SDK downloads the same
  archive.

The assembler, the ISA tests, the board test runner, the Lite-Mono fixes and the bug fixes listed above are new work. The
instruction set description was derived from the simulator code and checked against a board where the tests cover it.
Because it was not written from a specification, parts of it may be wrong. Its
[open questions](../src/Native/src/simulator/README.md#open-questions) list what is known to be uncertain.

**Licence.** `nncase-kpu` is published on PyPI under the Apache License 2.0, with the notice "Copyright 2018 Canaan Inc.",
and this branch relies on that licence for the compiler plugin and the simulator. The runtime archive has no licence
file of its own. It is a release asset of upstream's Apache 2.0 repository, and this branch treats it as covered by
that licence. The branch is distributed under the same [licence](../LICENSE). Canaan has not reviewed or endorsed
this work.

## Results: Lite-Mono on the board

[Lite-Mono](https://github.com/noahzn/Lite-Mono) estimates how far away everything in a picture is, from a single camera
image. It mixes convolutions with transformer-style attention. These numbers are for its smallest variant,
lite-mono-tiny, with a 640x192 input and 8-bit quantization, on a K230 board. Time per image covers setting the input,
running the model and reading the output.

| Change | Where | Time per image |
|---|---|---|
| Starting point | | 527 ms |
| Reflect Pad copies whole rows | runtime | 270 ms |
| Broadcast fills whole rows | runtime | 152 ms |
| GetItem copies one block | runtime | 126 ms |
| Graph rewrites: Linears as 1x1 Convs, q, k, v as three Convs | `model_runner` | 78 ms |
| `normalize_after_matmul` | `model_runner` | 73 ms |
| `CombineQuantizePad`: 8-bit data instead of float32 through the Pads | compiler | 61 ms |

Most of the gain has nothing to do with the NPU. The drop from 527 to 126 ms came from fixing slow CPU kernels, which
worked out an index for every element instead of copying whole rows. Their output is bit-identical to before. Only the
last 126 to 61 ms changed how the model maps onto the hardware.

The starting point is this branch's own runtime before those fixes, not a comparison with Canaan's released runtime.

Accuracy is measured as abs-rel, the mean error in distance relative to the true distance, so 0.112 is about 11%:

| Model | Runs on | abs-rel |
|---|---|---|
| lite-mono-tiny, float | onnxruntime on a PC | 0.107 |
| lite-mono-tiny, 8-bit, rewritten | K230 board | 0.112 |

**This test set is small.** It is 22 frames of one KITTI drive (`2011_09_26_drive_0048`), compared with LiDAR. Abs-rel
varies by about 0.03 from frame to frame, so treat the two numbers as "close", not as a measured gap of 0.005.

Each image's depth is scaled by its median against the LiDAR, because a single camera gives no absolute scale. These
numbers cannot be compared with the table in the Lite-Mono paper. The calibration images come from a different drive,
but adding zoomed crops to them was an idea that came from looking at this test drive, and it has not been confirmed
on another drive. Calibration is also not fully repeatable: one rebuild of the same model differed in 6 bytes of
quantization parameters and gave 0.1127 instead of 0.1124, at the same speed.

The commands and more detail are in the
[Lite-Mono section of model_runner](https://github.com/el3tty/model_runner#lite-mono).

## Building

There are three separate builds. The native build and the board build do not need the compiler build.

| Build | Gives you | Needs |
|---|---|---|
| Compiler | `Nncase.Cli`, to make `.kmodel` files | .NET SDK and the OrtKI packages |
| Native, on the PC | simulator, `nncaseruntime` for the PC, `k230-as`, `k230_isa_test` | C++20 compiler, CMake, Ninja, Python with pybind11 |
| Board runtime | `nncaseruntime` for the board | Linux or WSL, the K230 Linux SDK |

[`build.md`](build.md) is upstream's and does not cover any of this.

### Compiler

The compiler needs `libortki` and `OrtKISharp` 0.0.3, which are not on nuget.org. Download them from the
[OrtKI release](https://github.com/el3tty/OrtKI/releases/tag/v0.0.3-k230) into the folder `NuGet.Config` reads, next to
this repository. On Windows, use Git Bash:

```
mkdir -p ../OrtKI/CSharp/OrtKISharp/bin/OrtKISharp.csproj
curl -L --output-dir ../OrtKI/CSharp/OrtKISharp/bin/OrtKISharp.csproj -O https://github.com/el3tty/OrtKI/releases/download/v0.0.3-k230/libortki.0.0.3.nupkg -O https://github.com/el3tty/OrtKI/releases/download/v0.0.3-k230/OrtKISharp.0.0.3.nupkg
```

Linux needs glibc 2.38 or newer, so Ubuntu 24.04 works and Ubuntu 22.04 does not. Windows needs the Visual C++
Redistributable. Without that folder, restore stops with `NU1301: The local source ... doesn't exist`.

Then build:

```
dotnet build src/Nncase.Cli/Nncase.Cli.csproj -c Release
```

The compiler is `src/Nncase.Cli/bin/Release/net9.0/Nncase.Cli.exe` on Windows. It targets .NET 9, and these steps were
last checked with SDK 11.0.100-preview.3 on Windows.

To build the packages yourself, use the `k230` branch of [el3tty/OrtKI](https://github.com/el3tty/OrtKI). It links a source
build of onnxruntime from a sibling folder, and `libortki` holds both the Windows and the Linux library, so it needs both
builds. There is no up-to-date recipe, so start from `cmake/onnxruntimeConfig.cmake` and
`CSharp/libortki/libortki.nuspec` there. A rebuilt package has a different hash, so run
`dotnet restore src/Nncase.Cli/Nncase.Cli.csproj --force-evaluate` once to update the lock files.

### Native, on the PC

On Windows, run this from a Visual Studio developer shell. Python needs `pybind11` (`pip install pybind11`).

```
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=build-rel/install -S . -B build-rel
ninja -C build-rel install
```

The install folder then holds `nncaseruntime/` (Python module, simulator, `k230_isa_test`, and the tests in `isa_tests/`),
`nncase/` (compiler bindings) and `bin/` (`k230-as`). To check the simulator:

```
cd build-rel/install/nncaseruntime
./k230_isa_test isa_tests
```

It should end with `70 test(s), 70 passed, 0 failed`.

On Linux the same commands work with clang 22, passed with `-DCMAKE_C_COMPILER` and `-DCMAKE_CXX_COMPILER`. The test
program there is the same, and gives the same result.

## Getting the runtime onto the board

To try Lite-Mono on a board without building anything, use the
[board kit](https://github.com/el3tty/nncase/releases/tag/k230-board-kit-lite-mono): the runtime, the kmodel and the
scripts, for an SDK image with Python 3.13 and numpy. The rest of this section builds the runtime yourself.

The board runs Linux from the [K230 Linux SDK](https://github.com/kendryte/k230_linux_sdk). The results above used its
`dev` branch at `856bfe6f` with `k230_canmv_01studio_defconfig`. Other boards need their own defconfig, and that has not
been tried.

**1. Build an SD card image.** Follow the SDK's README: `sudo make toolchain_and_depend`, then
`make k230_canmv_01studio_defconfig`. Before `make`, use `make menuconfig` to check that python3, python-numpy,
python-pybind and the OpenSSH server are enabled. The board needs numpy and sshd, and this build uses the SDK's Python
and pybind11. The image lands in `output/k230_canmv_01studio_defconfig/images/`.

**2. Cross-compile the Python module**, on Linux or WSL. `<sdk>` is the SDK checkout and `<xuantie>` the toolchain the SDK
installs, `/opt/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.0.2`:

```
mkdir build-board && cd build-board
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 -DBUILDING_RUNTIME=1 \
    -DRISCV64_SYSROOT=<sdk>/output/k230_canmv_01studio_defconfig/host/riscv64-buildroot-linux-gnu/sysroot \
    -DCMAKE_C_COMPILER=<xuantie>/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=<xuantie>/bin/riscv64-unknown-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="-mcpu=c908v -mabi=lp64d -mtune=c908 -mrvv-v0p10-compatible -mrvv-auto-vectorize" \
    -DCMAKE_CXX_FLAGS="-mcpu=c908v -mabi=lp64d -mtune=c908 -mrvv-v0p10-compatible -mrvv-auto-vectorize" \
    ..
ninja nncaseruntime_python
```

This assumes `build-board` is inside the repository. The result is
`python/nncaseruntime/native/_nncaseruntime.cpython-313-riscv64-linux-gnu.so`. The runtime is linked in statically.
Add the target `k230_isa_test` to the `ninja` line to also get the board version of the ISA tests.

**3. Copy it to the board** as a Python package folder, from the repository root. `<dir>` is any folder on the board, and
the folder `<dir>/nncaseruntime` has to exist:

```
scp python/nncaseruntime/__init__.py build-board/python/nncaseruntime/native/_nncaseruntime.cpython-313-riscv64-linux-gnu.so root@<board>:<dir>/nncaseruntime/
```

**4. Check it:**

```
ssh root@<board> "PYTHONPATH=<dir> python3 -c 'import numpy, nncaseruntime; print(nncaseruntime.__version__)'"
```

This prints `2.11.0`. Always set `PYTHONPATH`, because an SDK image can have its own `nncaseruntime` in site-packages,
without these changes. Run as root, since the runtime opens `/dev/k230-gnne` and `/dev/mmz`.

To run a model, see [model_runner](https://github.com/el3tty/model_runner). For the operator tests on the board, see
the preparation section of [`tests/ReadMe.txt`](../tests/ReadMe.txt).

## Further reading

* [K230 NPU instruction set and simulator](../src/Native/src/simulator/README.md)
* [`k230_asm`: the assembler and its syntax](../src/Native/src/k230_asm/README.md)
* [K230 ISA tests](../src/Native/src/isa_test/README.md)
* [model_runner](https://github.com/el3tty/model_runner): Python scripts that prepare, compile, run and measure models on
  the K230, including the graph rewrites in the results table
