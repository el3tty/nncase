Running the K230 python test suite on the simulator and on the board, or compiling it against goldens
=====================================================================================================

run_k230_tests.py runs pytest test files for the k230 target:
  --mode sim    inference on the PC simulator (nncase.simulator.k230)
  --mode board  inference on a K230 board over ssh (compile, evaluator and CPU reference stay on the PC)
  --mode both   sim first, then board; the report also lists the cases that pass on the simulator but fail
                on the board
  --mode compile  only compile; the assembly of every K230 tile is compared with golden files (see below)

Every test file runs in its own pytest process, so a compiler crash only loses that file. Results go to
<output>/<mode>/ (logs/, junit/, tests_output/<test file>/ with --keep-cases); the summary is printed and written to
<output>/report.txt. The exit code is 1 if anything failed. In board runs the tests listed in BOARD_SKIP
(k230_test_plugin.py) are skipped, e.g. the random_*_like tests, whose reference cannot match the board; the
report lists them with the reason.

All commands below are run from this folder (tests/). <path_to> stands for a folder on the PC, <board_path_to>
for one on the board. On Git Bash, prefix board commands with MSYS_NO_PATHCONV=1, otherwise Git Bash turns
absolute board paths (/...) into Windows paths.


Sample commands (Windows)
-------------------------

Simulator, one test file:

  python run_k230_tests.py --mode sim --output k230_results ^
      --nncase-python-module <path_to>/nncase ^
      --compiler <path_to>/Nncase.Compiler.dll ^
      importer/onnx_/basic/test_relu.py

Board, one test file, only some cases (-k is passed to pytest):

  python run_k230_tests.py --mode board --output k230_results ^
      --nncase-python-module <path_to>/nncase ^
      --compiler <path_to>/Nncase.Compiler.dll ^
      --board root@<BOARD_IP> --board-runtime <board_path_to>/nncaseruntime ^
      -k "in_shape_00 and in_shape_10" importer/onnx_/basic/test_where.py

Both, whole folder:

  python run_k230_tests.py --mode both --output k230_results ^
      --nncase-python-module <path_to>/nncase ^
      --compiler <path_to>/Nncase.Compiler.dll ^
      --board root@<BOARD_IP> --board-runtime <board_path_to>/nncaseruntime ^
      importer/onnx_/basic

(^ continues a line in cmd; use ` in PowerShell and \ in bash, or put everything on one line.)

Other options (python run_k230_tests.py --help):
  --keep-cases         keep the case folders (test.kmodel, inputs, test_result.txt with cosine similarities)
  --opset / --ir       opset/IR version for onnx.helper models that set none (default 17 / 8);
                       --opset "" keeps the onnx default, which onnxruntime may refuse
  --file-timeout       seconds per test file (default 3600)
  --board-python       python on the board (default python3)
  --board-env          extra environment for the board-side runner, e.g. "LD_LIBRARY_PATH=<board_path_to>/lib"
  --board-timeout      seconds per case on the board, incl. starting a session (default 120)
  --board-dir          folder on the board for the session working folders (default: the board's temp
                       folder, usually /tmp; the runner prints the one it uses)
  --jobs               test files run at the same time (sim / compile; default 1 for sim, half the CPUs
                       for compile; the board always runs one file at a time)
  -- <args>            anything after -- goes to pytest unchanged


Compile mode (golden assembly)
------------------------------

  python run_k230_tests.py --mode compile --output k230_results ^
      --nncase-python-module <path_to>/nncase ^
      --compiler <path_to>/Nncase.Compiler.dll ^
      --k230-as <path_to>/k230-as.exe ^
      importer/onnx_/basic

Every case is only compiled (no onnxruntime reference, no evaluator, no inference), so it is fast and the
files run in parallel. The K230 tiles of the kmodel (the compiler's compile.text.asm dumps) are written one
instruction per line in k230-as syntax (src/Native/src/k230_asm) and compared with the golden file of the test
file: golden/k230/<path of the test file>.asm.gz, e.g. golden/k230/importer/onnx_/basic/test_relu.asm.gz
(gzip of plain text, see k230_golden.py; zcat / gzip -dc shows it). The random generators are seeded per test
from its name, so a case compiles the same model with the same calibration data in every run, whatever else
runs (-k selects cases without changing them).

Results per case:
  passed                  same assembly as the golden
  asm changed             the message shows the diff per tile; the whole text is in
                          <output>/compile/tests_output/<test file>/<case>/compile.asm (the case folders of
                          files with failures are kept)
  no golden               the case is not in the golden file (new case)
  missing case            the golden file has a case that the test file does not have any more (full runs only)
  dump / kmodel mismatch  a tile dump does not assemble (k230-as) to the code in the kmodel, i.e. the
                          dump does not show everything the code does; the check runs when k230-as is in
                          PATH (or given with --k230-as, e.g. <install prefix>/bin/k230-as.exe) and is off
                          with --k230-as ""

  --update-golden      write the goldens of the selected cases (with -k the other cases stay), e.g. after a
                       compiler change whose new code is wanted; review the change with git diff (below)
  --golden-dir         another golden folder (default: golden/<target> here)

To see golden changes as text in git diff, once per clone:

  git config diff.gzip.textconv "gzip -dc"

(golden/.gitattributes assigns the diff driver to *.asm.gz.)


Preparing the PC (host)
-----------------------

1. Build the compiler:            dotnet build (Nncase.Compiler.dll lands in src/Nncase.Compiler/bin/<config>/net9.0)
2. Build and install the native   cmake --build <build dir> && cmake --install <build dir>
   part (Ninja, host build):      the install prefix (CMAKE_INSTALL_PREFIX) then contains
                                    nncase/         __init__.py and _nncase*.pyd (compiler package): this folder
                                                    is --nncase-python-module
                                    nncaseruntime/  _nncaseruntime*.pyd, Nncase.Runtime.Native.dll, nethost.dll,
                                                    nncase.simulator.k230.dll (nncase/__init__.py finds them)
                                    bin/            k230-as.exe (assembler, compile mode: on PATH or --k230-as)
                                  The nncase package is needed in every mode: compiling and the evaluator run on
                                  the PC. The runner checks that python imports nncase from that folder.
3. Python packages:               pytest toml onnx onnxruntime onnxsim torch numpy opencv-python pillow
4. For board runs: an ssh client (Windows OpenSSH is fine) and key-based login to the board, see below.


Preparing the board
-------------------

The board runs Linux (k230_linux_sdk image). Nothing is installed on it except nncaseruntime: for every
test file the PC opens one ssh session and starts board/k230_serve.py in it (the script itself is sent over
the session), so python3, numpy and nncaseruntime are loaded once per file. The cases of that file are then
sent over the session (kmodel + inputs in, outputs back), working folder <board-dir>/k230_serve_<id>, which
is removed when the session ends. If a case crashes the board process (e.g. a segfault in the runtime) the
case fails with the board's stderr and the next case starts a new session; after --board-timeout the session
is killed on the board as well, so a hanging model does not keep the KPU busy.
To keep the kmodel and inputs of failing cases, run with --keep-cases (they stay on the PC).

1. Image requirements
   - python3 of the version the runtime module is built for (3.13 for the current sdk) with numpy
   - tar, and an ssh server (sshd) running
   - the KPU devices /dev/k230-gnne and /dev/mmz (the runtime opens both); run the tests as root
   - free space in the temp folder or --board-dir (a few MB per case)

2. Build the runtime python module for the board (WSL / Linux, riscv64 cross build):

     mkdir <build dir> && cd <build dir>
     cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 -DBUILDING_RUNTIME=1 \
         -DRISCV64_SYSROOT=<k230_linux_sdk>/output/<defconfig>/host/riscv64-buildroot-linux-gnu/sysroot \
         -DCMAKE_C_COMPILER=<xuantie>/bin/riscv64-unknown-linux-gnu-gcc \
         -DCMAKE_CXX_COMPILER=<xuantie>/bin/riscv64-unknown-linux-gnu-g++ \
         -DCMAKE_C_FLAGS="-mcpu=c908v -mabi=lp64d -mtune=c908 -mrvv-v0p10-compatible -mrvv-auto-vectorize" \
         -DCMAKE_CXX_FLAGS="-mcpu=c908v -mabi=lp64d -mtune=c908 -mrvv-v0p10-compatible -mrvv-auto-vectorize" \
         <path_to>/nncase
     ninja nncaseruntime_python

   <xuantie> is the folder of the Xuantie-900-gcc-linux toolchain (e.g. Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.0.2).
   Result: <build dir>/python/nncaseruntime/native/_nncaseruntime.cpython-313-riscv64-linux-gnu.so
   (the runtime is linked in statically; it only needs the C/C++ system libraries of the image).

3. Install it on the board as a python package folder, <board_path_to>/nncaseruntime:

     <board_path_to>/nncaseruntime/__init__.py     <- python/nncaseruntime/__init__.py from this repo
     <board_path_to>/nncaseruntime/_nncaseruntime.cpython-313-riscv64-linux-gnu.so

   This folder is --board-runtime (the runner puts its parent folder on PYTHONPATH). Copy, for example:

     scp ../python/nncaseruntime/__init__.py <so file> root@<BOARD_IP>:<board_path_to>/nncaseruntime/

4. Key-based ssh login (the runner uses ssh -o BatchMode=yes, it never answers a password prompt):

     ssh-keygen -t ed25519                         (on the PC, if there is no key yet)
     type %USERPROFILE%\.ssh\id_ed25519.pub | ssh root@<BOARD_IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

5. Check (on the PC):

     ssh -o BatchMode=yes root@<BOARD_IP> "PYTHONPATH=<board_path_to> python3 -c 'import numpy, nncaseruntime; print(nncaseruntime.__version__)'"

   run_k230_tests.py does the same check before a board run and prints the runtime it found (folder, module,
   size, md5). It stops if the import fails or if python takes nncaseruntime from anywhere else than
   --board-runtime (e.g. a package installed in the board's site-packages).

To run one kmodel by hand on the board (e.g. under gdb), copy infer/k230/ptq/test.kmodel and input/*.bin of a case kept
with --keep-cases and board/k230_run.py to the board and run:

     cd <case folder>
     PYTHONPATH=<board_path_to> python3 k230_run.py test.kmodel input_0_0.bin:float32:1,3,24,24
