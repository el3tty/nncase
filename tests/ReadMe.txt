Running the K230 python test suite on the simulator and on the board
====================================================================

run_k230_tests.py runs pytest test files for the k230 target:
  --mode sim    inference on the PC simulator (nncase.simulator.k230)
  --mode board  inference on a K230 board over ssh (compile, evaluator and CPU reference stay on the PC)
  --mode both   sim first, then board; the report also lists the cases that pass on the simulator but fail
                on the board

Every test file runs in its own pytest process, so a compiler crash only loses that file. Results go to
<output>/<mode>/ (logs/, junit/, tests_output/ with --keep-cases); the summary is printed and written to
<output>/report.txt. The exit code is 1 if anything failed.

All commands below are run from this folder (tests/). On Git Bash, prefix board commands with
MSYS_NO_PATHCONV=1, otherwise Git Bash turns /root/... and /tmp/... into Windows paths.


Sample commands
---------------

Simulator, one test file:

  python run_k230_tests.py --mode sim --output k230_results ^
      --install-dir ../build/install ^
      --compiler ../src/Nncase.Compiler/bin/Debug/net9.0/Nncase.Compiler.dll ^
      importer/onnx_/basic/test_relu.py

Board, one test file, only some cases (-k is passed to pytest):

  python run_k230_tests.py --mode board --output k230_results ^
      --install-dir ../build/install ^
      --compiler ../src/Nncase.Compiler/bin/Debug/net9.0/Nncase.Compiler.dll ^
      --board root@<BOARD_IP> --board-runtime /root/test/nncaseruntime --board-dir /tmp/nncase_tests ^
      -k "in_shape_00 and in_shape_10" importer/onnx_/basic/test_where.py

Both, whole folder:

  python run_k230_tests.py --mode both --output k230_results ^
      --install-dir ../build/install ^
      --compiler ../src/Nncase.Compiler/bin/Debug/net9.0/Nncase.Compiler.dll ^
      --board root@<BOARD_IP> --board-runtime /root/test/nncaseruntime --board-dir /tmp/nncase_tests ^
      importer/onnx_/basic

(^ continues a line in cmd; use ` in PowerShell and \ in bash, or put everything on one line.)

Other options (python run_k230_tests.py --help):
  --keep-cases         keep the case folders (test.kmodel, inputs, test_result.txt with cosine similarities)
  --opset / --ir       opset/IR version for onnx.helper models that set none (default 17 / 8);
                       --opset "" keeps the onnx default, which onnxruntime may refuse
  --file-timeout       seconds per test file (default 3600)
  --board-python       python on the board (default python3)
  --board-env          extra environment for the board-side runner, e.g. "LD_LIBRARY_PATH=/root/lib"
  --board-timeout      seconds per inference on the board (default 120)
  -- <args>            anything after -- goes to pytest unchanged


Preparing the PC (host)
-----------------------

1. Build the compiler:            dotnet build (Nncase.Compiler.dll lands in src/Nncase.Compiler/bin/<config>/net9.0)
2. Build and install the native   cmake --build build && cmake --install build
   part (Ninja, host build):      the install prefix (build/install) is --install-dir and must then contain
                                    nncase/         __init__.py and _nncase*.pyd (compiler package)
                                    nncaseruntime/  _nncaseruntime*.pyd, Nncase.Runtime.Native.dll, nethost.dll,
                                                    nncase.simulator.k230.dll
                                  It is needed in every mode: compiling and the evaluator run on the PC.
3. Python packages:               pytest toml onnx onnxruntime onnxsim torch numpy opencv-python pillow
4. For board runs: an ssh client (Windows OpenSSH is fine) and key-based login to the board, see below.


Preparing the board
-------------------

The board runs Linux (k230_linux_sdk image). Nothing runs permanently on it: for every test case the PC
sends the kmodel, the inputs and board/k230_run.py over ssh into <board-dir>/<case>, runs them with the
board's python3 and nncaseruntime, and fetches nncase_result_*.bin back. The case folder is removed after a
successful run and kept after a failure (the error message names it).

1. Image requirements
   - python3 of the version the runtime module is built for (3.13 for the current sdk) with numpy
   - tar, and an ssh server (sshd) running
   - the KPU devices /dev/k230-gnne and /dev/mmz (the runtime opens both); run the tests as root
   - free space in <board-dir> (a few MB per case; /tmp is fine)

2. Build the runtime python module for the board (WSL / Linux, riscv64 cross build):

     mkdir build && cd build
     cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=riscv64 -DBUILDING_RUNTIME=1 \
         -DRISCV64_SYSROOT=<k230_linux_sdk>/output/<defconfig>/host/riscv64-buildroot-linux-gnu/sysroot \
         -DCMAKE_C_COMPILER=<xuantie>/bin/riscv64-unknown-linux-gnu-gcc \
         -DCMAKE_CXX_COMPILER=<xuantie>/bin/riscv64-unknown-linux-gnu-g++ \
         -DCMAKE_C_FLAGS="-mcpu=c908v -mabi=lp64d -mtune=c908 -mrvv-v0p10-compatible -mrvv-auto-vectorize" \
         -DCMAKE_CXX_FLAGS="-mcpu=c908v -mabi=lp64d -mtune=c908 -mrvv-v0p10-compatible -mrvv-auto-vectorize" \
         ..
     ninja nncaseruntime_python

   <xuantie> is the Xuantie-900-gcc-linux toolchain (e.g. /opt/toolchain/Xuantie-900-gcc-linux-6.6.0-glibc-x86_64-V3.0.2).
   Result: build/python/nncaseruntime/native/_nncaseruntime.cpython-313-riscv64-linux-gnu.so
   (the runtime is linked in statically; it only needs the C/C++ system libraries of the image).

3. Install it on the board as a python package folder, e.g. /root/test/nncaseruntime:

     /root/test/nncaseruntime/__init__.py          <- python/nncaseruntime/__init__.py from this repo
     /root/test/nncaseruntime/_nncaseruntime.cpython-313-riscv64-linux-gnu.so

   This folder is --board-runtime (the runner puts its parent folder on PYTHONPATH). Copy, for example:

     scp ../python/nncaseruntime/__init__.py <so file> root@<BOARD_IP>:/root/test/nncaseruntime/

4. Key-based ssh login (the runner uses ssh -o BatchMode=yes, it never answers a password prompt):

     ssh-keygen -t ed25519                         (on the PC, if there is no key yet)
     type %USERPROFILE%\.ssh\id_ed25519.pub | ssh root@<BOARD_IP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

5. Check (on the PC):

     ssh -o BatchMode=yes root@<BOARD_IP> "PYTHONPATH=/root/test python3 -c 'import numpy, nncaseruntime; print(nncaseruntime.__version__)'"

   run_k230_tests.py does the same check before a board run and stops with the error if it fails.

To run one kmodel by hand on the board (e.g. under gdb), use board/k230_run.py from a case folder:

     cd <case folder>
     PYTHONPATH=/root/test python3 k230_run.py test.kmodel input_0.bin:float32:1,3,24,24
