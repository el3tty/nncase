PYTHON=$HOME/kendryte/.venv/bin/python

$PYTHON run_k230_tests.py --mode compile \
   --output tests_output \
   --nncase-python-module ../build-linux/install/nncase \
   --compiler ../src/Nncase.Compiler/bin/Release/net9.0/Nncase.Compiler.dll \
   --k230-as ../build-linux/install/bin/k230-as importer/onnx_/basic
