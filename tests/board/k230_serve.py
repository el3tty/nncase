# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Board-side inference server for the ssh path of the test harness (see Inference.run_ssh).

The PC starts it once per test file over ssh and sends it the cases on stdin, so python, numpy and nncaseruntime
are loaded once instead of once per case. Every message is one JSON line followed by raw bytes:

  -> {"kmodel": <bytes>, "inputs": [{"dtype": "float32", "shape": [1, 3], "size": <bytes>}, ...]}
     kmodel bytes, then the bytes of every input
  <- {"ok": true, "time_ms": <run time>, "outputs": [{"dtype": ..., "shape": [...], "size": <bytes>}, ...]}
     the bytes of every output
  <- {"ok": false, "error": "<traceback>"}

The first message is {"ready": true, "version": ...}. Anything else written to stdout (by python or the native
runtime) is redirected to stderr, which the PC keeps for error messages.
"""

import json
import os
import sys
import time
import traceback

import numpy as np
import nncaseruntime

_proto = os.fdopen(os.dup(1), 'wb', buffering=0)
os.dup2(2, 1)
sys.stdout = sys.stderr
_stdin = sys.stdin.buffer


def read_exact(size):
    data = bytearray()
    while len(data) < size:
        chunk = _stdin.read(size - len(data))
        if not chunk:
            raise EOFError('stdin closed')
        data += chunk
    return data


def send(header, payloads=()):
    _proto.write((json.dumps(header) + '\n').encode())
    for payload in payloads:
        _proto.write(payload)


def run_case(kmodel, inputs):
    interp = nncaseruntime.Interpreter()
    interp.load_model(bytes(kmodel))
    for idx, data in enumerate(inputs):
        interp.set_input_tensor(idx, nncaseruntime.RuntimeTensor.from_numpy(data))
    t1 = time.perf_counter()
    interp.run()
    t = (time.perf_counter() - t1) * 1000
    outputs = [np.ascontiguousarray(interp.get_output_tensor(i).to_numpy()) for i in range(interp.outputs_size)]
    return outputs, t


def main():
    send({'ready': True, 'version': getattr(nncaseruntime, '__version__', '?')})
    while True:
        line = _stdin.readline()
        if not line:
            return 0
        request = json.loads(line)
        kmodel = read_exact(request['kmodel'])
        inputs = [np.frombuffer(read_exact(i['size']), dtype=i['dtype']).reshape(i['shape'])
                  for i in request['inputs']]
        try:
            outputs, t = run_case(kmodel, inputs)
        except Exception:
            send({'ok': False, 'error': traceback.format_exc()})
            continue
        send({'ok': True, 'time_ms': t,
              'outputs': [{'dtype': o.dtype.name, 'shape': list(o.shape), 'size': o.nbytes} for o in outputs]},
             [o.tobytes() for o in outputs])


if __name__ == '__main__':
    sys.exit(main())
