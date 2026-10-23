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
"""Board-side runner for the SSH inference path of the test harness (see Inference.run_ssh).

usage: python3 k230_run.py <kmodel> <input.bin>:<dtype>:<d0,d1,...> ...

Runs the kmodel once with nncaseruntime and writes nncase_result_<i>.bin plus outputs.json
(dtype and shape of every output) into the current directory.
"""

import json
import sys
import time

import numpy as np
import nncaseruntime


def parse_input(arg):
    file, dtype, shape = arg.split(':')
    dims = [int(d) for d in shape.split(',')] if shape != '' else []
    return np.fromfile(file, dtype=dtype).reshape(dims)


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    interp = nncaseruntime.Interpreter()
    with open(argv[1], 'rb') as f:
        interp.load_model(f.read())

    for idx, arg in enumerate(argv[2:]):
        interp.set_input_tensor(idx, nncaseruntime.RuntimeTensor.from_numpy(parse_input(arg)))

    t1 = time.perf_counter()
    interp.run()
    t = (time.perf_counter() - t1) * 1000

    outputs = []
    for i in range(interp.outputs_size):
        output = interp.get_output_tensor(i).to_numpy()
        output.tofile(f'nncase_result_{i}.bin')
        outputs.append({'dtype': output.dtype.name, 'shape': list(output.shape)})

    with open('outputs.json', 'w') as f:
        json.dump({'outputs': outputs, 'time_ms': t}, f)
    print(f'interp run: {t:.3f} ms')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
