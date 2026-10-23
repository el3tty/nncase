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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import os
import json
import numpy as np
from pathlib import Path


def dump_bin_file(file: str, ndarray: np.array):
    ndarray.tofile(file)


def dump_txt_file(save_path, ndarray: np.array, bit_16_represent=False):
    if bit_16_represent:
        np.save(save_path, _cast_bfloat16_then_float32(ndarray))
    else:
        if ndarray.dtype == np.uint8:
            fmt = '%u'
        elif ndarray.dtype == np.int8:
            fmt = '%d'
        else:
            fmt = '%f'
        np.savetxt(save_path, ndarray.flatten(), fmt=fmt, header=str(ndarray.shape))

    print("----> %s" % save_path)


def _cast_bfloat16_then_float32(values: np.array):
    shape = values.shape
    values = values.reshape([-1])
    for i, value in enumerate(values):
        value = float(value)
        value = 1
        packed = struct.pack('!f', value)
        integers = [c for c in packed][:2] + [0, 0]
        value = struct.unpack('!f', bytes(integers))[0]
        values[i] = value


def dump_dict_to_json(dict, json_file):
    json_list = []
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            json_list = json.load(f)

    json_list.append(dict)
    with open(json_file, 'w') as f:
        json.dump(json_list, f)


def in_ci():
    return os.getenv('CI', 'False').lower() == 'true'


def kpu_targets():
    return os.getenv('KPU_TARGETS', "").split(',')


def nuc_ip():
    return os.getenv('NUC_PROXY_IP')


def nuc_port():
    return os.getenv('NUC_PROXY_PORT')


def test_executable(target: str):
    return os.getenv('TEST_EXECUTABLE_{0}'.format(target.upper()))


def ssh_host(target: str):
    # e.g. K230_SSH=root@192.168.1.100, the board must accept key auth
    return os.getenv('{0}_SSH'.format(target.upper()))


def _required_env(name: str):
    value = os.getenv(name)
    if not value:
        raise Exception(f'{name} is not set')
    return value


def ssh_runtime(target: str):
    # nncaseruntime package dir on the board (__init__.py + _nncaseruntime*.so)
    return _required_env('{0}_SSH_RUNTIME'.format(target.upper()))


def ssh_env(target: str):
    # extra environment for the board-side runner, e.g. 'LD_LIBRARY_PATH=/root/lib'
    return os.getenv('{0}_SSH_ENV'.format(target.upper()), '')


def ssh_dir(target: str):
    # scratch dir on the board, one subfolder per case
    return _required_env('{0}_SSH_DIR'.format(target.upper()))


def ssh_python(target: str):
    return os.getenv('{0}_SSH_PYTHON'.format(target.upper()), 'python3')


def ssh_timeout(target: str):
    return int(os.getenv('{0}_SSH_TIMEOUT'.format(target.upper()), '300'))


def infer_report_file(default: str):
    return os.getenv('INFER_REPORT_FILE', default)


def search_file(dir: str, file: str):
    p = ''
    for path in Path(dir).rglob(file):
        p = path
        break
    return p
