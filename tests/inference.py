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

from typing import List, Dict, Union, Tuple
import os
import nncase
import numpy as np
import test_utils
import preprocess_utils
import socket
import struct
import json
import atexit
import collections
import shlex
import subprocess
import threading
from test_utils import *
import time
from html import escape


def data_shape_list_string(data):
    return '\n'.join(map(lambda d: ' '.join(map(lambda x: str(x), d['model_shape'])), data))


class BoardSession:
    # One ssh connection with tests/board/k230_serve.py running on the board, reused for all cases of this
    # pytest process. A session that died (crash, timeout) is dropped and the next case starts a new one.
    _sessions = {}
    _count = 0

    @classmethod
    def get(cls, target):
        session = cls._sessions.get(target)
        if session is None or session.proc.poll() is not None:
            session = cls._sessions[target] = cls(target)
        return session

    @classmethod
    def drop(cls, target):
        session = cls._sessions.pop(target, None)
        if session is not None:
            session.close()

    @classmethod
    def close_all(cls):
        for target in list(cls._sessions):
            cls.drop(target)

    def __init__(self, target):
        BoardSession._count += 1
        self.host = test_utils.ssh_host(target)
        self.timeout = test_utils.ssh_timeout(target)
        runtime = test_utils.ssh_runtime(target)
        # unique per session, also on the remote command line so a timed out session can be killed on the board
        self.token = 'k230_serve_{0}_{1}_{2}'.format(int(time.time()), os.getpid(), BoardSession._count)
        work_dir = self.work_dir = shlex.quote('{0}/{1}'.format(test_utils.ssh_dir(target), self.token))
        # the server script comes over stdin (length line + source) and then keeps reading the cases from it
        bootstrap = ('import sys; n = int(sys.stdin.buffer.readline()); '
                     'exec(compile(sys.stdin.buffer.read(n), "k230_serve.py", "exec"))  # ' + self.token)
        run = ' '.join([f'PYTHONPATH={shlex.quote(os.path.dirname(runtime))}', test_utils.ssh_env(target),
                        test_utils.ssh_python(target), '-u', '-c', shlex.quote(bootstrap)])
        remote_cmd = f'rm -rf {work_dir}; mkdir -p {work_dir} && cd {work_dir} && {run}; cd /; rm -rf {work_dir}'
        self.proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', self.host, remote_cmd], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.log = collections.deque(maxlen=200)
        threading.Thread(target=self._read_log, daemon=True).start()

        with open(os.path.join(os.path.dirname(__file__), 'board', 'k230_serve.py'), 'rb') as f:
            script = f.read()
        hello = self._exchange(str(len(script)).encode() + b'\n' + script)
        if not hello.get('ready'):
            raise Exception(f'ssh infer failed on {self.host}: unexpected server start {hello}')

    def _read_log(self):
        for line in self.proc.stderr:
            self.log.append(line.decode(errors='replace').rstrip())

    def _read_exact(self, size):
        data = self.proc.stdout.read(size)
        if data is None or len(data) != size:
            raise EOFError
        return data

    def _exchange(self, request):
        # writes the request and reads the JSON header of the answer; a timeout kills the session
        timer = threading.Timer(self.timeout, self.proc.kill)
        timer.start()
        try:
            self.proc.stdin.write(request)
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise EOFError
            header = json.loads(line)
            header['_payloads'] = [self._read_exact(o['size']) for o in header.get('outputs', [])]
            return header
        except (EOFError, OSError, ValueError):
            timed_out = not timer.is_alive()
            self.proc.kill()
            self.proc.wait()
            time.sleep(0.2)  # let the log thread pick up the last stderr lines
            log = '\n'.join(self.log)
            if timed_out:
                # the board process may still hang (e.g. waiting for the KPU) and hold the device
                # (dropping the connection also ends the remote shell before it removes the work dir; rm goes first
                # because pkill -f also matches this shell, whose command line contains the token)
                subprocess.run(['ssh', '-o', 'BatchMode=yes', self.host,
                                f'rm -rf {self.work_dir}; pkill -f {self.token}'], capture_output=True, timeout=60)
                raise Exception(f'ssh infer timeout on {self.host} after {self.timeout} s:\n{log}')
            raise Exception(f'ssh infer failed on {self.host} (board process ended, exit {self.proc.returncode}):'
                            f'\n{log}')
        finally:
            timer.cancel()

    def infer(self, kmodel, inputs):
        header = {'kmodel': len(kmodel),
                  'inputs': [{'dtype': d.dtype.name, 'shape': list(d.shape), 'size': d.nbytes} for d in inputs]}
        request = (json.dumps(header) + '\n').encode() + kmodel + b''.join(
            np.ascontiguousarray(d).tobytes() for d in inputs)
        answer = self._exchange(request)
        if not answer['ok']:
            raise Exception(f'ssh infer failed on {self.host}:\n{answer["error"]}')
        outputs = [np.frombuffer(p, dtype=o['dtype']).reshape(o['shape'])
                   for p, o in zip(answer['_payloads'], answer['outputs'])]
        return outputs, answer['time_ms']

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


atexit.register(BoardSession.close_all)


class Inference:
    def run_inference(self, compiler, target, ptq_enabled, infer_dir):
        in_ci = test_utils.in_ci()
        kpu_targets = test_utils.kpu_targets()
        nuc_ip = test_utils.nuc_ip()
        nuc_port = test_utils.nuc_port()
        test_executable = test_utils.test_executable(target)
        running_on_evb = target in kpu_targets and nuc_ip is not None and nuc_port is not None and test_executable is not None and len(
            self.inputs) > 0 and len(self.outputs) > 0
        running_on_ssh = target in kpu_targets and test_utils.ssh_host(target) is not None and len(
            self.inputs) > 0 and len(self.outputs) > 0

        if self.cfg['infer_report_opt']['enabled']:
            self.infer_report_dict['priority'] = self.cfg['infer_report_opt']['priority']
            self.infer_report_dict['kind'] = self.cfg['infer_report_opt']['kind']
            self.infer_report_dict['model'] = self.cfg['infer_report_opt']['model_name']
            self.infer_report_dict['shape'] = ',<br/>'.join(
                map(lambda d: '[' + ','.join(map(lambda x: str(x), d['model_shape'])) + ']', self.inputs))
        if ptq_enabled:
            self.set_quant_opt(compiler)

            if self.cfg['infer_report_opt']['enabled']:
                case = os.path.basename(self.case_dir)
                self.infer_report_dict['if_quant_type'] = self.cfg['ptq_opt']['quant_type']
                self.infer_report_dict['w_quant_type'] = self.cfg['ptq_opt']['w_quant_type']

        compiler.compile()
        kmodel = compiler.gencode_tobytes()
        os.makedirs(infer_dir, exist_ok=True)
        if self.dynamic:
            self.dump_kmodel_desc(os.path.join(infer_dir, self.cfg['desc_name']))
        if not in_ci:
            with open(os.path.join(infer_dir, self.cfg['kmodel_name']), 'wb') as f:
                f.write(kmodel)

        compile_opt = self.cfg['compile_opt']
        if running_on_ssh:
            outputs = self.run_ssh(target, kmodel, compile_opt, infer_dir)
        elif running_on_evb:
            outputs = self.run_evb(target, kmodel, compile_opt, infer_dir)
        else:
            sim = nncase.Simulator()
            sim.load_model(kmodel)
            self.set_infer_input(sim, compile_opt)

            if self.cfg['infer_report_opt']['enabled']:
                t1 = time.perf_counter()

            sim.run()

            if self.cfg['infer_report_opt']['enabled']:
                t = (time.perf_counter() - t1) * 1000
                self.infer_report_dict['actual_fps'] = str(round(1000 / t, 3))

            outputs = self.dump_infer_output(sim, compile_opt, infer_dir)
        return outputs

    def set_infer_input(self, sim, compile_opt):
        for idx, value in enumerate(self.inputs):
            data = self.transform_input(
                value['data'], compile_opt['input_type'], "infer")[0]
            dtype = compile_opt['input_type']
            if compile_opt['preprocess'] and dtype != 'float32':
                if not test_utils.in_ci():
                    dump_bin_file(os.path.join(self.case_dir, f'input_{idx}_{dtype}.bin'), data)
                    dump_txt_file(os.path.join(self.case_dir, f'input_{idx}_{dtype}.txt'), data)

            sim.set_input_tensor(idx, nncase.RuntimeTensor.from_numpy(data))

    def dump_kmodel_desc(self, file):
        input_shapes = data_shape_list_string(self.inputs)
        output_shapes = data_shape_list_string(self.outputs)
        s = f"{len(self.inputs)} {len(self.outputs)}\n{input_shapes}\n{output_shapes}\n"
        with open(file, "w+") as f:
            f.write(s)

    def dump_infer_output(self, sim, compile_opt, infer_dir):
        return self.postprocess_infer_output(
            [sim.get_output_tensor(i).to_numpy() for i in range(sim.outputs_size)], compile_opt, infer_dir)

    def postprocess_infer_output(self, raw_outputs, compile_opt, infer_dir):
        outputs = []
        for i, output in enumerate(raw_outputs):
            if compile_opt['preprocess']:
                if(compile_opt['output_layout'] == 'NHWC' and self.model_type in ['caffe', 'onnx']):
                    output = np.transpose(output, [0, 3, 1, 2])
                elif (compile_opt['output_layout'] == 'NCHW' and self.model_type in ['tflite']):
                    output = np.transpose(output, [0, 2, 3, 1])
                elif compile_opt['output_layout'] not in ["NCHW", "NHWC"] and compile_opt['output_layout'] != "":
                    tmp_perm = [int(idx) for idx in compile_opt['output_layout'].split(",")]
                    output = np.transpose(
                        output, preprocess_utils.get_source_transpose_index(tmp_perm))
            outputs.append(output)
            if not test_utils.in_ci():
                dump_bin_file(os.path.join(infer_dir, f'nncase_result_{i}.bin'), output)
                dump_txt_file(os.path.join(infer_dir, f'nncase_result_{i}.txt'), output)
        return outputs

    def run_ssh(self, target, kmodel, compile_opt, infer_dir):
        inputs = [self.transform_input(value['data'], compile_opt['input_type'], "infer")[0]
                  for value in self.inputs]
        session = BoardSession.get(target)
        try:
            raw_outputs, time_ms = session.infer(kmodel, inputs)
        except Exception:
            if session.proc.poll() is not None:
                # the board process is gone (crash, timeout): the next case starts a fresh one
                BoardSession.drop(target)
            raise
        if self.cfg['infer_report_opt']['enabled']:
            self.infer_report_dict['actual_fps'] = str(round(1000 / time_ms, 3))
        return self.postprocess_infer_output(raw_outputs, compile_opt, infer_dir)

    def send_msg(self, sock, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        sock.sendall(msg)

    def recv_msg(self, sock):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        return self.recvall(sock, msglen)

    def recvall(self, sock, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def run_evb(self, target, kmodel, compile_opt, infer_dir):
        ip = test_utils.nuc_ip()
        port = test_utils.nuc_port()
        test_executable = test_utils.test_executable(target)

        # connect server
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, int(port)))

        # send target
        dummy = self.recv_msg(client_socket)
        target_dict = {}
        target_dict['target'] = target
        self.send_msg(client_socket, json.dumps(target_dict).encode())

        # send header
        dummy = self.recv_msg(client_socket)
        header_dict = {}
        header_dict['case'] = os.path.basename(self.case_dir)
        header_dict['app'] = 1
        header_dict['kmodel'] = 1
        header_dict['inputs'] = len(self.inputs)
        header_dict['description'] = 1 if self.dynamic else 0
        header_dict['outputs'] = len(self.outputs)
        header_dict['cfg_cmds'] = self.config_cmds()
        self.send_msg(client_socket, json.dumps(header_dict).encode())

        # send app
        dummy = self.recv_msg(client_socket)
        file_dict = {}
        file_dict['file_name'] = os.path.basename(test_executable)
        file_dict['file_size'] = os.path.getsize(test_executable)
        self.send_msg(client_socket, json.dumps(file_dict).encode())
        dummy = self.recv_msg(client_socket)
        with open(test_executable, 'rb') as f:
            client_socket.sendall(f.read())

        # send kmodel
        dummy = self.recv_msg(client_socket)
        file_dict['file_name'] = self.cfg['kmodel_name']
        file_dict['file_size'] = len(kmodel)
        self.send_msg(client_socket, json.dumps(file_dict).encode())
        dummy = self.recv_msg(client_socket)
        client_socket.sendall(kmodel)

        # send inputs
        for idx, value in enumerate(self.inputs):
            dummy = self.recv_msg(client_socket)
            data = self.transform_input(
                value['data'], compile_opt['input_type'], "infer")[0]
            file_dict['file_name'] = f'input_{idx}.bin'
            file_dict['file_size'] = data.size * data.itemsize
            self.send_msg(client_socket, json.dumps(file_dict).encode())
            dummy = self.recv_msg(client_socket)
            client_socket.sendall(data.tobytes())

        # send kmodel.desc
        if self.dynamic:
            dummy = self.recv_msg(client_socket)
            desc_file = os.path.join(infer_dir, self.cfg['desc_name'])
            file_dict['file_name'] = os.path.basename(desc_file)
            file_dict['file_size'] = os.path.getsize(desc_file)
            self.send_msg(client_socket, json.dumps(file_dict).encode())
            dummy = self.recv_msg(client_socket)
            with open(desc_file, 'rb') as f:
                client_socket.sendall(f.read())

        # get infer result
        outputs = []
        header_dict = {}
        ret = self.recv_msg(client_socket)
        header_dict = json.loads(ret.decode())
        ret = header_dict['msg']
        if header_dict['type'].find('finish') != -1:
            if self.cfg['infer_report_opt']['enabled']:
                self.stat_target(infer_dir, ret)

            self.send_msg(client_socket, f"pls send outputs".encode())

            # recv outputs
            for i in range(len(self.outputs)):
                header = self.recv_msg(client_socket)
                file_dict = json.loads(header.decode())
                file_size = file_dict['file_size']
                self.send_msg(client_socket, f"pls send file".encode())

                buffer = bytearray(file_size)
                buffer = self.recvall(client_socket, file_size)

                output = np.frombuffer(buffer, dtype=self.outputs[i]['dtype'])
                outputs.append(output)

                if not test_utils.in_ci():
                    dump_bin_file(os.path.join(infer_dir, f'nncase_result_{i}.bin'), output)
                    dump_txt_file(os.path.join(infer_dir, f'nncase_result_{i}.txt'), output)

            client_socket.close()
        else:
            client_socket.close()

            if self.cfg['infer_report_opt']['enabled']:
                self.infer_report_dict['result'] = 'Fail'
                self.infer_report_dict['remark'] = escape(ret)
                prefix, suffix = os.path.splitext(self.infer_report_file)
                json_file = f'{prefix}_{os.path.basename(self.case_dir)}{suffix}'
                dump_dict_to_json(self.infer_report_dict, json_file)

            raise Exception(ret)

        return outputs
