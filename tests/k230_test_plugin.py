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
"""pytest plugin used by run_k230_tests.py (-p k230_test_plugin).

NNCASE_TEST_DEFAULT_OPSET / NNCASE_TEST_DEFAULT_IR: opset and IR version that onnx.helper.make_model uses when a
    test gives no opset_imports. onnx >= 1.17 stamps its newest (in-development) opset, which onnxruntime refuses.
    Empty NNCASE_TEST_DEFAULT_OPSET keeps the onnx default. The IR version is capped at what onnxruntime reads.
NNCASE_TEST_TARGETS: comma separated targets to run (e.g. 'k230'); every other target in config.toml is disabled.
NNCASE_TEST_ROOT: folder for the case folders (default: tests_output in the current folder).

The tests in RANDOM_SKIP are always skipped: their values come from std::default_random_engine and the std
distributions, which every C++ standard library implements differently.
In board runs (<TARGET>_SSH set) the tests in BOARD_SKIP are skipped with the reason given there.

NNCASE_TEST_COMPILE_ONLY=1 (run_k230_tests.py --mode compile): TestRunner.run only compiles the model (no onnxruntime
    reference, no evaluator, no simulator) and compares the assembly of every K230 tile with the golden file of the
    test file (k230_golden.py) in NNCASE_TEST_GOLDEN_DIR. NNCASE_TEST_UPDATE_GOLDEN=1 writes the golden files instead.
    NNCASE_TEST_K230_AS: k230-as; when set, every tile is assembled and has to give the bytes in the kmodel, i.e. the
    golden text holds everything the machine code does. The random generators are seeded per test (from its name), so
    the models and the calibration data do not depend on what else runs.
"""

import difflib
import os
import random
import subprocess
import tempfile
import zlib

import numpy as np
import pytest
from onnx import helper

_collected = {}         # test file -> names of the selected tests

# tests whose values depend on the C++ standard library (MSVC, libstdc++, libc++ draw other sequences for the same
# seed): neither a golden file nor the onnxruntime reference of one platform matches another, by test function name
_RANDOM = 'random values: std::default_random_engine and the std distributions differ between C++ standard libraries'
RANDOM_SKIP = dict.fromkeys(['test_random_normal', 'test_random_normal_like', 'test_random_uniform',
                             'test_random_uniform_like'], _RANDOM)

# tests whose reference cannot match a board run, by test function name
BOARD_SKIP = {}

_MAX_ORT_IR = 13

_opset = os.getenv('NNCASE_TEST_DEFAULT_OPSET', '17')
_ir = os.getenv('NNCASE_TEST_DEFAULT_IR', '8')
_make_model = helper.make_model


def _make_model_compat(*args, **kwargs):
    if _opset and 'opset_imports' not in kwargs:
        kwargs['opset_imports'] = [helper.make_opsetid('', int(_opset))]
        if _ir:
            kwargs.setdefault('ir_version', int(_ir))
    model = _make_model(*args, **kwargs)
    model.ir_version = min(model.ir_version, _MAX_ORT_IR)
    return model


helper.make_model = _make_model_compat

_targets = [t for t in os.getenv('NNCASE_TEST_TARGETS', '').split(',') if t]
if _targets:
    import test_runner

    _validate_config = test_runner.TestRunner.validte_config

    def _validate_config_selected(self, config):
        for name, target in config['target'].items():
            if name not in _targets:
                target['eval'] = False
                target['infer'] = False
        return _validate_config(self, config)

    test_runner.TestRunner.validte_config = _validate_config_selected

# case folders below this folder instead of ./tests_output (run_k230_tests.py gives every test file its own)
_root = os.getenv('NNCASE_TEST_ROOT')
if _root:
    import test_runner

    _validate_config_root = test_runner.TestRunner.validte_config

    def _validate_config_with_root(self, config):
        config['root'] = _root
        return _validate_config_root(self, config)

    test_runner.TestRunner.validte_config = _validate_config_with_root

_board_run = any(os.getenv('{0}_SSH'.format(t.upper())) for t in _targets)


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = getattr(item, 'originalname', None) or item.name.split('[')[0]
        if name in RANDOM_SKIP:
            item.add_marker(pytest.mark.skip(reason=RANDOM_SKIP[name]))
        elif _board_run and name in BOARD_SKIP:
            item.add_marker(pytest.mark.skip(reason='board run: ' + BOARD_SKIP[name]))


# ---- compile mode ----------------------------------------------------------------------------------------------------

_compile_only = os.getenv('NNCASE_TEST_COMPILE_ONLY') == '1'
_golden_dir = os.getenv('NNCASE_TEST_GOLDEN_DIR', '')
_update_golden = os.getenv('NNCASE_TEST_UPDATE_GOLDEN') == '1'
_k230_as = os.getenv('NNCASE_TEST_K230_AS', '')
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_current = {}           # the running test: item, number of compiled models
_goldens = {}           # test file -> {case: text}
_ran = {}               # test file -> {case: text} of this session


def _seed(name):
    seed = zlib.crc32(name.encode())
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


if _compile_only:
    _seed('collection')


def pytest_collection_finish(session):
    # after -k: only these tests get their golden replaced
    for item in session.items:
        _collected.setdefault(str(item.fspath), set()).add(item.name)


def pytest_runtest_setup(item):
    if _compile_only:
        _current.update(item=item, runs=0)
        _seed(item.name)


def _check_assembles(tile, lines, kmodel):
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, 'tile.s'), os.path.join(d, 'tile.bin')
        with open(src, 'w', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
        p = subprocess.run([_k230_as, '-o', out, src], capture_output=True, text=True)
        if p.returncode != 0:
            return f'{tile}: k230-as failed: {p.stderr.strip()}'
        with open(out, 'rb') as f:
            code = f.read()
    return None if code in kmodel else f'{tile}: the dump does not assemble to the code in the kmodel'


def _split_tiles(text):
    import k230_golden

    tiles = {}
    for line in text.splitlines():
        if line.startswith(k230_golden.TILE):
            tile = tiles.setdefault(line[len(k230_golden.TILE):], [])
        else:
            tile.append(line)
    return tiles


def _compile_only_run(self, model_file):
    import nncase
    import k230_golden

    if not self.inputs:
        self.parse_model(model_file)
    self.generate_all_data()
    model_content = self.read_model_file(model_file)
    text, errors = [], []
    for k_target, v_target in self.cfg['target'].items():
        if not v_target['infer']:
            continue
        for k_mode, v_mode in v_target['mode'].items():
            if not v_mode['enabled']:
                continue
            dump_dir = os.path.join(self.case_dir, 'compile', k_target, k_mode)
            os.makedirs(dump_dir)
            compiler = nncase.Compiler(self.get_compile_options(k_target, dump_dir))
            self.import_model(compiler, model_content, nncase.ImportOptions())
            if k_mode == 'ptq':
                self.set_quant_opt(compiler)
            compiler.compile()
            kmodel = compiler.gencode_tobytes()
            with open(os.path.join(dump_dir, self.cfg['kmodel_name']), 'wb') as f:
                f.write(kmodel)
            for tile, (lines, _) in k230_golden.collect_tiles(dump_dir).items():
                text.append(f'{k230_golden.TILE}{k_target}/{k_mode} {tile}\n' + ''.join(l + '\n' for l in lines))
                if _k230_as:
                    error = _check_assembles(tile, lines, kmodel)
                    if error:
                        errors.append(error)
    text = ''.join(text)

    item = _current['item']
    _current['runs'] += 1
    case = item.name + (f' #{_current["runs"]}' if _current['runs'] > 1 else '')
    test_file = str(item.fspath)
    _ran.setdefault(test_file, {})[case] = text
    with open(os.path.join(self.case_dir, 'compile.asm'), 'w', newline='\n') as f:
        f.write(text)
    assert not errors, 'dump / kmodel mismatch: ' + '; '.join(errors)
    if _update_golden:
        return
    if test_file not in _goldens:
        _goldens[test_file] = k230_golden.read_golden(
            k230_golden.golden_path(_golden_dir, test_file, _tests_dir))
    golden = _goldens[test_file].get(case)
    assert golden is not None, f'no golden: {case} is not in the golden file'
    if golden != text:
        golden_tiles, tiles = _split_tiles(golden), _split_tiles(text)
        diff = []
        for tile in list(golden_tiles) + [t for t in tiles if t not in golden_tiles]:
            diff += difflib.unified_diff(golden_tiles.get(tile, []), tiles.get(tile, []), f'golden {tile}',
                                         f'actual {tile}', n=1, lineterm='')
        shown = '\n'.join(diff[:40]) + ('\n...' if len(diff) > 40 else '')
        assert False, f'asm changed ({len(diff)} diff lines, full text: {os.path.join(self.case_dir, 'compile.asm')})\n{shown}'


if _compile_only:
    import test_runner

    test_runner.TestRunner.run = _compile_only_run


def pytest_sessionfinish(session, exitstatus):
    if not (_compile_only and _update_golden):
        return
    import k230_golden

    # the selected tests are replaced by what they compiled now (a test that failed loses its golden), the others
    # (deselected with -k) stay
    for test_file, selected in _collected.items():
        path = k230_golden.golden_path(_golden_dir, test_file, _tests_dir)
        cases = {case: text for case, text in k230_golden.read_golden(path).items()
                 if case.split(' #')[0] not in selected}
        cases.update(_ran.get(test_file, {}))
        if cases:
            k230_golden.write_golden(path, cases)
        elif path.exists():
            path.unlink()
