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

In board runs (<TARGET>_SSH set) the tests in BOARD_SKIP are skipped with the reason given there.
"""

import os

import pytest
from onnx import helper

# tests whose reference cannot match a board run, by test function name
BOARD_SKIP = {
    'test_random_normal_like': 'random values: the board runtime (libstdc++) draws another sequence than the '
                               'onnxruntime reference (MSVC) for the same seed',
    'test_random_uniform_like': 'random values: the board runtime (libstdc++) draws another sequence than the '
                                'onnxruntime reference (MSVC) for the same seed',
}

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

_board_run = any(os.getenv('{0}_SSH'.format(t.upper())) for t in _targets)


def pytest_collection_modifyitems(config, items):
    if not _board_run:
        return
    for item in items:
        reason = BOARD_SKIP.get(getattr(item, 'originalname', None) or item.name.split('[')[0])
        if reason:
            item.add_marker(pytest.mark.skip(reason='board run: ' + reason))
