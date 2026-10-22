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
#
"""nncase runtime."""
import os

if os.name == "nt":
    # the runtime loads nncase.simulator.<kind>.dll at model load time; Python >= 3.8 does not search this folder
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))

try:
    # host install: the extension module is next to this file (<prefix>/nncaseruntime)
    from ._nncaseruntime import TensorDesc, RuntimeTensor, Interpreter, __version__
except ModuleNotFoundError:
    # runtime (board) install: the extension module is installed into lib/, on sys.path
    from _nncaseruntime import TensorDesc, RuntimeTensor, Interpreter, __version__
