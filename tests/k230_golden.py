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
"""Golden assembly of the compile mode (run_k230_tests.py --mode compile).

A golden file holds the K230 code of every case of one test file, gzip compressed (mtime 0, so equal text gives
equal bytes):

    # case: test_relu[in_shape0]
    # tile: k230/ptq TileAct1_0
    l2_load_conf 0, 0, ...
    ...

Every instruction is one line in the syntax of k230-as (src/Native/src/k230_asm): the instruction blocks of the
compiler dump (compile.text.asm) without 'pc offset', 'opcode' and the 'binding_*' lines, the remaining values in
order. Tile names are renumbered per case (the compiler numbers them per process, so they depend on what ran before).
The case and tile lines are '#' comments, so the text of a tile (or of a whole file) can be given to k230-as as it is.
"""

import gzip
import io
import os
import re
from pathlib import Path

# comment lines that start a case / a tile
CASE = '# case: '
TILE = '# tile: '


def compact_dump(text):
    """compile.text.asm -> one instruction per line"""
    out = []
    for block in text.split('pc offset: ')[1:]:
        lines = block.split('\n')
        name = lines[1].split()[1]
        ops = [l.split(':', 1)[1].strip() for l in lines[2:]
               if ':' in l and not l.strip().startswith(('opcode', 'binding_'))]
        out.append(name + (' ' + ', '.join(ops) if ops else ''))
    return out


def collect_tiles(dump_dir):
    """{normalized tile name: [instruction lines]} of every compile.text.asm below dump_dir"""
    found = []
    for path in Path(dump_dir).rglob('compile.text.asm'):
        m = re.fullmatch(r'(.*)_(\d+)', path.parent.name)
        kind, number = (m.group(1), int(m.group(2))) if m else (path.parent.name, -1)
        found.append((number, kind, path))
    tiles = {}
    index = {}
    for number, kind, path in sorted(found):
        i = index.get(kind, 0)
        index[kind] = i + 1
        tiles[f'{kind}_{i}'] = (compact_dump(path.read_text()), path)
    return tiles


def golden_path(golden_dir, test_file, tests_dir):
    rel = Path(test_file).resolve().relative_to(Path(tests_dir).resolve()).with_suffix('.asm.gz')
    return Path(golden_dir) / rel


def read_golden(path):
    """{case: text}; empty when the file does not exist"""
    cases = {}
    if not os.path.exists(path):
        return cases
    with gzip.open(path, 'rt', newline='\n') as f:
        text = f.read()
    name = None
    for line in text.splitlines(keepends=True):
        if line.startswith(CASE):
            name = line[len(CASE):].rstrip('\n')
            cases[name] = ''
        elif name is not None:
            cases[name] += line
    return cases


def write_golden(path, cases):
    text = ''.join(f'{CASE}{name}\n{cases[name]}' for name in sorted(cases))
    buf = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0, compresslevel=9) as f:
        f.write(text.encode())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(buf.getvalue())
