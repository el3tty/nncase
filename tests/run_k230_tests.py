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
"""Runs the python test suite for k230 on the PC simulator, on the board over ssh, or both; or only compiles it.

Every test file runs in its own pytest process (a compiler crash only loses that file) and writes a junit report.
At the end a per-file table is printed; with --mode both, also the cases that pass on the simulator but fail on
the board.

--mode compile only compiles every case and compares the assembly of its K230 tiles with the golden files
(--golden-dir, one gzip file per test file, see k230_golden.py); --update-golden writes them. Nothing runs on the
simulator or the board, so the test files can run in parallel (--jobs).

examples:
  (from tests/, see ReadMe.txt)
  python run_k230_tests.py --mode sim --output k230_results \\
      --nncase-python-module <path_to>/nncase \\
      --compiler <path_to>/Nncase.Compiler.dll \\
      importer/onnx_/basic/test_relu.py
  python run_k230_tests.py --mode both --output k230_results \\
      --nncase-python-module <path_to>/nncase \\
      --compiler <path_to>/Nncase.Compiler.dll \\
      --board root@<BOARD_IP> --board-runtime <board_path_to>/nncaseruntime \\
      importer/onnx_/basic
  python run_k230_tests.py --mode compile --output k230_results \\
      --nncase-python-module <path_to>/nncase \\
      --compiler <path_to>/Nncase.Compiler.dll \\
      importer/onnx_/basic
"""

import argparse
import collections
import concurrent.futures
import os
import posixpath
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import k230_golden

TESTS_DIR = Path(__file__).resolve().parent
REPO_DIR = TESTS_DIR.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+', help='test files or folders')
    p.add_argument('--mode', choices=['sim', 'board', 'both', 'compile'], default='sim')
    p.add_argument('--target', default='k230', help='target to test, other targets are disabled (default: k230)')
    p.add_argument('-k', dest='keyword', help='pytest -k expression')
    p.add_argument('--output', required=True, help='results folder, one subfolder per mode')
    p.add_argument('--keep-cases', action='store_true',
                   help='keep tests_output case folders (inputs, kmodels, results) instead of deleting them')
    p.add_argument('--file-timeout', type=int, default=3600, help='seconds per test file (default: 3600)')
    p.add_argument('--jobs', type=int,
                   help='test files run at the same time in --mode sim / compile (default: 1 for sim, half the CPUs '
                        'for compile); the board runs one file at a time')

    golden = p.add_argument_group('compile mode')
    golden.add_argument('--golden-dir', help='golden files (default: tests/golden/<target>)')
    golden.add_argument('--update-golden', action='store_true',
                        help='write the golden files of the selected cases instead of comparing')
    golden.add_argument('--k230-as',
                        help='k230-as that checks that every tile dump assembles to the code in the kmodel '
                             '(default: k230-as on PATH if there is one; "" disables the check)')

    host = p.add_argument_group('host')
    host.add_argument('--nncase-python-module', required=True,
                      help='folder of the nncase python package (the one with __init__.py and _nncase), e.g. '
                           '<install prefix>/nncase; it finds the runtime and the simulator itself; needed in every '
                           'mode')
    host.add_argument('--compiler', required=True, help='Nncase.Compiler.dll')
    host.add_argument('--opset', default='17',
                      help='default opset for onnx.helper models without one, "" keeps the onnx default (default: 17)')
    host.add_argument('--ir', default='8', help='IR version that goes with --opset (default: 8)')

    board = p.add_argument_group('board (required for --mode board/both)')
    board.add_argument('--board', help='ssh destination, e.g. root@192.168.10.2 (key auth, no password prompt)')
    board.add_argument('--board-runtime', help='nncaseruntime package folder on the board')
    board.add_argument('--board-dir',
                       help='folder on the board for the working folders of the ssh sessions (each session creates '
                            'its own k230_serve_* folder and removes it); default: the temp folder of the board '
                            '(python tempfile.gettempdir(), usually /tmp)')
    board.add_argument('--board-python', default='python3', help='python on the board (default: python3)')
    board.add_argument('--board-env', default='', help="extra environment on the board, e.g. 'LD_LIBRARY_PATH=/x'")
    board.add_argument('--board-timeout', type=int, default=120, help='seconds per inference (default: 120)')

    p.add_argument('pytest_args', nargs=argparse.REMAINDER, help='after "--": extra pytest arguments')
    args = p.parse_args()
    if args.mode in ('board', 'both'):
        missing = [f'--{n.replace("_", "-")}' for n in ('board', 'board_runtime') if not getattr(args, n)]
        if missing:
            p.error(f'{", ".join(missing)} required for --mode {args.mode}')
    if args.pytest_args and args.pytest_args[0] == '--':
        args.pytest_args = args.pytest_args[1:]
    if args.update_golden and args.mode != 'compile':
        p.error('--update-golden needs --mode compile')
    if args.golden_dir is None:
        args.golden_dir = str(TESTS_DIR / 'golden' / args.target)
    if args.k230_as is None:
        args.k230_as = shutil.which('k230-as') or ''
    if args.jobs is None:
        args.jobs = max(1, (os.cpu_count() or 2) // 2) if args.mode == 'compile' else 1
    return args


def collect_files(paths):
    files = []
    for path in [Path(p).resolve() for p in paths]:
        if path.is_dir():
            files += sorted(path.rglob('test_*.py'))
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f'no such test file or folder: {path}')
    return files


def make_env(args, mode):
    env = dict(os.environ)
    # the given nncase package, not the sources in python/nncase (they have no _nncase next to them)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(Path(args.nncase_python_module).resolve().parent), str(TESTS_DIR)] +
        ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env['NNCASE_COMPILER'] = str(Path(args.compiler).resolve())
    env['NNCASE_TEST_TARGETS'] = args.target
    env['NNCASE_TEST_DEFAULT_OPSET'] = args.opset
    env['NNCASE_TEST_DEFAULT_IR'] = args.ir
    t = args.target.upper()
    for k in [k for k in env if k.startswith(f'{t}_SSH') or k in ('NUC_PROXY_IP', 'NUC_PROXY_PORT', 'CI')]:
        del env[k]
    if mode == 'board':
        env['KPU_TARGETS'] = args.target
        env[f'{t}_SSH'] = args.board
        env[f'{t}_SSH_RUNTIME'] = args.board_runtime
        env[f'{t}_SSH_DIR'] = args.board_dir
        env[f'{t}_SSH_PYTHON'] = args.board_python
        env[f'{t}_SSH_ENV'] = args.board_env
        env[f'{t}_SSH_TIMEOUT'] = str(args.board_timeout)
    else:
        env.pop('KPU_TARGETS', None)
    if mode == 'compile':
        env['NNCASE_TEST_COMPILE_ONLY'] = '1'
        # set iteration order (e.g. parametrize values from a set) must not change between runs
        env['PYTHONHASHSEED'] = '0'
        env['NNCASE_TEST_GOLDEN_DIR'] = str(Path(args.golden_dir).resolve())
        env['NNCASE_TEST_UPDATE_GOLDEN'] = '1' if args.update_golden else '0'
        env['NNCASE_TEST_K230_AS'] = str(Path(args.k230_as).resolve()) if args.k230_as else ''
    return env


def check_setup(args, env, modes):
    module = Path(args.nncase_python_module)
    for path, what in [(module / '__init__.py', '--nncase-python-module'), (Path(args.compiler), '--compiler')]:
        if not path.exists():
            sys.exit(f'{what}: {path} not found')
    probe = subprocess.run([sys.executable, '-c', 'import nncase; print("nncase:", nncase.__file__); '
                            'print("target:", nncase.check_target("{0}"))'.format(args.target)],
                           env=env, capture_output=True, text=True)
    if probe.returncode != 0 or 'target: True' not in probe.stdout:
        sys.exit(f'host check failed (import nncase / check_target({args.target})):\n{probe.stdout}{probe.stderr}')
    loaded = next(l for l in probe.stdout.splitlines() if l.startswith('nncase:'))[len('nncase:'):].strip()
    if Path(loaded).resolve().parent != module.resolve():
        sys.exit(f'host check failed: --nncase-python-module is {module}, but python imports nncase from '
                 f'{Path(loaded).parent} (no nncase package in {module}?)')
    print(f'host: nncase: {loaded}')
    if 'board' in modes:
        # which nncaseruntime python really imports: without a package in --board-runtime it silently falls back to
        # one installed on the board (e.g. /usr/lib/python3.*/site-packages)
        code = ("import os, glob, time, hashlib, tempfile, numpy, nncaseruntime as m; "
                "d = os.path.dirname(os.path.realpath(m.__file__)); "
                f"print('want=' + os.path.realpath('{args.board_runtime}')); "
                "print('dir=' + d); print('version=' + str(getattr(m, '__version__', '?'))); "
                "print('tmp=' + tempfile.gettempdir()); "
                "[print('so=%s size=%d md5=%s mtime=%s' % (os.path.basename(f), os.path.getsize(f), "
                "hashlib.md5(open(f, 'rb').read()).hexdigest()[:12], "
                "time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f))))) "
                "for f in sorted(glob.glob(d + '/_nncaseruntime*.so'))]")
        cmd = (f'PYTHONPATH={posixpath.dirname(args.board_runtime)} {args.board_env} {args.board_python} -c '
               f'"{code}"')
        probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', args.board, cmd],
                               capture_output=True, text=True, timeout=60)
        if probe.returncode != 0:
            hint = ('\n(Git Bash turned a board path into a Windows path: run with MSYS_NO_PATHCONV=1)'
                    if ':/' in args.board_runtime + (args.board_dir or '') else '')
            sys.exit(f'board check failed on {args.board}:\n{probe.stdout}{probe.stderr}{hint}')
        info = dict(l.split('=', 1) for l in probe.stdout.splitlines() if l[:4] in ('want', 'dir=', 'vers', 'tmp='))
        modules = [l[3:] for l in probe.stdout.splitlines() if l.startswith('so=')]
        print(f'board: {args.board}')
        print(f'  runtime used: {info["dir"]} (nncaseruntime {info["version"]})')
        for module in modules:
            print(f'  module:       {module}')
        if not args.board_dir:
            args.board_dir = info['tmp']
        print(f'  scratch dir:  {args.board_dir} (a k230_serve_* folder per ssh session, removed at its end)')
        if info['dir'] != info['want']:
            sys.exit(f'board check failed: --board-runtime is {args.board_runtime}, but python imports nncaseruntime '
                     f'from {info["dir"]} (no nncaseruntime package in {args.board_runtime}?)')


def file_name(f):
    return f.stem if f.parent.name == 'basic' else '_'.join(f.relative_to(TESTS_DIR).with_suffix('').parts)


def run_file(args, mode, out, env, f):
    name = file_name(f)
    junit = out / 'junit' / f'{name}.xml'
    cases = out / 'tests_output' / name
    # --capture=sys: native output (e.g. the .NET message of a crash) goes straight to the log, not into a
    # per-test buffer that is lost when the process dies
    cmd = [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider', '-p', 'k230_test_plugin', '-v',
           '--tb=line', '--capture=sys', f'--junitxml={junit}', str(f)] + (['-k', args.keyword] if args.keyword else []) + \
        args.pytest_args
    start = time.time()
    with open(out / 'logs' / f'{name}.log', 'w') as log:
        try:
            rc = subprocess.run(cmd, cwd=out, env=dict(env, NNCASE_TEST_ROOT=str(cases)), stdout=log,
                                stderr=subprocess.STDOUT, timeout=args.file_timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
    row = summarize_file(name, junit, rc, time.time() - start, out / 'logs' / f'{name}.log')
    if mode == 'compile' and not args.update_golden and not args.keyword and row['cases']:
        golden = k230_golden.read_golden(k230_golden.golden_path(args.golden_dir, f, TESTS_DIR))
        missing = sorted(case for case in golden if case.split(' #')[0] not in row['cases'])
        if missing:
            row['counts']['missing case'] = len(missing)
            row['first']['missing case'] = f'in the golden file, not in the test file: {", ".join(missing)}'[:300]
    # compile mode keeps the cases of a file with failures: the messages point to their compile.asm
    if not args.keep_cases and (mode != 'compile' or not set(row['counts']) - {'passed', 'skipped'}):
        shutil.rmtree(cases, ignore_errors=True)
    return row


def run_mode(args, mode, files):
    out = Path(args.output).resolve() / mode
    shutil.rmtree(out, ignore_errors=True)
    (out / 'junit').mkdir(parents=True)
    (out / 'logs').mkdir()
    env = make_env(args, mode)
    jobs = 1 if mode == 'board' else args.jobs
    rows = {}
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        running = {pool.submit(run_file, args, mode, out, env, f): f for f in files}
        for done in concurrent.futures.as_completed(running):
            row = done.result()
            rows[running[done]] = row
            print(f'[{mode} {len(rows)}/{len(files)}] {format_row(row)}', flush=True)
    return [rows[f] for f in files]


def failure_kind(message):
    if 'asm changed' in message:
        return 'asm changed'
    if 'no golden' in message:
        return 'no golden'
    if 'dump / kmodel mismatch' in message:
        return 'dump / kmodel mismatch'
    if 'Fault result in infer' in message:
        return 'infer mismatch'
    if 'Fault result in eval' in message:
        return 'eval mismatch'
    if 'ssh infer' in message:
        return 'board error'
    if 'onnxruntime' in message or 'ONNXRuntime' in message:
        return 'cpu reference'
    return 'other'


def summarize_file(name, junit, rc, secs, log):
    row = {'name': name, 'rc': rc, 'secs': secs, 'counts': collections.Counter(), 'cases': {}, 'first': {}}
    if rc == 5:
        row['counts']['no tests selected'] = 1
        return row
    if not junit.exists():
        # the pytest process died (e.g. an unhandled .NET exception in the compiler): name the case it was in
        lines = log.read_text(errors='replace').splitlines() if log.exists() else []
        started = [l.split('::', 1)[1].split()[0] for l in lines if '::' in l and '.py::' in l]
        reason = next((l[l.find(m):].strip() for l in lines for m in ('Unhandled exception', 'Fatal Python error')
                       if m in l), '')
        if not reason:
            # e.g. pytest could not load the plugin (a missing python module): the first python exception line
            reason = next((l.strip() for l in lines if re.match(r'^\w+(\.\w+)*(Error|Exception): ', l)), '')
        if not reason:
            # e.g. "python: No module named pytest": the last line the process printed
            reason = next((l.strip() for l in reversed(lines) if l.strip()), '')
        row['counts']['crashed'] = 1
        row['first']['crashed'] = f"in {started[-1] if started else 'no case (before the first one)'}: {reason}"[:300]
        return row
    for tc in ET.parse(junit).getroot().iter('testcase'):
        failure = tc.find('failure')
        if failure is None:
            failure = tc.find('error')
        if tc.find('skipped') is not None:
            kind = 'skipped'
            row['first'].setdefault(kind, (tc.find('skipped').get('message') or '').replace('\n', ' ')[:200])
        elif failure is None:
            kind = 'passed'
        else:
            kind = failure_kind((failure.get('message') or '') + (failure.text or '')[-2000:])
            row['first'].setdefault(kind, (failure.get('message') or '').replace('\n', ' ')[:200])
        row['counts'][kind] += 1
        row['cases'][tc.get('name')] = kind
    return row


def format_row(row):
    counts = ', '.join(f'{v} {k}' for k, v in sorted(row['counts'].items()))
    rc = row['rc']
    rc = f'0x{rc:08X}' if isinstance(rc, int) and rc > 255 else rc
    return f"{row['name']}: {counts} ({row['secs']:.0f} s, rc={rc})"


def report(args, results):
    lines = []
    for mode, rows in results.items():
        total = collections.Counter()
        for row in rows:
            total.update(row['counts'])
        lines.append(f'===== {mode}: ' + ', '.join(f'{v} {k}' for k, v in sorted(total.items())))
        for row in rows:
            if set(row['counts']) - {'passed'}:
                lines.append('  ' + format_row(row))
                for kind, msg in row['first'].items():
                    lines.append(f'      {kind}: {msg}')
    if len(results) == 2:
        sim = {(r['name'], c): k for r in results['sim'] for c, k in r['cases'].items()}
        board = {(r['name'], c): k for r in results['board'] for c, k in r['cases'].items()}
        diff = sorted(key for key, kind in board.items()
                      if kind not in ('passed', 'skipped') and sim.get(key) == 'passed')
        lines.append(f'===== pass on simulator, fail on board: {len(diff)}')
        lines += [f'  {name}::{case}  ({board[(name, case)]})' for name, case in diff]
    text = '\n'.join(lines)
    print(text)
    (Path(args.output).resolve() / 'report.txt').write_text(text + '\n')
    print(f'results: {Path(args.output).resolve()}')


def main():
    args = parse_args()
    files = collect_files(args.paths)
    modes = ['sim', 'board'] if args.mode == 'both' else [args.mode]
    check_setup(args, make_env(args, 'sim'), modes)
    if args.mode == 'compile':
        print(f'golden: {Path(args.golden_dir).resolve()}' + (' (update)' if args.update_golden else ''))
        print(f'k230-as: {args.k230_as or "(no check)"}')
    results = {mode: run_mode(args, mode, files) for mode in modes}
    report(args, results)
    failed = any(set(r['counts']) - {'passed', 'skipped'} for rows in results.values() for r in rows)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
