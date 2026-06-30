#!/usr/bin/env python3
"""Launch the SyTuner-ported pyATF benchmarks (examples/full_examples/
{dpcpp,acpp}__{matmul,conv2d,pnpoly}), collect every result/log file they
write, and zip them up.

Deliberately a plain Python script, not bash: argument handling, the per-run
file bookkeeping, and the zipping are all things bash makes needlessly fragile.

Every argument is passed to each benchmark script EXPLICITLY, with this
script's own copy of the default values (duplicated from
examples/full_examples/_bench_common.py on purpose, not imported from it) --
the point is that this launcher's behavior can't silently drift just because
someone edited a default somewhere else. See docs/notes/benchmark-ports.md.

Runs the 6 benchmarks SEQUENTIALLY, on purpose: they all target the same GPU,
and running two at once would skew both their timing measurements.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --max-fevals 10 --runs 1   # quick smoke test
"""
import argparse
import json
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_EXAMPLES = REPO_ROOT / 'examples' / 'full_examples'

# Own copies of the enforced defaults -- NOT imported from _bench_common.py.
# See the module docstring for why that's intentional.
DEFAULT_MAX_FEVALS = 1024
DEFAULT_WARMUP_RUNS = 2
DEFAULT_MEASUREMENT_RUNS = 5
DEFAULT_RUNS = 1
DEFAULT_DPCPP_TARGET = 'nvidia:sm_89'
DEFAULT_ACPP_TARGETS = 'generic'
# Next to this script, not /tmp -- output should survive a reboot and be
# where you'd actually look for it.
DEFAULT_RESULTS_BASE = Path(__file__).resolve().parent / 'dir_dumps'

BENCHMARKS = [
    ('dpcpp', 'matmul', FULL_EXAMPLES / 'dpcpp__matmul' / 'dpcpp__matmul.py'),
    ('acpp', 'matmul', FULL_EXAMPLES / 'acpp__matmul' / 'acpp__matmul.py'),
    ('dpcpp', 'conv2d', FULL_EXAMPLES / 'dpcpp__conv2d' / 'dpcpp__conv2d.py'),
    ('acpp', 'conv2d', FULL_EXAMPLES / 'acpp__conv2d' / 'acpp__conv2d.py'),
    ('dpcpp', 'pnpoly', FULL_EXAMPLES / 'dpcpp__pnpoly' / 'dpcpp__pnpoly.py'),
    ('acpp', 'pnpoly', FULL_EXAMPLES / 'acpp__pnpoly' / 'acpp__pnpoly.py'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-fevals', type=int, default=DEFAULT_MAX_FEVALS)
    parser.add_argument('--warmup-runs', type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument('--measurement-runs', type=int, default=DEFAULT_MEASUREMENT_RUNS)
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS)
    parser.add_argument('--target', default=DEFAULT_DPCPP_TARGET, help='dpcpp.py AoT target')
    parser.add_argument('--acpp-targets', default=DEFAULT_ACPP_TARGETS)
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Where benchmark scripts write log/result files '
                             '(default: a fresh DEFAULT_RESULTS_BASE/launch_<ts>_<rand> dir)')
    parser.add_argument('--zip-dir', type=Path, default=DEFAULT_RESULTS_BASE,
                        help=f'Where to write the results zip (default: {DEFAULT_RESULTS_BASE})')
    args = parser.parse_args()

    session = uuid.uuid4().hex[:8]
    output_dir = args.output_dir or (DEFAULT_RESULTS_BASE / f'launch_{int(time.time())}_{session}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Session     : {session}')
    print(f'Output dir  : {output_dir}')
    print(f'Running {len(BENCHMARKS)} benchmark(s) sequentially '
         f'(GPU timing isolation -- not parallelized on purpose)')

    launcher_log = {
        'session_id': session,
        'started_at': datetime.now().isoformat(),
        'output_dir': str(output_dir),
        'enforced_args': {
            'max_fevals': args.max_fevals,
            'warmup_runs': args.warmup_runs,
            'measurement_runs': args.measurement_runs,
            'runs': args.runs,
            'target': args.target,
            'acpp_targets': args.acpp_targets,
        },
        'invocations': [],
    }

    for compiler, workload, script_path in BENCHMARKS:
        cmd = [
            sys.executable, str(script_path),
            '--max-fevals', str(args.max_fevals),
            '--warmup-runs', str(args.warmup_runs),
            '--measurement-runs', str(args.measurement_runs),
            '--runs', str(args.runs),
            '--output-dir', str(output_dir),
        ]
        cmd += ['--target', args.target] if compiler == 'dpcpp' else ['--acpp-targets', args.acpp_targets]

        print(f'\n=== {compiler} {workload} ===')
        print(' '.join(cmd))
        start = time.time()
        ret = subprocess.run(cmd)
        duration = time.time() - start
        launcher_log['invocations'].append({
            'compiler': compiler,
            'workload': workload,
            'command': cmd,
            'returncode': ret.returncode,
            'duration_seconds': duration,
        })
        if ret.returncode != 0:
            print(f'[WARN] {compiler} {workload} exited with code {ret.returncode}')

    launcher_log['finished_at'] = datetime.now().isoformat()
    launcher_log_path = output_dir / f'launcher_log_{session}.json'
    with open(launcher_log_path, 'w') as f:
        json.dump(launcher_log, f, indent=2)

    # Gather everything -- per-run pyATF logs, per-run flat result jsons, and
    # this launcher's own log, all already living in output_dir -- and zip it.
    json_files = sorted(output_dir.glob('*.json'))
    args.zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.zip_dir / f'pyatf_results_{session}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for jf in json_files:
            zf.write(jf, arcname=jf.name)

    print(f'\nCollected {len(json_files)} json file(s) from {output_dir}')
    print(f'Results zip: {zip_path}')


if __name__ == '__main__':
    main()
