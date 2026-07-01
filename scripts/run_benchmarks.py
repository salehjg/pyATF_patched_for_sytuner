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

The one thing you normally DO need to set per host is `--machine`: it picks the
right dpcpp AoT `--target` for that GPU, pins the run to a single GPU, and caps
the CPU cores used for compilation to this box's fair share (whole machine for
single-GPU hosts, total/num_gpus for multi-GPU ones). See MACHINES below and
docs/notes/cpu-core-pinning-taskset.md.

Usage:
    python scripts/run_benchmarks.py --machine furore
    python scripts/run_benchmarks.py --machine lrz --max-fevals 10 --runs 1  # quick smoke test
    python scripts/run_benchmarks.py                                          # no host preset
"""
import argparse
import json
import os
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

# One --machine value == the full arg preset for that host. Only two things
# actually vary between machines:
#   * dpcpp_target -- the dpcpp AoT --target for that GPU (an entry in
#     pyatf.cost_functions.dpcpp._TARGETS). acpp is ALWAYS 'generic': its SSCP
#     JIT specializes for the live device at runtime, so one value fits every
#     backend.
#   * num_gpus -- the box's parallel-GPU count. We always run on ONE GPU; this
#     is used ONLY to size this run's CPU-core share for compilation:
#     (allowed cores) // num_gpus. Single-GPU hosts (num_gpus=1) get every core;
#     multi-GPU hosts get a 1/num_gpus slice via taskset, so a run doesn't
#     oversubscribe cores that a peer job on another GPU would want. See
#     docs/notes/cpu-core-pinning-taskset.md.
# 'device_kind' ('cuda'|'hip'|'level_zero') confines AdaptiveCpp to this host's
# backend (see device_env) and marks which hosts need explicit single-GPU
# pinning (only multi-device Level Zero, i.e. lrz).
MACHINES = {
    'furore':   dict(device_kind='cuda',       dpcpp_target='nvidia:sm_70',  num_gpus=1),  # Tesla V100S (Volta)
    'ravello':  dict(device_kind='hip',        dpcpp_target='amd:gfx908',    num_gpus=1),  # Instinct MI100
    'p16g2':    dict(device_kind='cuda',       dpcpp_target='nvidia:sm_89',  num_gpus=1),  # RTX 2000 Ada
    'darkserv': dict(device_kind='cuda',       dpcpp_target='nvidia:sm_120', num_gpus=1),  # RTX 5060 Ti (Blackwell)
    'albori':   dict(device_kind='level_zero', dpcpp_target='intel:dg2',     num_gpus=1),  # Arc A770 (DG2/Alchemist)
    # 4x Data Center GPU Max 1550 (Ponte Vecchio), each exposing 2 tiles => 8
    # level_zero devices. We pin to a SINGLE tile (FLAT hierarchy, device 0) and
    # take 1/8 of the cores.
    'lrz':      dict(device_kind='level_zero', dpcpp_target='intel:pvc',     num_gpus=8),
}


def available_cpus():
    """Ordered list of CPU ids this process may run on, read from the affinity
    mask in /proc/self/status (respects cgroups/SLURM cpusets), falling back to
    a full 0..N-1 range. Ported from SyTuner's runme_common.available_cpus."""
    try:
        text = Path('/proc/self/status').read_text()
        line = next(l for l in text.splitlines() if l.startswith('Cpus_allowed_list:'))
        spec = line.split(':', 1)[1].strip()
    except Exception:
        spec = f'0-{(os.cpu_count() or 1) - 1}'
    cpus = []
    for rng in spec.split(','):
        rng = rng.strip()
        if '-' in rng:
            lo, hi = rng.split('-')
            cpus.extend(range(int(lo), int(hi) + 1))
        elif rng:
            cpus.append(int(rng))
    return cpus or [0]


def compress_cpus(cpus):
    """Render a CPU id list as a taskset -c spec ('0-3,32-35'), grouping only
    genuinely adjacent ids so the spec never spans a gap in the allowed mask."""
    parts = []
    i = 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        parts.append(str(cpus[i]) if i == j else f'{cpus[i]}-{cpus[j]}')
        i = j + 1
    return ','.join(parts)


def core_cpuset(num_gpus):
    """This run's CPU-core share as a taskset -c spec: the first
    (allowed cores)//num_gpus of the affinity mask. num_gpus==1 -> the whole
    allowed set (single-GPU hosts use every core)."""
    cpus = available_cpus()
    per = max(1, len(cpus) // num_gpus)
    return compress_cpus(cpus[:per])


def device_env(machine_cfg, compiler):
    """Environment for the benchmark (and its compiled program). Returns a full
    env dict to hand to subprocess. Mirrors runme_common.device_env.

    Two things happen here:

    * acpp backend confinement (every machine). AdaptiveCpp otherwise probes
      ALL backends and prints scary warnings for the ones this host lacks --
      e.g. the Level Zero ``zeInit() failed`` lines on an NVIDIA box.
      ACPP_VISIBILITY_MASK pins it to just this GPU's backend. (Harmless for
      dpcpp, which reads its own selector, so we only set it for acpp.)

    * single-GPU pinning (multi-device Level Zero only, i.e. lrz). Pin device 0
      under a deterministic FLAT hierarchy (each tile its own root device):
      AdaptiveCpp honors ZE_AFFINITY_MASK, DPC++ its own ONEAPI_DEVICE_SELECTOR.
      Single-GPU hosts need nothing -- sycl::gpu_selector_v already has one
      choice.
    """
    env = os.environ.copy()
    device_kind = machine_cfg.get('device_kind')
    if compiler == 'acpp' and device_kind:
        env['ACPP_VISIBILITY_MASK'] = {'cuda': 'cuda', 'hip': 'hip', 'level_zero': 'ze'}[device_kind]
    if device_kind == 'level_zero' and machine_cfg.get('num_gpus', 1) > 1:
        env['ZE_FLAT_DEVICE_HIERARCHY'] = 'FLAT'
        if compiler == 'acpp':
            env['ZE_AFFINITY_MASK'] = '0'
        else:  # dpcpp
            env['ONEAPI_DEVICE_SELECTOR'] = 'level_zero:0'
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-fevals', type=int, default=DEFAULT_MAX_FEVALS)
    parser.add_argument('--warmup-runs', type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument('--measurement-runs', type=int, default=DEFAULT_MEASUREMENT_RUNS)
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS)
    parser.add_argument('--machine', choices=sorted(MACHINES),
                        help='Host preset: sets the dpcpp --target for this GPU, pins to '
                             'a single GPU, and caps compile cores to this box\'s share '
                             '(see MACHINES).')
    parser.add_argument('--target', default=None,
                        help=f'dpcpp.py AoT target (overrides --machine; '
                             f'default without --machine: {DEFAULT_DPCPP_TARGET})')
    parser.add_argument('--acpp-targets', default=None,
                        help=f'acpp.py --acpp-targets value (default: {DEFAULT_ACPP_TARGETS})')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Where benchmark scripts write log/result files '
                             '(default: a fresh DEFAULT_RESULTS_BASE/launch_<ts>_<rand> dir)')
    parser.add_argument('--zip-dir', type=Path, default=DEFAULT_RESULTS_BASE,
                        help=f'Where to write the results zip (default: {DEFAULT_RESULTS_BASE})')
    args = parser.parse_args()

    # Resolve the compiler targets and the single-GPU CPU/device pinning. An
    # explicit --machine supplies the defaults; explicit --target/--acpp-targets
    # still win over it. Without --machine, fall back to this launcher's own
    # defaults and run on every core with no device pinning.
    machine_cfg = MACHINES[args.machine] if args.machine else {}
    dpcpp_target = args.target or machine_cfg.get('dpcpp_target', DEFAULT_DPCPP_TARGET)
    acpp_targets = args.acpp_targets or DEFAULT_ACPP_TARGETS
    num_gpus = machine_cfg.get('num_gpus', 1)
    # taskset only when we actually subdivide (multi-GPU hosts); single-GPU
    # hosts keep the default all-cores affinity and don't need taskset present.
    cpuset = core_cpuset(num_gpus) if num_gpus > 1 else None
    taskset_prefix = ['taskset', '-c', cpuset] if cpuset else []

    session = uuid.uuid4().hex[:8]
    output_dir = args.output_dir or (DEFAULT_RESULTS_BASE / f'launch_{int(time.time())}_{session}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Session     : {session}')
    print(f'Machine     : {args.machine or "(none -- launcher defaults)"}')
    print(f'dpcpp target: {dpcpp_target}   acpp targets: {acpp_targets}')
    print(f'CPU cores   : {cpuset or "all (single GPU)"}')
    print(f'Output dir  : {output_dir}')
    print(f'Running {len(BENCHMARKS)} benchmark(s) sequentially '
         f'(GPU timing isolation -- not parallelized on purpose)')

    launcher_log = {
        'session_id': session,
        'started_at': datetime.now().isoformat(),
        'output_dir': str(output_dir),
        'machine': args.machine,
        'cpuset': cpuset,
        'enforced_args': {
            'max_fevals': args.max_fevals,
            'warmup_runs': args.warmup_runs,
            'measurement_runs': args.measurement_runs,
            'runs': args.runs,
            'target': dpcpp_target,
            'acpp_targets': acpp_targets,
        },
        'invocations': [],
    }

    for compiler, workload, script_path in BENCHMARKS:
        cmd = taskset_prefix + [
            sys.executable, str(script_path),
            '--max-fevals', str(args.max_fevals),
            '--warmup-runs', str(args.warmup_runs),
            '--measurement-runs', str(args.measurement_runs),
            '--runs', str(args.runs),
            '--output-dir', str(output_dir),
        ]
        cmd += ['--target', dpcpp_target] if compiler == 'dpcpp' else ['--acpp-targets', acpp_targets]

        print(f'\n=== {compiler} {workload} ===')
        print(' '.join(cmd))
        start = time.time()
        ret = subprocess.run(cmd, env=device_env(machine_cfg, compiler))
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
