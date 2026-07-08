"""Shared plumbing for the SyTuner-ported benchmark examples
(dpcpp/acpp x matmul/conv2d/pnpoly) -- argparse defaults, unique output
paths, and the flat result-JSON convention.

Unlike pyatf/cost_functions/*.py, these benchmark scripts deliberately share
this helper instead of duplicating it six times: this is launcher/tooling
code, not a cost-function backend, and the things it does (unique-path
generation, result-dict assembly) are exactly the kind of fiddly bookkeeping
that's worth getting right once rather than six times.
"""
import argparse
import json
import shutil
import socket
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Enforced defaults -- see docs/notes/benchmark-ports.md for where these came
# from (SyTuner's KernelTuner runner: tune_common.py / runme_common.py).
DEFAULT_MAX_FEVALS = 250
DEFAULT_WARMUP_RUNS = 2
DEFAULT_MEASUREMENT_RUNS = 5
DEFAULT_RUNS = 5
DEFAULT_DPCPP_TARGET = 'nvidia:sm_89'
DEFAULT_ACPP_TARGETS = 'generic'


def add_common_args(parser: argparse.ArgumentParser, compiler: str, default_output_dir: str) -> argparse.ArgumentParser:
    """`default_output_dir` is required, not a module constant: each
    benchmark script lives in its own directory and should default to
    dumping output next to itself (`<script's dir>/dir_dumps`), not into one
    shared location every script would otherwise collide into. Callers pass
    `str(Path(__file__).resolve().parent / 'dir_dumps')`."""
    parser.add_argument('--max-fevals', type=int, default=DEFAULT_MAX_FEVALS,
                        help=f'Evaluation budget per run (default: {DEFAULT_MAX_FEVALS})')
    parser.add_argument('--warmup-runs', type=int, default=DEFAULT_WARMUP_RUNS,
                        help=f'Warmup kernel launches before timing (default: {DEFAULT_WARMUP_RUNS})')
    parser.add_argument('--measurement-runs', type=int, default=DEFAULT_MEASUREMENT_RUNS,
                        help=f'Timed kernel launches per evaluation, median taken (default: {DEFAULT_MEASUREMENT_RUNS})')
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS,
                        help=f'Independent from-scratch tuning runs (default: {DEFAULT_RUNS})')
    parser.add_argument('--output-dir', default=default_output_dir,
                        help=f'Where to write log/result files (default: {default_output_dir})')
    if compiler == 'dpcpp':
        parser.add_argument('--target', default=DEFAULT_DPCPP_TARGET,
                            help=f'dpcpp.py AoT target (default: {DEFAULT_DPCPP_TARGET})')
    elif compiler == 'acpp':
        parser.add_argument('--acpp-targets', default=DEFAULT_ACPP_TARGETS,
                            help=f'acpp.py --acpp-targets value (default: {DEFAULT_ACPP_TARGETS})')
    else:
        raise ValueError(f'unknown compiler: {compiler}')
    parser.add_argument('--max-workgroup-size', type=int, default=None,
                        help='Override the device max work-group size instead of '
                             'querying it off the live SYCL device')
    parser.add_argument('--max-shared-memory', type=int, default=None,
                        help='Override the device local/shared memory size in BYTES '
                             'instead of querying it off the live SYCL device')
    return parser


# Tiny SYCL program printing the two device limits that bound every
# benchmark's search space. It is compiled and run through the SAME
# cost-function plumbing (compiler, targets, toolchain flags, runtime env) as
# the benchmark itself, so the device being asked is exactly the one the
# tuning will run on.
_LIMITS_PROBE_SOURCE = '''
#include <sycl/sycl.hpp>
#include <fstream>
int main() {
    sycl::queue q{ sycl::gpu_selector_v };
    auto dev = q.get_device();
    std::ofstream out("__LIMITS_FILE_PATH__");
    out << dev.get_info<sycl::info::device::max_work_group_size>() << ' '
        << dev.get_info<sycl::info::device::local_mem_size>() << '\\n';
    return 0;
}
'''


def get_device_limits(compiler: str, args: argparse.Namespace) -> Dict[str, int]:
    """{max_workgroup_size, max_shared_memory} of the live SYCL device.

    The pyATF-side equivalent of SyTuner's sytuner.query_device_limits(): the
    same two SYCL queries (max_work_group_size, local_mem_size), answered by
    the live device. Hardcoding is wrong here -- 48 KB shared memory is NOT
    universal (the NVIDIA cards report 48 KB, but MI100 has 64 KB and the
    Max 1550 128 KB), and a too-small value silently shrinks the search space.
    Like SyTuner, there is no silent fallback: if the probe cannot run, the
    limits must be given explicitly (--max-workgroup-size/--max-shared-memory,
    which also override a successful query).
    """
    overrides = {'max_workgroup_size': args.max_workgroup_size,
                 'max_shared_memory': args.max_shared_memory}
    if all(v is not None for v in overrides.values()):
        print(f'device limits (CLI overrides): {overrides}')
        return overrides

    tmpdir = tempfile.mkdtemp(prefix='pyatf_devlimits_')
    limits_path = Path(tmpdir) / 'limits.txt'
    probe_source = _LIMITS_PROBE_SOURCE.replace('__LIMITS_FILE_PATH__', str(limits_path))
    try:
        if compiler == 'dpcpp':
            from pyatf.cost_functions import dpcpp
            probe = dpcpp.CostFunction(probe_source).target(args.target)
        else:
            from pyatf.cost_functions import acpp
            probe = acpp.CostFunction(probe_source).targets(args.acpp_targets)
        probe({})  # no tunables -- just compile the probe once and run it
        wg, slm = limits_path.read_text().split()
        limits = {'max_workgroup_size': int(wg), 'max_shared_memory': int(slm)}
    except Exception as e:
        sys.exit(f'ERROR: could not query the SYCL device for its limits ({e}).\n'
                 'Refusing to guess a search space. Pass --max-workgroup-size and\n'
                 '--max-shared-memory (bytes) explicitly instead.')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    limits.update({k: v for k, v in overrides.items() if v is not None})
    print(f'device limits (queried live): max_workgroup_size={limits["max_workgroup_size"]}, '
          f'max_shared_memory={limits["max_shared_memory"]} bytes')
    return limits


def new_session_id() -> str:
    """One short random id per script invocation -- shared by every run
    within that invocation, so files from the same run of the script are
    visibly grouped, while still being unique across separate invocations
    (even of the identical script with identical args)."""
    return uuid.uuid4().hex[:8]


def run_paths(output_dir: str, compiler: str, workload: str, run_idx: int, session: str) -> Dict[str, Path]:
    """Unique file paths for one (script invocation, run index). The shared
    `session` id plus `run_idx` guarantees no two runs -- even concurrent
    invocations of the same script -- ever write to the same path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{compiler}_{workload}_run{run_idx}_{session}'
    return {
        'log_file': out / f'{stem}.json',
        'result_file': out / f'{stem}_result.json',
        'cost_file': out / f'{stem}_cost.txt',
        'device_file': out / f'{stem}_device.txt',
    }


def read_device_name(device_file: Path) -> str:
    try:
        return device_file.read_text().strip()
    except FileNotFoundError:
        return 'unknown'  # every evaluation failed before writing it


def write_result_json(path: Path, **fields: Any) -> None:
    with open(path, 'w') as f:
        json.dump(fields, f, indent=2)


def base_result_fields(workload: str, compiler: str, args: argparse.Namespace,
                       run_idx: int, session: str, script_name: str,
                       device_name: str, log_file: Path,
                       config: Optional[Dict[str, Any]], min_cost: Optional[float],
                       tuning_data) -> Dict[str, Any]:
    """Fields common to every benchmark's flat result dict. Each benchmark
    script adds its own problem-size fields (prefixed problem_*) and merges
    in `config` itself -- done here via **config passthrough callers handle,
    since key names differ per benchmark."""
    fields: Dict[str, Any] = {
        'workload': workload,
        'compiler': compiler,
        'device_name': device_name,
        'script': script_name,
        'session_id': session,
        'run_index': run_idx,
        'runs_total': args.runs,
        'max_fevals': args.max_fevals,
        'warmup_runs': args.warmup_runs,
        'measurement_runs': args.measurement_runs,
        'hostname': socket.gethostname(),
        'log_file': str(log_file),
        'tuning_start_timestamp': tuning_data.tuning_start_timestamp.isoformat(),
        'total_tuning_duration_seconds': tuning_data.total_tuning_duration.total_seconds(),
        'search_technique': tuning_data.search_technique.get('kind'),
        'constrained_search_space_size': tuning_data.constrained_search_space_size,
        'unconstrained_search_space_size': tuning_data.unconstrained_search_space_size,
        'number_of_evaluated_configurations': tuning_data.number_of_evaluated_configurations,
        'number_of_evaluated_valid_configurations': tuning_data.number_of_evaluated_valid_configurations,
        'number_of_evaluated_invalid_configurations': tuning_data.number_of_evaluated_invalid_configurations,
        'best_cost': min_cost,
        'evaluations_to_min_cost': tuning_data.evaluations_to_min_cost(),
    }
    if compiler == 'dpcpp':
        fields['target'] = args.target
    else:
        fields['acpp_targets'] = args.acpp_targets
    duration_to_min_cost = tuning_data.duration_to_min_cost()
    fields['duration_to_min_cost_seconds'] = (
        duration_to_min_cost.total_seconds() if duration_to_min_cost is not None else None
    )
    fields.update(config or {})
    return fields
