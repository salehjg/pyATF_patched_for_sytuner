import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

from pyatf.tuning_data import Configuration, Cost, CostFunctionError


def source(source: str):
    return source


def path(path: str):
    with open(path, 'r') as f:
        return f.read()


class CostFunction:
    def __init__(self, source: str):
        self._source = source

        self._compiler = 'acpp'
        self._acpp_targets = 'generic'
        self._extra_flags = []
        self._cost_file: Optional[str] = None

        self._workdir = tempfile.mkdtemp(prefix='pyatf_acpp_')

    def __del__(self):
        shutil.rmtree(self._workdir, ignore_errors=True)

    def compiler(self, compiler: str):
        self._compiler = compiler
        return self

    def targets(self, acpp_targets: str):
        self._acpp_targets = acpp_targets
        return self

    def flags(self, flags: Iterable[str]):
        self._extra_flags = list(flags)
        return self

    def cost_file(self, cost_file: str):
        self._cost_file = cost_file
        return self

    def __call__(self, configuration: Configuration) -> Cost:
        src_path = Path(self._workdir) / 'program.cpp'
        bin_path = Path(self._workdir) / 'program'
        src_path.write_text(self._source)

        # unlike clang++ -fsycl, plain acpp defaults to -O0 -- fatal for a
        # tool whose whole point is measuring/tuning performance
        argv = [self._compiler, f'--acpp-targets={self._acpp_targets}', '-O3', *self._extra_flags]
        for tp_name, tp_value in configuration.items():
            argv.append(f'-D{tp_name}={tp_value}')
        argv += [str(src_path), '-o', str(bin_path)]

        ret = subprocess.run(argv)
        if ret.returncode != 0:
            raise CostFunctionError('acpp compile failed: ' + ' '.join(argv))

        # No env manipulation here: the toolchain env (setvars.all.sh) is the
        # single source of truth -- including the nvhpc CUDA forward-compat
        # libcuda gate -- so the compiled program just inherits it.
        run_start = time.perf_counter_ns()
        ret = subprocess.run([str(bin_path)])
        run_end = time.perf_counter_ns()
        if ret.returncode != 0:
            raise CostFunctionError('compiled program exited with non-zero status')

        if self._cost_file is None:
            return float(run_end - run_start)
        with open(self._cost_file, 'r') as f:
            return float(f.read())
