import os
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


# DPC++ only JITs SPIR-V at the driver level, which really only covers Intel GPUs.
# NVIDIA/AMD need the CUDA/HIP codegen backends, and those are ahead-of-time only.
_TARGETS = {
    'intel:pvc':     ('spir64_gen',         ['-Xsycl-target-backend', '-device pvc']),
    'intel:dg2':     ('spir64_gen',         ['-Xsycl-target-backend', '-device dg2']),  # Arc A770 (DG2/Alchemist)
    'nvidia:sm_70':  ('nvptx64-nvidia-cuda', ['-Xsycl-target-backend=nvptx64-nvidia-cuda', '--cuda-gpu-arch=sm_70']),
    'nvidia:sm_80':  ('nvptx64-nvidia-cuda', ['-Xsycl-target-backend=nvptx64-nvidia-cuda', '--cuda-gpu-arch=sm_80']),
    'nvidia:sm_89':  ('nvptx64-nvidia-cuda', ['-Xsycl-target-backend=nvptx64-nvidia-cuda', '--cuda-gpu-arch=sm_89']),
    'nvidia:sm_120': ('nvptx64-nvidia-cuda', ['-Xsycl-target-backend=nvptx64-nvidia-cuda', '--cuda-gpu-arch=sm_120']),
    'amd:gfx90a':    ('amdgcn-amd-amdhsa',   ['-Xsycl-target-backend=amdgcn-amd-amdhsa', '--offload-arch=gfx90a']),
    'amd:gfx908':    ('amdgcn-amd-amdhsa',   ['-Xsycl-target-backend=amdgcn-amd-amdhsa', '--offload-arch=gfx908']),  # MI100
}


def _runtime_env(target: Optional[str]):
    # nvhpc's CUDA forward-compat libcuda.so is older than the real driver and
    # shadows it on LD_LIBRARY_PATH, which makes device init fail with
    # CUDA_ERROR_SYSTEM_DRIVER_MISMATCH -- strip it so the real one is used
    env = os.environ.copy()
    ld_library_path = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = ':'.join(p for p in ld_library_path.split(':') if not p.endswith('/compat'))
    # without this, the SYCL device enumeration can pick the wrong backend and
    # fail with CUDA_ERROR_INVALID_DEVICE even though the device is fine
    if target is not None and target.startswith('nvidia:'):
        env.setdefault('ONEAPI_DEVICE_SELECTOR', 'cuda:gpu')
    return env


class CostFunction:
    def __init__(self, source: str):
        self._source = source

        self._compiler = 'clang++'
        self._target: Optional[str] = None
        self._target_flags = []
        self._extra_flags = []
        self._cost_file: Optional[str] = None

        self._workdir = tempfile.mkdtemp(prefix='pyatf_dpcpp_')

    def __del__(self):
        shutil.rmtree(self._workdir, ignore_errors=True)

    def compiler(self, compiler: str):
        self._compiler = compiler
        return self

    def target(self, target: str):
        if target not in _TARGETS:
            raise ValueError(f'unknown target "{target}", use target_flags() for hardware not in the table')
        triple, flags = _TARGETS[target]
        self._target = target
        self._target_flags = [f'-fsycl-targets={triple}', *flags]
        # clang doesn't honor $CUDA_PATH for libdevice lookup, has to be a flag
        if target.startswith('nvidia:') and 'CUDA_PATH' in os.environ:
            self._target_flags.append(f'--cuda-path={os.environ["CUDA_PATH"]}')
        return self

    def target_flags(self, flags: Iterable[str]):
        self._target_flags = list(flags)
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

        # -fuse-ld=lld dodges a glibc/conda toolchain conflict in the default linker
        argv = [self._compiler, '-fsycl', '-fuse-ld=lld', *self._target_flags, *self._extra_flags]
        for tp_name, tp_value in configuration.items():
            argv.append(f'-D{tp_name}={tp_value}')
        argv += [str(src_path), '-o', str(bin_path)]

        ret = subprocess.run(argv)
        if ret.returncode != 0:
            raise CostFunctionError('dpcpp AoT compile failed: ' + ' '.join(argv))

        run_env = _runtime_env(self._target)
        run_start = time.perf_counter_ns()
        ret = subprocess.run([str(bin_path)], env=run_env)
        run_end = time.perf_counter_ns()
        if ret.returncode != 0:
            raise CostFunctionError('compiled program exited with non-zero status')

        if self._cost_file is None:
            return float(run_end - run_start)
        with open(self._cost_file, 'r') as f:
            return float(f.read())
