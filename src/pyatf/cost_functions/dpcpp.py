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


def _gcc_install_dir() -> Optional[str]:
    """The GCC install dir (the directory holding crtbegin.o) of the g++ on
    PATH, asked from g++ itself.

    The Intel/LLVM clang++ does NOT auto-discover a conda gcc: it lives under
    $CONDA_PREFIX with the x86_64-conda-linux-gnu triple, outside clang's
    default GCC scan (clang's sysroot is its own llvm install dir). Without
    help, clang finds no C++ toolchain at all and the SYCL headers fail to
    locate even <type_traits>. Hand clang the exact install dir, asked from gcc
    itself so there is no version/triple to hardcode.

    Returns None if g++ is absent or the runtime object can't be found (then no
    flag is added -- identical to the old behavior). Mirrors
    runme_common.BenchmarkRunner._gcc_install_dir in the SyTuner tree.
    """
    gxx = shutil.which('g++')
    if not gxx:
        return None
    try:
        out = subprocess.run([gxx, '-print-file-name=crtbegin.o'],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
    crt = Path(out)
    # -print-file-name echoes the bare name back when it can't locate the file.
    return str(crt.parent.resolve()) if crt.is_file() else None


def _runtime_env(target: Optional[str]):
    # We do NOT touch LD_LIBRARY_PATH: the toolchain env (setvars.all.sh) is the
    # single source of truth for it, including the nvhpc CUDA forward-compat
    # libcuda gate. The one thing setvars deliberately leaves unset is the CUDA
    # device selector -- without it the SYCL device enumeration can pick the
    # wrong backend and fail with CUDA_ERROR_INVALID_DEVICE even though the
    # device is fine -- so that is all we add here.
    env = os.environ.copy()
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
        # Point clang at the (conda) gcc toolchain once -- it can't find one on
        # its own, so without this even <type_traits> fails to resolve. Empty if
        # no usable g++ is on PATH (then the compile is unchanged). See
        # _gcc_install_dir.
        gcc_dir = _gcc_install_dir()
        self._toolchain_flags = [f'--gcc-install-dir={gcc_dir}'] if gcc_dir else []

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

        # -O3 to MATCH acpp.py (which pins -O3): clang++ -fsycl defaults to -O2,
        # not -O0, so this looked unnecessary -- but -O2 under-optimizes branchy
        # kernels (measured on sm_120: pnpoly ~1.45x slower at -O2 than -O3, which
        # alone flipped the acpp/dpcpp verdict). Both toolchains must compile at the
        # same -O3 or the cross-compiler comparison is invalid.
        # -fuse-ld=lld dodges a glibc/conda toolchain conflict in the default linker
        argv = [self._compiler, '-fsycl', '-fuse-ld=lld', '-O3',
                *self._toolchain_flags, *self._target_flags, *self._extra_flags]
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
