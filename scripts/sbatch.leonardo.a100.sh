#!/bin/bash
#SBATCH --job-name=pyatf-bench-leonardo
#SBATCH --output=orep.%j.txt
#SBATCH --error=erep.%j.txt
#SBATCH --time=3-00:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --export=NONE

# Single-shot launcher for the 6-benchmark pyATF suite on a CINECA Leonardo
# "Booster" node (4x NVIDIA A100). Runs run_benchmarks.py once, to completion
# -- no campaign/resume logic. Submit from this scripts/ dir:
#
#   sbatch --account=<PROJECT> sbatch.leonardo.a100.sh [extra args...]
#
# There is no #SBATCH --account on purpose -- the project is chosen at submit
# time. Extra args are forwarded to run_benchmarks.py, e.g.
# "--max-fevals 10 --runs 1" for a smoke test. Run natively instead
# (./sbatch.leonardo.a100.sh) and the #SBATCH lines are ignored as comments.
#
# Resources: the suite uses ONE A100 sequentially, so request a 1/4-node slice
# (1 GPU, 8 of the 32 cores, ~1/4 of the memory), NOT the whole node. CINECA
# bills the max of the (gpu, cpu, mem) node-fractions, so this costs 1/4 of a
# node-hour instead of a full one. run_benchmarks.py's `leonardo` preset
# (num_gpus=1) then uses all 8 allocated cores, and SLURM exposes only the one
# GPU. boost_qos_lprod raises the wall cap to 4 days if you bump --time past 24 h.
#
# --export=NONE: do NOT inherit the submitting shell's environment (sbatch's
# default is --export=ALL). The toolchain recipes' setvars.sh files dedup
# PATH/LD_LIBRARY_PATH entries, so inherited dirs KEEP their stale positions
# instead of being re-prepended -- a polluted login shell silently reorders
# toolchain precedence inside the job. That is how oneAPI Basekit's
# sycl-ls/libsycl.so.8 intermittently shadowed dpcpp's CUDA-enabled ones here
# ("no gpu available" while nvidia-smi worked). With NONE, Slurm recreates a
# clean login env on the node and setvars.sh rebuilds the same ordering every
# time. To hand one variable through anyway: sbatch --export=VAR=value
# (implies NONE for the rest).

# Capture our args BEFORE sourcing setvars: Intel's oneAPI setvars overwrites
# the script's positional parameters ($@) with its component list.
saved_args=("$@")

# $MACHINE drives setvars.sh (toolchain + conda Python env). Batch shells
# don't source ~/.bashrc, so set it here.
export MACHINE=leonardo

# Under SLURM the script runs from a private spool copy, so "$0" is not the
# real path -- use SLURM_SUBMIT_DIR (submit from this scripts/ dir). Run
# natively, SLURM_SUBMIT_DIR is unset and dirname "$0" is right.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# setvars.sh lives at the SyTuner repo root, three levels up from scripts/.
# Override with SYTUNER_ROOT=... if your checkout differs.
SYTUNER_ROOT="${SYTUNER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
source "$SYTUNER_ROOT/setvars.sh" || { echo "ERROR: could not source $SYTUNER_ROOT/setvars.sh" >&2; exit 1; }

# Preflight diagnostic for the job log: sycl-ls must resolve to the DPC++
# (llvmintel) build and list the allocated A100 under the cuda backend. The
# selector is scoped to this one call -- the benchmark runs don't need it
# exported (pyatf's dpcpp cost function sets its own per run).
ONEAPI_DEVICE_SELECTOR=cuda:gpu sycl-ls

exec python3 "$SCRIPT_DIR/run_benchmarks.py" --machine leonardo "${saved_args[@]}"
