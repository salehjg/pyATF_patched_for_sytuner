#!/bin/bash
#SBATCH --job-name=pyatf-bench-lrz
#SBATCH --output=orep.%j.txt
#SBATCH --error=erep.%j.txt
#SBATCH --time=47:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=NONE

# Single-shot launcher for the 6-benchmark pyATF suite on an LRZ (SuperMUC-NG
# Phase 2) Intel Max 1550 node. Runs run_benchmarks.py once, to completion --
# no campaign/resume logic. Submit from this scripts/ dir:
#
#   sbatch --account=<PROJECT> sbatch.lrz.max1550.sh [extra args...]
#
# There is no #SBATCH --account on purpose -- the project is chosen at submit
# time. Extra args are forwarded to run_benchmarks.py, e.g.
# "--max-fevals 10 --runs 1" for a smoke test. Run natively instead
# (./sbatch.lrz.max1550.sh) and the #SBATCH lines are ignored as comments.
#
# Resources: LRZ allocates and accounts whole nodes, so there is no fractional
# slice to request. --time=47:00:00 stays 1 h under the 48 h wall cap as a
# safety margin. run_benchmarks.py's `lrz` preset pins the run to a single
# tile (FLAT hierarchy, device 0) and 1/8 of the cores. Adjust --partition if
# the Max 1550s live on a different queue of your allocation.
#
# --export=NONE: do NOT inherit the submitting shell's environment (sbatch's
# default is --export=ALL). The toolchain recipes' setvars.sh files dedup
# PATH/LD_LIBRARY_PATH entries, so inherited dirs KEEP their stale positions
# instead of being re-prepended -- a polluted login shell silently reorders
# toolchain precedence inside the job (see sbatch.leonardo.a100.sh for the
# shadowing incident this caused). With NONE, Slurm recreates a clean login
# env on the node and setvars.sh rebuilds the same ordering every time. To
# hand one variable through anyway: sbatch --export=VAR=value (implies NONE
# for the rest).

# Capture our args BEFORE sourcing setvars: Intel's oneAPI setvars overwrites
# the script's positional parameters ($@) with its component list.
saved_args=("$@")

# $MACHINE drives setvars.sh (toolchain + conda Python env). Batch shells
# don't source ~/.bashrc, so set it here.
export MACHINE=lrz

# Under SLURM the script runs from a private spool copy, so "$0" is not the
# real path -- use SLURM_SUBMIT_DIR (submit from this scripts/ dir). Run
# natively, SLURM_SUBMIT_DIR is unset and dirname "$0" is right.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# setvars.sh lives at the SyTuner repo root, three levels up from scripts/.
# Override with SYTUNER_ROOT=... if your checkout differs.
SYTUNER_ROOT="${SYTUNER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
source "$SYTUNER_ROOT/setvars.sh" || { echo "ERROR: could not source $SYTUNER_ROOT/setvars.sh" >&2; exit 1; }

# Preflight diagnostic for the job log: sycl-ls should list the Max 1550
# tiles as level_zero devices (per-benchmark device pinning happens inside
# run_benchmarks.py's `lrz` preset).
sycl-ls

exec python3 "$SCRIPT_DIR/run_benchmarks.py" --machine lrz "${saved_args[@]}"
