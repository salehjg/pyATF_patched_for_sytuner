#!/bin/bash
#SBATCH --job-name=pyatf-bench-lrz
#SBATCH --output=orep.%j.txt
#SBATCH --error=erep.%j.txt
#SBATCH --time=47:00:00
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Single-shot launcher for the 6-benchmark suite on an LRZ Intel Max 1550 node.
# Runs run_benchmarks.py once, to completion -- no campaign/resume logic.
# Works both under SLURM (sbatch sbatch.lrz.max1550.sh) and natively, run
# directly as a normal script (./sbatch.lrz.max1550.sh) -- the #SBATCH lines are
# just comments off the batch scheduler.
#
# Account (SLURM only): no `#SBATCH --account` -- pass it at submit time:
#   sbatch --account=<PROJECT> sbatch.lrz.max1550.sh
# Partition: "general" needs >=17 nodes; a single-node job usually belongs on
# "micro". Set --partition to whatever queue holds the Max 1550s.

# $MACHINE drives setvars.sh, which brings up the toolchain (dpcpp/icpx +
# AdaptiveCpp) and activates the conda Python env. Batch shells don't source
# ~/.bashrc, so set it here.
# Capture our own args BEFORE sourcing setvars: Intel's oneAPI setvars overwrites
# the script's positional parameters ($@) with its component list.
saved_args=("$@")
export MACHINE=lrz

# Locate this script's dir. Under SLURM the script runs from a private spool
# copy, so "$0" is not the real path -- use SLURM_SUBMIT_DIR (submit from this
# scripts/ dir). Run natively, SLURM_SUBMIT_DIR is unset and dirname "$0" is right.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# setvars.sh lives at the SyTuner repo root, three levels up from scripts/.
# Override with SYTUNER_ROOT=... if your checkout differs.
SYTUNER_ROOT="${SYTUNER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
source "$SYTUNER_ROOT/setvars.sh" || { echo "ERROR: could not source $SYTUNER_ROOT/setvars.sh" >&2; exit 1; }

# --machine lrz pins to one tile (FLAT, device 0) and 1/8 of the cores. Extra
# args (captured above) are forwarded, e.g. --max-fevals 10 --runs 1 for a smoke test.
exec python3 "$SCRIPT_DIR/run_benchmarks.py" --machine lrz "${saved_args[@]}"
