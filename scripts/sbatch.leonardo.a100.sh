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

# Single-shot launcher for the 6-benchmark suite on a CINECA Leonardo "Booster"
# node (4x NVIDIA A100). Runs run_benchmarks.py once, to completion -- no
# campaign/resume logic. Works both under SLURM (sbatch sbatch.leonardo.a100.sh)
# and natively, run directly as a normal script (./sbatch.leonardo.a100.sh) --
# the #SBATCH lines are just comments off the batch scheduler.
#
# The suite only ever uses ONE A100 sequentially, so we request just a 1/4-node
# slice -- 1 GPU, 8 of the 32 cores, ~1/4 of the memory -- NOT the whole node.
# CINECA bills the max of the (gpu, cpu, mem) node-fractions, so this costs 1/4
# of a node-hour instead of a full one (no --exclusive, no --gres=gpu:4).
# SLURM caps us to those 8 cores and exposes only the one GPU; run_benchmarks.py's
# `leonardo` preset is num_gpus=1 so it uses all 8 (no further subdivision).
# boost_qos_lprod raises the wall cap to 4 days if you bump --time past 24h.
#
# Account (SLURM only): no `#SBATCH --account` -- pass it at submit time:
#   sbatch --account=<PROJECT> sbatch.leonardo.a100.sh

# $MACHINE drives setvars.sh, which brings up the toolchain (AdaptiveCpp + nvhpc
# CUDA) and activates the conda Python env. Batch shells don't source ~/.bashrc,
# so set it here.
# Capture our own args BEFORE sourcing setvars: Intel's oneAPI setvars overwrites
# the script's positional parameters ($@) with its component list.
saved_args=("$@")
export MACHINE=leonardo

# Locate this script's dir. Under SLURM the script runs from a private spool
# copy, so "$0" is not the real path -- use SLURM_SUBMIT_DIR (submit from this
# scripts/ dir). Run natively, SLURM_SUBMIT_DIR is unset and dirname "$0" is right.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

export ONEAPI_DEVICE_SELECTOR=cuda:gpu

# setvars.sh lives at the SyTuner repo root, three levels up from scripts/.
# Override with SYTUNER_ROOT=... if your checkout differs.
SYTUNER_ROOT="${SYTUNER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
source "$SYTUNER_ROOT/setvars.sh" || { echo "ERROR: could not source $SYTUNER_ROOT/setvars.sh" >&2; exit 1; }

sycl-ls

# Extra args (captured above) are forwarded, e.g. --max-fevals 10 --runs 1 for a smoke test.
exec python3 "$SCRIPT_DIR/run_benchmarks.py" --machine leonardo "${saved_args[@]}"
