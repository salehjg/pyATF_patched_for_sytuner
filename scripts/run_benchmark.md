# `run_benchmarks.py` — the `--machine` presets

`scripts/run_benchmarks.py` launches the six SyTuner-ported pyATF benchmarks
(`{dpcpp,acpp}__{matmul,conv2d,pnpoly}`) sequentially, collects every
result/log JSON they write, and zips them up.

The one thing that genuinely differs between hosts — which GPU, which AoT
target, how many CPU cores to compile on — is folded into a single
**`--machine`** flag. Pass a machine name and nothing else:

```bash
python scripts/run_benchmarks.py --machine furore
```

Everything below is derived from that one word. Without `--machine`, the
launcher keeps its old defaults (`nvidia:sm_89`, all cores, no GPU pinning).

## Supported machines

| `--machine` | GPU | dpcpp `--target` | acpp `--acpp-targets` | Compile cores | Single-GPU pinning |
|---|---|---|---|---|---|
| `furore`   | Tesla V100S (Volta)        | `nvidia:sm_70`  | `generic` | all | — |
| `ravello`  | Instinct MI100             | `amd:gfx908`    | `generic` | all | — |
| `p16g2`    | RTX 2000 Ada               | `nvidia:sm_89`  | `generic` | all | — |
| `darkserv` | RTX 5060 Ti (Blackwell)    | `nvidia:sm_120` | `generic` | all | — |
| `albori`   | Arc A770 (DG2/Alchemist)   | `intel:dg2`     | `generic` | all | — |
| `lrz`      | 4× Data Center GPU Max 1550 (Ponte Vecchio) | `intel:pvc` | `generic` | 6 (hard cap) | one tile (`FLAT`, device 0) |
| `leonardo` | A100 (Ampere), 1-GPU slice | `nvidia:sm_80`  | `generic` | 8 (hard cap) | — (SLURM exposes one GPU) |

The table lives in `MACHINES` at the top of `run_benchmarks.py`; add a host by
adding a row.

## Running under SLURM (`sbatch`)

Two ready-made batch scripts run the whole suite to completion in a single
allocation (no campaign/resume machinery):

| Script | Host | `--machine` |
|---|---|---|
| `scripts/sbatch.lrz.max1550.sh`  | LRZ Intel Max 1550 node | `lrz` |
| `scripts/sbatch.leonardo.a100.sh`| CINECA Leonardo A100 node | `leonardo` |

Each one exports `MACHINE=<host>`, sources the SyTuner-tree `setvars.sh` (three
levels up — brings up the toolchain and the conda Python env), then execs
`run_benchmarks.py --machine <host>`. Submit from this `scripts/` dir and pass
the project account:

```bash
sbatch --account=<PROJECT> sbatch.leonardo.a100.sh
sbatch --account=<PROJECT> sbatch.lrz.max1550.sh --max-fevals 10 --runs 1  # smoke
```

Extra args after the script are forwarded to `run_benchmarks.py`. The same
scripts also run **without SLURM** — invoke one directly (`./sbatch.lrz.max1550.sh`)
and the `#SBATCH` lines are ignored as plain comments, so `setvars.sh` + the run
happen exactly as they would in a batch job. Override the SyTuner root with
`SYTUNER_ROOT=...` if your checkout layout differs.

**Resource sizing.** The two hosts allocate differently because their sites bill
differently:

- **`leonardo`** requests a **1/4-node slice** — `--gres=gpu:1 --cpus-per-task=8
  --mem=120G`, no `--exclusive`. The suite only ever uses one A100, and CINECA
  bills the *max* of the (GPU, CPU, memory) node-fractions, so a 1-GPU slice
  costs 1/4 of a node-hour, not a whole one. The `leonardo` preset sets
  `taskset_cores=8`, a hard cap: with a `--cpus-per-task=8` slice that's every
  allocated core, but if the job is ever handed more, `taskset` still clamps it
  to 8.
- **`lrz`** requests the **whole node** (`--exclusive`), because SuperMUC-NG
  allocates and accounts per *complete node* — there is no fractional slice to
  request. Its preset sets `taskset_cores=6` (1/8 of the 48-core node), so
  `run_benchmarks.py` pins one tile and `taskset`s the compiles to 6 cores.

## What each preset controls

### 1. The dpcpp AoT target

DPC++ ahead-of-time-compiles for the exact GPU, so each machine maps to one
entry in `pyatf.cost_functions.dpcpp._TARGETS`. Volta → `sm_70`, Ada → `sm_89`,
Blackwell → `sm_120`, MI100 → `gfx908`, Arc A770 → `dg2`, Max 1550 (which *is*
Ponte Vecchio) → `pvc`.

Two of these were added to `_TARGETS` for this work: **`amd:gfx908`** (MI100)
and **`intel:dg2`** (Arc A770). The rest were already present.

### 2. acpp is always `generic`

AdaptiveCpp's `generic` SSCP target JIT-compiles for the *live* device at
runtime, so a single value is correct on every backend (NVIDIA, AMD, Intel).
There is no per-machine acpp target to set.

### 3. CPU cores for compilation

The acpp/dpcpp compiles are what actually consume CPU. How many cores a run may
use is set per machine by an explicit **`taskset_cores`** hard cap (see
[`docs/notes/cpu-core-pinning-taskset.md`](../../../docs/notes/cpu-core-pinning-taskset.md)):

- **Uncapped hosts** (every machine except `lrz`/`leonardo`): no `taskset_cores`,
  so a run gets **every allocated core**. No `taskset` is used at all — the
  default affinity is already the whole allocation, and there's no dependency on
  `taskset` being installed.
- **Capped hosts** (`lrz` → 6, `leonardo` → 8): each benchmark subprocess is
  wrapped in `taskset -c <share>`, where the share is the **first `N`** ids of
  the affinity mask (`N` = the cap). It's a *hard* cap: if fewer than `N` cores
  are allocated, the run just uses whatever it was given. On `lrz`'s 48-core
  `--exclusive` node that's `taskset -c 0-5`, a 1/8 CPU slice so the run doesn't
  oversubscribe cores a peer job on another tile would want.

The affinity mask is read **live** (`/proc/self/status` → `Cpus_allowed_list`,
falling back to `os.cpu_count()`); no per-machine core *list* is hardcoded, only
the cap count. A **sparse / non-contiguous** allocation (e.g. `0,1,2,5,6,7,12,30`)
is honored exactly — we take the first `N` of *those* ids and `taskset` to them,
gaps preserved (`0-2,5-7` for `N=6`).

### 4. acpp backend confinement (every machine)

AdaptiveCpp otherwise probes **all** backends at startup and prints alarming
warnings for the ones a host doesn't have — most visibly a burst of
`ze_backend: Call to zeInit() failed` on an NVIDIA box. Each machine's
`device_kind` (`cuda` | `hip` | `level_zero`) sets `ACPP_VISIBILITY_MASK`
(`cuda` | `hip` | `ze`) for the **acpp** runs, confining it to this GPU's
backend so that noise disappears. dpcpp reads its own selector, so the mask is
set for acpp only.

### 5. Single-GPU device pinning

The benchmark C++ uses `sycl::gpu_selector_v`, which already picks exactly one
device — so **single-GPU hosts need no positional masking**.

`lrz` is the exception: 4 Max 1550 cards, each exposing 2 tiles = 8 Level Zero
devices. The preset pins the run to a **single tile** under a deterministic
`FLAT` hierarchy (every tile is its own root device), device 0. Because the two
SYCL runtimes read different knobs, the launcher sets them per compiler:

| Compiler | Environment (lrz) |
|---|---|
| acpp  | `ACPP_VISIBILITY_MASK=ze`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=0` |
| dpcpp | `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:0` |

These are exported into the benchmark subprocess and inherited by the compiled
program it runs. This mirrors `benchmarks/scripts/runme_common.py`'s
`device_env` in the SyTuner tree.

## dpcpp toolchain fix (`--gcc-install-dir`)

Separately from the presets, `pyatf/cost_functions/dpcpp.py` now auto-discovers
the gcc install dir (`g++ -print-file-name=crtbegin.o` → its parent) and passes
`--gcc-install-dir=<dir>` to every clang++ compile. The Intel/LLVM `clang++`
does **not** find a conda-provided gcc on its own — its default GCC scan doesn't
reach `$CONDA_PREFIX/.../x86_64-conda-linux-gnu` — so without this flag even
`#include <type_traits>` fails and every dpcpp evaluation errors out
(`min_cost=None`). This is discovered once per `CostFunction` and is a no-op if
no usable `g++` is on `PATH`. It complements the existing `--cuda-path` and
`/compat` `LD_LIBRARY_PATH` handling in the same file, and mirrors
`runme_common`'s `_gcc_install_dir` in the SyTuner tree.

> **Why a tile, not a whole card?** A `gpu_selector` picks one device = one
> tile under `FLAT`, giving a clean single-compute-tile measurement, and the
> 6-core cap (1/8 of the 48-core node) fair-shares against per-tile jobs on the
> shared node. This matches the reference runme (`num_gpus = 8`). To instead
> treat a physical card as "the GPU" you'd select `COMPOSITE` and bump the cap to
> 12 (1/4 of the node) — not the current default.

## Overrides

`--machine` only fills in *defaults*. Anything you pass explicitly still wins:

```bash
# Use furore's core/GPU handling but force a different dpcpp target:
python scripts/run_benchmarks.py --machine furore --target nvidia:sm_80

# Quick smoke test on lrz (still one tile, still 6 cores):
python scripts/run_benchmarks.py --machine lrz --max-fevals 10 --runs 1
```

`--target` overrides the dpcpp target; `--acpp-targets` overrides the acpp
target. The CPU cap follows the machine's `taskset_cores` and the GPU pinning its
`num_gpus`; neither can be overridden per-invocation (change the `MACHINES` row if
a host's layout changes).

## What the run records

The banner and `launcher_log_<session>.json` capture the resolved machine name,
the `taskset` cpuset, the dpcpp/acpp targets actually used, and the full command
line (including the `taskset` prefix) for each of the six invocations — so a
result zip is self-describing about which host preset produced it.

## Note on the `p16g2` name

The machine key is `p16g2` (as requested). The SyTuner tree's runner for the
same box is `benchmarks/scripts/runme.p16.rtx2000.py` with `MACHINE=p16`; the
two names refer to the same RTX 2000 Ada host.
