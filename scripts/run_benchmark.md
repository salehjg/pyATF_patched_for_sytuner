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
| `lrz`      | 4× Data Center GPU Max 1550 (Ponte Vecchio) | `intel:pvc` | `generic` | total ÷ 8 | one tile (`FLAT`, device 0) |

The table lives in `MACHINES` at the top of `run_benchmarks.py`; add a host by
adding a row.

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
use follows the rule in
[`docs/notes/cpu-core-pinning-taskset.md`](../../../docs/notes/cpu-core-pinning-taskset.md):

> **(cores this process is allowed to use) ÷ (number of parallel GPU slots)**

- **Single-GPU hosts** (`num_gpus = 1`): the divisor is 1, so a run gets **every
  core**. No `taskset` is used at all — the default affinity is already the whole
  machine, and there's no dependency on `taskset` being installed.
- **Multi-GPU hosts** (`lrz`, `num_gpus = 8`): each benchmark subprocess is
  wrapped in `taskset -c <share>`, where the share is the first
  `allowed_cores ÷ 8` of the affinity mask. On a 48-core node that's
  `taskset -c 0-5` (6 cores). Even though we run on one GPU, we take only a
  1/8 CPU slice so the run doesn't oversubscribe cores a peer job on another
  tile would want.

The allowed-core count is read **live** from the process affinity mask
(`/proc/self/status` → `Cpus_allowed_list`, falling back to `os.cpu_count()`);
no per-machine core count is hardcoded. A non-contiguous mask (e.g.
`0-3,32-35`) is honored exactly.

### 4. Single-GPU device pinning

The benchmark C++ uses `sycl::gpu_selector_v`, which already picks exactly one
device — so **single-GPU hosts need no masking**.

`lrz` is the exception: 4 Max 1550 cards, each exposing 2 tiles = 8 Level Zero
devices. The preset pins the run to a **single tile** under a deterministic
`FLAT` hierarchy (every tile is its own root device), device 0. Because the two
SYCL runtimes read different knobs, the launcher sets them per compiler:

| Compiler | Environment |
|---|---|
| acpp  | `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=0`, `ACPP_VISIBILITY_MASK=ze` |
| dpcpp | `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:0` |

These are exported into the benchmark subprocess and inherited by the compiled
program it runs. This mirrors `benchmarks/scripts/runme_common.py`'s
`device_env` in the SyTuner tree.

> **Why a tile, not a whole card?** A `gpu_selector` picks one device = one
> tile under `FLAT`, giving a clean single-compute-tile measurement, and the
> ÷8 core share fair-shares against per-tile jobs on the shared node. This
> matches the reference runme (`num_gpus = 8`). To instead treat a physical
> card as "the GPU" you'd switch to `COMPOSITE` and ÷4 — not the current
> default.

## Overrides

`--machine` only fills in *defaults*. Anything you pass explicitly still wins:

```bash
# Use furore's core/GPU handling but force a different dpcpp target:
python scripts/run_benchmarks.py --machine furore --target nvidia:sm_80

# Quick smoke test on lrz (still one tile, still ÷8 cores):
python scripts/run_benchmarks.py --machine lrz --max-fevals 10 --runs 1
```

`--target` overrides the dpcpp target; `--acpp-targets` overrides the acpp
target. The CPU/GPU pinning follows the machine's `num_gpus` and can't be
overridden per-invocation (change the `MACHINES` row if a host's layout
changes).

## What the run records

The banner and `launcher_log_<session>.json` capture the resolved machine name,
the `taskset` cpuset, the dpcpp/acpp targets actually used, and the full command
line (including the `taskset` prefix) for each of the six invocations — so a
result zip is self-describing about which host preset produced it.

## Note on the `p16g2` name

The machine key is `p16g2` (as requested). The SyTuner tree's runner for the
same box is `benchmarks/scripts/runme.p16.rtx2000.py` with `MACHINE=p16`; the
two names refer to the same RTX 2000 Ada host.
