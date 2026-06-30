#!/usr/bin/env python3
"""Plot tuning convergence from pyATF log files, in the same two-panel
convergence style as SyTuner's benchmarks/scripts/plot_convergence.py for
KernelTuner caches.

A pyATF log file is whatever Tuner().log_file(path) wrote -- the JSON dump of
TuningData (see pyatf/tuning_data.py). Scans the given paths for such files and
renders one two-panel convergence figure per log: cost vs (a) evaluation number
and (b) wall-clock time since tuning start, both with a "best so far" step line.
Unlike the KernelTuner-cache version, there's no running-minimum to compute --
pyATF already tracks it as `improvement_history`, so that's used directly.

Requires matplotlib and seaborn (not pyATF's own dependencies):
    pip install matplotlib seaborn

Usage:
    python scripts/plot_convergence.py --input run1.json --input logs/ --output plots/

Each figure is named after its log file's stem, e.g. a log saved to
dpcpp_matmul_run1.json becomes plot01_dpcpp_matmul_run1.png.

Caveat: unlike kt_cache.json, a pyATF log file has no fixed name -- it's
whatever path was passed to log_file(...). Pointing --input at a directory
glob-matches every *.json in it, so make sure that directory only contains
pyATF logs (or pass exact file paths instead). Files that don't parse as a
pyATF log are skipped with a message, not silently ignored.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List

import seaborn as sns
import matplotlib
# This script only ever *saves* figures (no plt.show()); force the headless Agg
# backend so it never touches a GUI toolkit.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_hms(s: str) -> float:
    """Parse pyATF's "H:MM:SS.ffffff" timedelta_since_tuning_start (or
    duration_to_min_cost / total_tuning_duration) format into seconds."""
    h, m, sec = s.split(':')
    return int(h) * 3600 + int(m) * 60 + float(sec)


class ConvergencePlotter:
    def __init__(self, log_file: Path, output_path: Path, name: str, unit: str):
        self.log_file = log_file
        self.output_path = output_path
        # Descriptive token used for both the output filename and the figure
        # identity -- the log file's stem, e.g. "dpcpp_matmul_run1".
        self.name = name
        self.unit = unit

    # ------------------------------------------------------------ styling ---
    def _apply_style(self) -> None:
        # Same "beautiful defaults" block SyTuner's plot.py/plot_convergence.py use.
        sns.set_theme(
            style="whitegrid",
            context="talk",
            font="DejaVu Sans",
            rc={
                "figure.dpi": 160,
                "savefig.dpi": 400,  # crisp export
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
                "axes.edgecolor": "0.15",
                "grid.color": "0.86",
                "grid.linewidth": 0.9,
                "axes.spines.top": False,
                "axes.spines.right": False,
            },
        )
        self.best_color = sns.color_palette("crest", 3)[1]
        self.point_color = "0.6"

    # ------------------------------------------------------- output path ---
    def _resolve_output_path(self) -> Path:
        out = self.output_path
        out.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", self.name).strip("_")
        return out / f"plot01_{safe}.png"

    # -------------------------------------------------------------- load ---
    def _load(self):
        meta = json.load(open(self.log_file))
        history = meta.get("history")
        improvement = meta.get("improvement_history")
        if history is None or improvement is None:
            raise ValueError("not a pyATF log file (missing history/improvement_history)")

        valid = [e for e in history if e.get("valid") and e.get("cost") is not None]
        if not valid:
            raise ValueError("no valid evaluations in history")

        evals = [e["evaluations"] for e in valid]
        costs = [float(e["cost"]) for e in valid]
        secs = [_parse_hms(e["timedelta_since_tuning_start"]) for e in valid]

        # pyATF already tracks the running minimum as improvement_history --
        # no need to recompute it the way the KernelTuner-cache version does.
        best_evals = [e["evaluations"] for e in improvement]
        best_costs = [float(e["cost"]) for e in improvement]
        best_secs = [_parse_hms(e["timedelta_since_tuning_start"]) for e in improvement]

        return meta, evals, costs, secs, best_evals, best_costs, best_secs

    # -------------------------------------------------------------- plot ---
    def plot01(self) -> Path:
        """Two-panel convergence figure: cost vs evaluation number (left) and
        vs wall-clock time (right), each with a best-so-far step line."""
        self._apply_style()
        meta, evals, costs, secs, best_evals, best_costs, best_secs = self._load()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        title = self.name
        technique = meta.get("search_technique", {}).get("kind")
        if technique:
            title += f"  ({technique})"
        fig.suptitle(title, fontsize=12, fontweight="bold")

        scatter_kw = dict(s=28, alpha=0.45, color=self.point_color,
                          edgecolor="none", label="each config", zorder=2)
        line_kw = dict(color=self.best_color, lw=2.6, label="best so far", zorder=3)

        # ---- (a) vs evaluation number ----
        ax1.scatter(evals, costs, **scatter_kw)
        ax1.step(best_evals, best_costs, where="post", **line_kw)
        ax1.set_xlabel("evaluation #", fontsize=12, fontweight="bold")
        ax1.set_ylabel(f"cost ({self.unit})", fontsize=12, fontweight="bold")
        ax1.set_yscale("log")
        ax1.set_title(f"Convergence vs evaluations\nbest = {min(costs):.4g} {self.unit}",
                      pad=12, fontsize=14, fontweight="bold")

        # ---- (b) vs wall-clock ----
        ax2.scatter(secs, costs, **scatter_kw)
        ax2.step(best_secs, best_costs, where="post", **line_kw)
        ax2.set_xlabel("wall-clock time (s)", fontsize=12, fontweight="bold")
        ax2.set_ylabel(f"cost ({self.unit})", fontsize=12, fontweight="bold")
        ax2.set_yscale("log")
        total_secs = secs[-1] if secs else 0.0
        ax2.set_title(f"Convergence vs wall-clock\ntotal tuning time = {total_secs:.0f} s",
                      pad=12, fontsize=14, fontweight="bold")

        for ax in (ax1, ax2):
            ax.grid(True, which="both", ls="--", alpha=0.35)
            ax.set_axisbelow(True)
            leg = ax.legend(fontsize=10, frameon=True, framealpha=0.9,
                            edgecolor="0.8", loc="upper right")
            leg.get_frame().set_linewidth(0.8)

        out = self._resolve_output_path()
        fig.savefig(out, format="png", bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {self.name}: {len(evals)} evals, best {min(costs):.4g} {self.unit}, "
              f"{total_secs:.0f} s  ->  {out.name}")
        return out


def _find_logs(inputs: List[Path]) -> List[Path]:
    found: List[Path] = []
    for p in inputs:
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("*.json")))
    # dedupe, preserve order
    seen, out = set(), []
    for f in found:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            out.append(f)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convergence plots from pyATF Tuner().log_file(...) outputs")
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="Directory to scan for *.json pyATF log files (can be given "
             "multiple times); a direct path to a log file also works.")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory (created if missing); one .png per log is "
             "written into it.")
    parser.add_argument(
        "--unit", default="ns",
        help="Cost unit shown on the y-axis and in printed summaries "
             "(default: ns -- every built-in pyATF cost function reports "
             "nanoseconds; override if yours doesn't).")
    args = parser.parse_args()

    logs = _find_logs(args.input)
    print(f"Input       : {args.input}")
    print(f"Output path : {args.output}")
    print(f"Found {len(logs)} json file(s)")

    ok = 0
    for log in logs:
        name = log.stem
        try:
            ConvergencePlotter(log, args.output, name, args.unit).plot01()
            ok += 1
        except Exception as e:
            print(f"[SKIP] {name} ({log}): {e}")
    print(f"Done: {ok}/{len(logs)} plotted")


if __name__ == "__main__":
    main()
