"""Plot a finished experiment's summary.csv.

    python plot_experiment.py results/atom_window

Each panel shows one metric against the first sweep parameter, one line per
value of the second, with error bars = standard error over seeds.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("atom_frac", "Particles in isolated p+e pairs"),
    ("chain_frac", "Particles in clusters of 3+"),
    ("free_frac", "Free particles"),
    ("bind_per_particle", "Binding energy / particle in clusters"),
    ("iso_bond_persistence", "Bond persistence with bath off"),
    ("iso_energy_drift", "Energy drift with bath off (relative)"),
]


def main(folder: str) -> None:
    prov = json.load(open(os.path.join(folder, "provenance.json")))
    sweeps = [s["name"] for s in prov["config"].get("sweep", [])]
    if not 1 <= len(sweeps) <= 2:
        raise SystemExit("plot_experiment.py handles 1- or 2-parameter sweeps")
    rows = list(csv.DictReader(open(os.path.join(folder, "summary.csv"))))
    x_name, line_name = sweeps[0], (sweeps[1] if len(sweeps) == 2 else None)
    lines = sorted({float(r[line_name]) for r in rows}) if line_name else [None]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (metric, title) in zip(axes.flat, PANELS):
        for i, lv in enumerate(lines):
            sel = [r for r in rows if line_name is None or float(r[line_name]) == lv]
            sel.sort(key=lambda r: float(r[x_name]))
            x = [float(r[x_name]) for r in sel]
            y = [float(r[f"{metric}_mean"]) for r in sel]
            e = [float(r[f"{metric}_sem"]) for r in sel]
            label = f"{line_name} = {lv}" if line_name else None
            ax.errorbar(x, y, yerr=e, marker="o", ms=4, capsize=3, lw=1.6,
                        color=f"C{i}", label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_name)
        if metric == "iso_energy_drift":
            ax.set_yscale("log")
        elif metric.endswith("_frac") or metric == "iso_bond_persistence":
            ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    if line_name:
        axes.flat[0].legend(fontsize=9)
    n = int(rows[0]["n"])
    fig.suptitle(f"{prov['config']['name']}: {prov['runs']} runs, {n} seeds per point, "
                 f"error bars = s.e.m.   (commit {prov['git_commit'][:8]})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(folder, "summary.png")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/atom_window")
