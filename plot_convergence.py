"""Two-sided convergence plot for experiments with an `init` sweep.

    python plot_convergence.py results/equilibrium_check

One row per parameter point (every sweep except `init`), three columns
(atoms, chains, free). Each panel shows the fraction over time from the
free start and the paired start: mean over seeds, shaded band = min to max.
If the two curves meet, the shared value is the equilibrium answer.

Clusters of two are always one + and one − particle (like charges repel and
never bind), so chains = 1 − free − atoms.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"random": ("C0", "free start"), "pairs": ("C3", "paired start")}


def load(folder: str):
    prov = json.load(open(os.path.join(folder, "provenance.json")))
    n = sum(s["count"] for s in prov["config"]["species"])
    point_keys = [s["name"] for s in prov["config"]["sweep"] if s["name"] != "init"]
    series = defaultdict(lambda: defaultdict(list))  # point -> init -> [runs]
    for r in csv.DictReader(open(os.path.join(folder, "results.csv"))):
        rows = list(csv.DictReader(open(os.path.join(folder, "runs", r["run_id"], "log.csv"))))
        t = np.array([float(x["time"]) for x in rows])
        free = np.array([float(x["free"]) for x in rows]) / n
        atoms = np.array([2 * float(x["pairs_1p1e"]) for x in rows]) / n
        point = tuple(r[k] for k in point_keys)
        series[point][r["init"]].append((t, atoms, 1 - free - atoms, free))
    iso_start = None
    steps = [p["steps"] * prov["config"]["system"]["dt"] for p in prov["config"]["protocol"]]
    if len(steps) > 1:
        iso_start = sum(steps[:-1])
    return prov, point_keys, series, iso_start


def main(folder: str) -> None:
    prov, keys, series, iso_start = load(folder)
    points = sorted(series, key=lambda p: tuple(float(v) for v in p))
    fig, axes = plt.subplots(len(points), 3, figsize=(14, 3.1 * len(points)), squeeze=False)
    for row, point in zip(axes, points):
        label = ", ".join(f"{k} = {v}" for k, v in zip(keys, point))
        for col, (ax, title) in enumerate(zip(row, ["atoms (isolated p+e)", "chains (3+)", "free"])):
            for init, runs in sorted(series[point].items()):
                color, name = STYLE.get(init, ("C2", init))
                t = runs[0][0]
                ys = np.array([r[col + 1] for r in runs])
                ax.plot(t, ys.mean(0), color=color, lw=1.6, label=f"{name} (n={len(runs)})")
                ax.fill_between(t, ys.min(0), ys.max(0), color=color, alpha=0.18, lw=0)
            if iso_start is not None:
                ax.axvline(iso_start, color="0.5", ls=":", lw=1)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            ax.set_title(f"{title} | {label}", fontsize=10)
        row[0].set_ylabel("fraction of particles")
    axes[0, 0].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("time" + ("   (dotted line: bath off)" if iso_start else ""))
    fig.suptitle(f"{prov['config']['name']}: does each point converge from both sides?  "
                 f"(band = min to max over seeds, commit {prov['git_commit'][:8]})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(folder, "convergence.png")
    fig.savefig(out, dpi=100)
    print(out)


def endpoint_table(folder: str) -> list[dict]:
    """Final fractions per point and start, averaged over seeds."""
    _, keys, series, _ = load(folder)
    out = []
    for point, by_init in sorted(series.items(), key=lambda kv: tuple(float(v) for v in kv[0])):
        for init, runs in sorted(by_init.items()):
            end = np.array([[r[1][-1], r[2][-1], r[3][-1]] for r in runs])
            out.append({**dict(zip(keys, point)), "init": init, "n": len(runs),
                        "atoms": float(end[:, 0].mean()), "atoms_sd": float(end[:, 0].std(ddof=1)),
                        "chains": float(end[:, 1].mean()), "free": float(end[:, 2].mean())})
    return out


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/equilibrium_check"
    main(folder)
    for r in endpoint_table(folder):
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
