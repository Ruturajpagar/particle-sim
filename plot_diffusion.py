"""Figure for results/diffusion: τ extrapolation and emergent pair distances.

    python plot_diffusion.py [results/diffusion]
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HARTREE_EV = 27.211386


def main(out: str = "results/diffusion") -> None:
    d = json.load(open(f"{out}/summary.json"))
    systems = d["systems"]
    edges = np.array(d["hist_edges"])
    mids = 0.5 * (edges[1:] + edges[:-1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for i, (name, s) in enumerate(systems.items()):
        if len(s["points"]) < 2 or not s["pauli_free"]:
            continue                # Pauli-violating runs are shown in the other panels
        taus = np.array([p["tau"] for p in s["points"]])
        rel = taus / taus.max()
        es = np.array([p["E"] for p in s["points"]]) - s["exact"]
        errs = np.array([p["err"] for p in s["points"]])
        line = ax.errorbar(rel, es * HARTREE_EV, errs * HARTREE_EV, fmt="o", capsize=3, label=name)
        c = line[0].get_color()
        x = np.linspace(0, 1, 20)
        ax.plot(x, (s["E0"] - s["exact"] + s["slope"] * x * taus.max()) * HARTREE_EV, "-", color=c)
        ax.errorbar([0], [(s["E0"] - s["exact"]) * HARTREE_EV], [s["E0_err"] * HARTREE_EV],
                    fmt="s", color=c, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("τ / largest τ")
    ax.set_ylabel("E − exact (eV)")
    ax.set_title("vs exact, Pauli-free sets (squares: τ → 0)")
    ax.legend(fontsize=7)

    def pair_panel(ax, name, pairs, title):
        s = systems.get(name)
        if s is None:
            return
        h = np.array(s["pair_hist"])
        for idx, label in pairs:
            dens = h[idx] / (h[idx].sum() * np.diff(edges))
            ax.plot(mids, dens, label=label)
        ax.set_xlabel("distance (a0)")
        ax.set_ylabel("walker density (∝ ψ₀, not |ψ₀|²)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # pair order is numpy.triu_indices: for [p, p, e, e] -> (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
    pair_panel(axes[1], "H2", [(0, "p–p"), (1, "p–e"), (5, "e–e")],
               "H₂: two protons settle at a fixed distance")
    if "H2 parallel spins" in systems:
        h = np.array(systems["H2 parallel spins"]["pair_hist"])
        axes[1].plot(mids, h[0] / (h[0].sum() * np.diff(edges)), ":", color="C0",
                     label="p–p, parallel spins (no Pauli)")
        axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 6)
    # [p, e, e] -> (0,1) (0,2) (1,2)
    pair_panel(axes[2], "H-", [(0, "p–e"), (2, "e–e")], "H⁻: the second electron stays bound")
    axes[2].set_xlim(0, 12)
    fig.tight_layout()
    fig.savefig(f"{out}/diffusion.png", dpi=105)
    print(f"wrote {out}/diffusion.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
