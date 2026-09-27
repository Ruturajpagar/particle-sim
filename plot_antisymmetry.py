"""Figure for results/antisymmetry: energies vs exact per rule set, and τ extrapolation.

    python plot_antisymmetry.py [results/antisymmetry]
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HARTREE_EV = 27.211386


def main(out: str = "results/antisymmetry") -> None:
    d = json.load(open(f"{out}/summary.json"))
    systems = d["systems"]
    names = list(systems)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(names))
    packets = [(systems[n]["E_packets"] - systems[n]["exact"]) * HARTREE_EV for n in names]
    walk = [(systems[n]["E0"] - systems[n]["exact"]) * HARTREE_EV for n in names]
    walk_err = [systems[n]["E0_err"] * HARTREE_EV for n in names]
    ax.bar(x - 0.27, packets, 0.27, label="Phase 3: wave packets (source of the node)", color="#9aa5b1")
    ax.bar(x, walk, 0.27, yerr=walk_err, capsize=3, label="Phase 5: diffusion + antisymmetry", color="#2f6fb0")
    if "Li" in names:
        i = names.index("Li")
        ax.bar(i + 0.27, (d["phase4_li_no_antisymmetry"] - systems["Li"]["exact"]) * HARTREE_EV, 0.27,
               label="Phase 4: diffusion, no antisymmetry", color="#c8553d")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("E − exact (eV)")
    ax.set_title("energy relative to the exact value, per rule set")
    ax.legend(fontsize=8)

    for n in names:
        s = systems[n]
        taus = np.array([p["tau"] for p in s["points"]])
        rel = taus / taus.max()
        es = (np.array([p["E"] for p in s["points"]]) - s["exact"]) * HARTREE_EV
        errs = np.array([p["err"] for p in s["points"]]) * HARTREE_EV
        line = bx.errorbar(rel, es, errs, fmt="o", capsize=3, label=n)
        c = line[0].get_color()
        xx = np.linspace(0, 1, 20)
        bx.plot(xx, (s["E0"] - s["exact"] + s["slope"] * xx * taus.max()) * HARTREE_EV, color=c)
        bx.errorbar([0], [(s["E0"] - s["exact"]) * HARTREE_EV], [s["E0_err"] * HARTREE_EV],
                    fmt="s", color=c, capsize=3)
    bx.axhline(0, color="k", lw=0.8)
    bx.set_xlabel("τ / largest τ")
    bx.set_ylabel("E − exact (eV)")
    bx.set_title("Phase 5 time-step extrapolation (squares: τ → 0)")
    bx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/antisymmetry.png", dpi=105)
    print(f"wrote {out}/antisymmetry.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
