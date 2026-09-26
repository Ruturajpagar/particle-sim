"""Diagnostic dashboard for a finished run (one PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from emergent import diagnostics as dg


def make_report(sim, path: str) -> None:
    rows = sim.rows
    t = np.array([r["time"] for r in rows])
    col = lambda k: np.array([np.nan if r[k] == "" else r[k] for r in rows], dtype=float)
    t_iso = next((r["time"] for r in rows if r["segment"] == "isolated"), None)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Phase 1 — emergent clustering from bare charges", fontsize=14)

    def mark(a):
        if t_iso is not None:
            a.axvline(t_iso, color="grey", ls=":", lw=1)
            a.text(t_iso, a.get_ylim()[1], " bath off", va="top", fontsize=8, color="grey")

    a = ax[0, 0]
    a.plot(t, col("KE"), label="kinetic")
    a.plot(t, col("PE"), label="potential")
    a.plot(t, col("E"), label="total", lw=2)
    a.set_title("Energy"); a.legend(); mark(a)

    a = ax[0, 1]
    a.plot(t, col("T_kin"), label="measured T")
    a.plot(t, col("T_bath"), "--", label="bath T")
    a.set_title("Temperature"); a.legend(); mark(a)

    a = ax[0, 2]
    a.plot(t, col("free"), label="free particles")
    a.plot(t, col("clusters"), label="clusters (≥2)")
    a.plot(t, col("pairs_1p1e"), label="p+e pairs ('atoms')")
    a.plot(t, col("largest"), label="largest cluster size")
    a.set_title("Emergent structure"); a.legend(fontsize=8); mark(a)

    a = ax[1, 0]
    a.plot(t, col("spatial_entropy"), label="spatial (normalised)")
    a.plot(t, col("size_entropy") / np.log(sim.state.n), label="cluster-size (normalised)")
    a.set_title("Entropy proxies"); a.legend(); mark(a)

    a = ax[1, 1]
    a.plot(t, col("bind_per_particle"), label="binding energy / particle")
    a.set_ylabel("energy")
    b = a.twinx()
    b.plot(t, col("bond_persistence"), color="C3", alpha=0.6, label="bond persistence")
    b.set_ylim(0, 1.05); b.set_ylabel("persistence")
    a.set_title("Cluster stability")
    a.legend(loc="upper left", fontsize=8); b.legend(loc="lower left", fontsize=8); mark(a)

    a = ax[1, 2]
    s = sim.state
    adj = dg.bound_pairs(s, sim._fr, sim.r_bond)
    i, j = np.nonzero(np.triu(adj))
    for u, v in zip(i, j):
        a.plot(s.pos[[u, v], 0], s.pos[[u, v], 1], color="0.6", lw=0.8, zorder=1)
    pos = s.kind == 0
    a.scatter(s.pos[pos, 0], s.pos[pos, 1], s=40, c="C3", label="+ (heavy)", zorder=2)
    a.scatter(s.pos[~pos, 0], s.pos[~pos, 1], s=14, c="C0", label="− (light)", zorder=3)
    a.set_aspect("equal"); a.set_xlim(0, sim.box.size); a.set_ylim(0, sim.box.size)
    a.set_title("Final state" + (" (x-y projection)" if s.dim == 3 else "") + " — lines = bound pairs")
    a.legend(fontsize=8, loc="upper right")

    for a in ax.flat[:5]:
        a.set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
