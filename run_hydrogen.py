"""Does real hydrogen emerge from fundamental rules alone?

One proton and one electron in atomic units (ħ = e = m_e = 1, real mass
ratio 1836.15). No rule mentions atoms. Friction on velocity stands in for
radiation: it only removes energy.

  A  Coulomb only                          -> classical collapse expected
  B  Coulomb + UncertaintyCore (ξ = 1)     -> what does the pair settle into?
     for wall stiffness α = 2, 5, 10, 20 and 8 random starts each
  C  stability: kick a settled pair, switch friction off, watch
  D  numerical check: halve dt

Real hydrogen, for comparison after the fact: ground-state energy
−0.49973 Hartree (−13.598 eV, reduced mass included), Bohr radius
1.0005 a0 with reduced mass (0.5295 Å).

    python run_hydrogen.py            # ~2 min on 4 cores
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from multiprocessing import get_context

import numpy as np

from emergent import Coulomb, Species
from emergent.interactions import UncertaintyCore
from emergent.phasespace import PhaseSpaceSystem, integrate

OUT = "results/hydrogen_emergence"
PROTON = Species("p", mass=1836.15267, charge=+1.0)
ELECTRON = Species("e", mass=1.0, charge=-1.0)
MU = PROTON.mass / (PROTON.mass + ELECTRON.mass)
HARTREE_EV, BOHR_A = 27.211386, 0.529177
REAL_E, REAL_R = -MU / 2, 1 / MU          # exact nonrelativistic hydrogen, reduced mass
GAMMA, DT, T_END = 0.3, 0.002, 150.0


def system(alpha: float | None) -> PhaseSpaceSystem:
    momentum_rules = [UncertaintyCore(xi=1.0, hbar=1.0, alpha=alpha)] if alpha else []
    return PhaseSpaceSystem([PROTON, ELECTRON], np.array([0, 1]), [Coulomb(k=1.0)],
                            momentum_rules, r_min=1e-4)


def random_start(seed: int):
    """Electron 3-8 a0 away, random direction and momentum, initial pair
    energy between −0.15 and +0.02 Hartree (loosely bound to slightly free)."""
    rng = np.random.default_rng(seed)
    r0 = rng.uniform(3, 8)
    e0 = rng.uniform(-0.15, 0.02)
    e0 = max(e0, -0.9 / r0)  # keep the kinetic energy positive
    pmag = np.sqrt(2 * MU * (e0 + 1 / r0))
    rhat = rng.normal(size=3); rhat /= np.linalg.norm(rhat)
    phat = rng.normal(size=3); phat /= np.linalg.norm(phat)
    q = np.array([np.zeros(3), r0 * rhat])
    p = np.array([-pmag * phat, pmag * phat])
    return q, p, r0, e0


def pair_state(q, p):
    r = np.linalg.norm(q[1] - q[0])
    prel = (PROTON.mass * p[1] - ELECTRON.mass * p[0]) / (PROTON.mass + ELECTRON.mass)
    return r, np.linalg.norm(prel)


def relative_energy(sys_, q, p):
    """Energy in the centre-of-mass frame (friction also slows the COM)."""
    P = p.sum(0)
    return sys_.energy(q, p) - float(P @ P) / (2 * sys_.mass.sum())


def run_case(args):
    label, alpha, seed, dt, t_end = args
    sys_ = system(alpha)
    q, p, r0, e0 = random_start(seed)
    t0 = time.perf_counter()
    times, rs, es, us = [], [], [], []
    # Friction can only remove energy, so any rise means the integrator has
    # broken down (a close pass it cannot resolve). Check often enough to see it.
    chunk = 500 if alpha else 20
    t, collapsed_at, breakdown, e_prev, r_min = 0.0, None, None, np.inf, np.inf
    while t < t_end - 1e-9:
        tr = integrate(sys_, q, p, dt, chunk, gamma=GAMMA, record_every=chunk)
        q, p = tr.q[-1], tr.p[-1]
        t += chunk * dt
        r, pr = pair_state(q, p)
        e = relative_energy(sys_, q, p)
        r_min = min(r_min, r)
        times.append(t); rs.append(r); es.append(e); us.append(r * pr)
        if not np.isfinite(e) or e > e_prev + 1e-6 * max(1.0, abs(e_prev)):
            breakdown = t
            break
        if r < 1e-2:
            collapsed_at = t
            break
        e_prev = e
    return {"label": label, "alpha": alpha or 0.0, "seed": seed, "dt": dt,
            "r0": r0, "E0": e0, "collapsed_at": collapsed_at, "breakdown_at": breakdown,
            "r_min_seen": r_min, "E_last_good": e_prev if np.isfinite(e_prev) else es[-1],
            "E_final": es[-1], "r_final": rs[-1], "rp_final": us[-1],
            "t_final": t, "wall_s": round(time.perf_counter() - t0, 1),
            "series": {"t": times, "r": rs, "E": es, "rp": us}}


def stability_check(settled_q, settled_p, alpha, seed=0):
    """Kick the settled electron by ~10% of its momentum, run without friction."""
    rng = np.random.default_rng(seed)
    sys_ = system(alpha)
    q, p = settled_q.copy(), settled_p.copy()
    kick = 0.1 * np.linalg.norm(p[1]) * rng.normal(size=3) / np.sqrt(3)
    p[1] += kick
    p[0] -= kick
    tr = integrate(sys_, q, p, DT, 50000, gamma=0.0, record_every=100)  # 100 a.u.
    r = np.linalg.norm(tr.q[:, 1] - tr.q[:, 0], axis=1)
    drift = abs(tr.energy[-1] - tr.energy[0]) / abs(tr.energy[0])
    return {"alpha": alpha, "r_min": float(r.min()), "r_max": float(r.max()),
            "E_kicked": float(tr.energy[0]), "rel_energy_drift": float(drift)}


def settle(alpha, seed, t_end=T_END):
    sys_ = system(alpha)
    q, p, _, _ = random_start(seed)
    tr = integrate(sys_, q, p, DT, int(t_end / DT), gamma=GAMMA, record_every=int(t_end / DT))
    return tr.q[-1], tr.p[-1]


def make_figure(results: list[dict], alphas: list[float], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    col = {2.0: "C0", 5.0: "C1", 10.0: "C2", 20.0: "C3"}
    for r in results:
        s = r["series"]
        if r["label"] == "coulomb_only":
            n = len(s["t"]) - (1 if r["breakdown_at"] else 0)   # drop the broken frame
            ax[0, 0].plot(s["t"][:n], s["r"][:n], color="k", lw=1, alpha=0.8)
            ax[0, 1].plot(s["t"][:n], s["E"][:n], color="k", lw=1, alpha=0.8)
        elif r["label"] == "uncertainty":
            c = col[r["alpha"]]
            ax[0, 0].plot(s["t"], s["r"], color=c, lw=0.8, alpha=0.6)
            ax[0, 1].plot(s["t"], s["E"], color=c, lw=0.8, alpha=0.6)
            ax[1, 1].plot(s["t"], s["rp"], color=c, lw=0.8, alpha=0.6)
    for a in alphas:
        ax[0, 0].plot([], [], color=col[a], label=f"Coulomb + uncertainty, α = {a:g}")
    ax[0, 0].plot([], [], color="k", label="Coulomb only (until numerics break)")
    ax[0, 0].axhline(REAL_R, color="grey", ls="--", lw=1)
    ax[0, 0].set(yscale="log", xlabel="time (a.u.)", ylabel="proton–electron distance (a0)",
                 title="Pair distance: collapse vs settling")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(REAL_E, color="grey", ls="--", lw=1, label="real hydrogen −0.4997")
    ax[0, 1].set(ylim=(-1.6, 0.1), xlabel="time (a.u.)", ylabel="pair energy (Hartree)",
                 title="Energy (Coulomb-only runs fall below the plot)")
    ax[0, 1].legend(fontsize=8)

    inv = np.linspace(0, 0.55, 100)
    ax[1, 0].plot(inv, -MU / (2 + inv), color="0.5", lw=1, label="analytic minimum −μ/(2 + 1/α)")
    for a in alphas:
        E = [r["E_final"] for r in results if r["label"] == "uncertainty" and r["alpha"] == a]
        ax[1, 0].plot(np.full(len(E), 1 / a), E, "o", color=col[a], ms=5)
    ax[1, 0].plot([0], [REAL_E], "k*", ms=12, label="real hydrogen (α → ∞ limit)")
    ax[1, 0].set(xlabel="1 / α (wall softness)", ylabel="settled energy (Hartree)",
                 title="Settled energy of 8 random starts per α")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].axhline(1.0, color="grey", ls="--", lw=1)
    ax[1, 1].set(ylim=(0, 3), xlabel="time (a.u.)", ylabel="r · p  (units of ħ)",
                 title="Phase-space product settles at exactly ħ")
    fig.suptitle("Hydrogen from fundamental rules: 1 proton + 1 electron, atomic units, "
                 "no atom-specific rule", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(8))
    alphas = [2.0, 5.0, 10.0, 20.0]
    jobs = [("coulomb_only", None, s, DT, T_END) for s in seeds[:4]]
    jobs += [("uncertainty", a, s, DT, T_END) for a in alphas for s in seeds]
    jobs += [("uncertainty_half_dt", 20.0, s, DT / 2, T_END) for s in seeds[:2]]
    t0 = time.perf_counter()
    with get_context("fork").Pool(4) as pool:
        results = pool.map(run_case, jobs)

    q_set, p_set = settle(10.0, seed=3)
    stab = stability_check(q_set, p_set, 10.0)

    rows = [{k: v for k, v in r.items() if k != "series"} for r in results]
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{OUT}/series.json", "w") as f:
        json.dump([{k: r[k] for k in ("label", "alpha", "seed", "series")} for r in results], f)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{OUT}/provenance.json", "w") as f:
        json.dump({"git_commit": commit, "gamma": GAMMA, "dt": DT, "t_end": T_END,
                   "proton_mass": PROTON.mass, "stability": stab,
                   "wall_s": round(time.perf_counter() - t0, 1)}, f, indent=2)

    print("\n=== A: Coulomb only ===")
    for r in rows:
        if r["label"] == "coulomb_only":
            why = (f"reached r < 0.01 a0 at t = {r['collapsed_at']:.2f}" if r["collapsed_at"]
                   else f"integrator broke down (energy rose) at t = {r['breakdown_at']:.2f}"
                   if r["breakdown_at"] else "no collapse seen")
            print(f"  seed {r['seed']}: {why}; smallest r seen {r['r_min_seen']:.3g} a0, "
                  f"last trustworthy E = {r['E_last_good']:.3g} Hartree")
    print("\n=== B: Coulomb + uncertainty (ξ = 1) ===")
    print(f"  {'α':>4} | {'E final (Hartree)':>22} {'analytic':>9} | {'r final (a0)':>17} {'analytic':>8} | r·p")
    for a in alphas:
        sel = [r for r in rows if r["label"] == "uncertainty" and r["alpha"] == a]
        E = np.array([r["E_final"] for r in sel]); R = np.array([r["r_final"] for r in sel])
        U = np.array([r["rp_final"] for r in sel])
        print(f"  {a:>4.0f} | {E.mean():>12.6f} ± {E.std():.1e} {-MU / (2 + 1 / a):>9.5f} | "
              f"{R.mean():>8.5f} ± {R.std():.1e} {(1 + 1 / (2 * a)) / MU:>8.5f} | {U.mean():.5f}")
    print(f"  real hydrogen: E = {REAL_E:.5f} Hartree ({REAL_E * HARTREE_EV:.3f} eV), r = {REAL_R:.5f} a0")
    half = [r for r in rows if r["label"] == "uncertainty_half_dt"]
    full = {r["seed"]: r for r in rows if r["label"] == "uncertainty" and r["alpha"] == 20.0}
    print("\n=== D: dt check (α = 20) ===")
    for r in half:
        print(f"  seed {r['seed']}: E(dt) = {full[r['seed']]['E_final']:.8f}, E(dt/2) = {r['E_final']:.8f}")
    print("\n=== C: stability after a 10% kick, no friction, 100 a.u. ===")
    print(f"  r stays in [{stab['r_min']:.3f}, {stab['r_max']:.3f}] a0, "
          f"relative energy drift {stab['rel_energy_drift']:.1e}")
    make_figure(results, alphas, f"{OUT}/hydrogen.png")
    print(f"\n{len(jobs)} runs in {time.perf_counter() - t0:.0f} s -> {OUT}/")


if __name__ == "__main__":
    main()
