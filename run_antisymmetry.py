"""Phase 5: diffusion plus the antisymmetry rule (fixed node from the packets).

Rules, identical for every particle:
  1. every particle random-walks with diffusion constant ħ/2m
  2. configurations multiply or die with weight exp(−τ[V − E_ref])
  3. exchanging identical same-spin fermions flips the wave's sign: walkers
     keep the sign of their region and are removed at the node
Constants: ħ, masses, charges, spins.

The node is the one the Phase 3 wave-packet rules find by themselves from
random starts (emergent/wavepacket.py, same systems, 16 starts). Nothing
about atoms or shells is supplied. A node that is wrong can only raise the
energy, so every value here is an upper bound on the exact one.

Stated before running:
  * He with parallel spins, Li and Be have identical same-spin electrons.
    Phase 4 (no antisymmetry) put Li 30 eV too low; with the rule the
    energies must lie above exact values, never below.
  * He⁺ and Li⁺ need no node; they are run with the same code to give
    ionization energies from the simulation's own numbers.

Walkers are guided (emergent/guide.py) by the same packet wave plus the
exact pair cusp factors. For a fixed node the guide cannot change the
energy, only the noise; unguided walkers were too noisy for Li and Be
(results/antisymmetry/FINDINGS.md).

Numerical settings: τ ∈ τ₀ × {4, 2, 1} / coupling², extrapolated linearly;
1000 walkers with the population-control correction; 4 independent chunks
per point; centre of mass removed (exact).

    python run_antisymmetry.py      # ~80 min on 4 cores
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

from emergent import Species  # noqa: E402
from emergent.diffusion import DiffusionSystem, extrapolate  # noqa: E402
from emergent.guide import PacketGuide, run_guided  # noqa: E402
from emergent.wavepacket import WavePacketSystem, ground_state  # noqa: E402

OUT = "results/antisymmetry"
HARTREE_EV = 27.211386

E = Species("e", 1.0, -1.0)
HE4 = Species("He-4", 7294.29954, +2.0, fermion=False)
LI7 = Species("Li-7", 12786.39, +3.0)
BE9 = Species("Be-9", 16424.2, +4.0)

TAU0 = 0.01
N_TARGET = 1000
CHUNKS = 4
# name: (species, kinds, spins, t_equil, t_measure per chunk)
SYSTEMS = {
    "He+":              ([HE4, E], [0, 1], [0, +1], 10, 150),
    "He parallel spins": ([HE4, E], [0, 1, 1], [0, +1, +1], 20, 200),
    "Li+":              ([LI7, E], [0, 1, 1], [+1, +1, -1], 20, 200),
    "Li":               ([LI7, E], [0, 1, 1, 1], [+1, +1, -1, +1], 30, 200),
    "Be":               ([BE9, E], [0, 1, 1, 1, 1], [+1, +1, -1, +1, -1], 30, 100),
}
FACTORS = (4, 2, 1)
# Exact non-relativistic energies (Hartree). Where only the infinite-mass
# value is published, it is scaled by the reduced-mass factor (normal mass
# shift only; the remaining specific mass shift is ≲ 1e-4 Hartree).
mu = lambda M: 1 / (1 + 1 / M)
EXACT = {
    "He+": -2.0 * mu(7294.29954),
    "He parallel spins": -2.175229378 * mu(7294.29954),   # 2³S, Drake
    "Li+": -7.279913413 * mu(12786.39),
    "Li": -7.477451930,                                   # Puchalski & Pachucki, Li-7
    "Be": -14.667356498 * mu(16424.2),                    # Pachucki & Komasa
}
MEASURED = {  # eV (NIST)
    "He parallel-spin ionization": 4.767750,
    "Li ionization": 5.391715,
    "Li binding": 5.391715 + 75.64009 + 122.45436,
    "Be binding": 9.32270 + 18.21115 + 153.8962 + 217.7186,
}
PHASE4_LI = -8.6511                                       # results/diffusion (corrected rerun): no antisymmetry


_nodes: dict = {}


def packet_node(name: str):
    """Node from the Phase 3 rules for this particle set (cached per process)."""
    if name not in _nodes:
        sp, kinds, spins = SYSTEMS[name][:3]
        wp = WavePacketSystem(sp, kinds, spins)
        e, x, hits, capped, _ = ground_state(wp, n_starts=16, seed=11)
        _nodes[name] = (PacketGuide(wp, x), e, hits, x)
    return _nodes[name]


def chunk(job):
    name, factor, seed = job
    sp, kinds, spins, t_eq, t_meas = SYSTEMS[name]
    sys_ = DiffusionSystem(sp, kinds, spins)
    guide, e_wp, _, _ = packet_node(name)
    tau = TAU0 * factor / sys_.coupling ** 2
    t0 = time.perf_counter()
    r = run_guided(sys_, guide, tau, t_eq, t_meas, n_target=N_TARGET, seed=seed)
    return {"system": name, "tau_factor": factor, "tau": tau, "seed": seed,
            "E": r.energy, "err": r.error, "E_uncorrected": r.energy_raw,
            "walkers": round(r.walkers_mean, 1), "acceptance": round(r.acceptance, 4),
            "capped_fraction": r.capped_fraction, "has_node": guide.has_node,
            "wall_s": round(time.perf_counter() - t0, 1)}


def combine(rows):
    w = np.array([1 / r["err"] ** 2 for r in rows])
    e = np.array([r["E"] for r in rows])
    mean = float((w * e).sum() / w.sum())
    sem_chunks = float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else 0.0
    return mean, max(float(1 / np.sqrt(w.sum())), sem_chunks)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    names = [n for n in SYSTEMS if len(sys.argv) < 2 or n in sys.argv[1:]]
    # the packet node for each set, found once here (also recorded)
    packets = {}
    for n in names:
        node, e_wp, hits, x = packet_node(n)
        R, s = WavePacketSystem(*SYSTEMS[n][:3]).unpack(x)
        packets[n] = {"E_packets": e_wp, "hits_of_16": hits,
                      "node_groups": sum(len(g) > 1 for g in node.groups),
                      "centres": np.round(R - R[0], 4).tolist(), "widths": np.round(s, 4).tolist()}
        print(f"packet node for {n}: E = {e_wp:.5f}, {hits}/16 starts, widths {np.round(s, 3)}", flush=True)

    jobs = [(n, f, 100 + c) for n in names for f in FACTORS for c in range(CHUNKS)]
    cost = lambda j: SYSTEMS[j[0]][4] * DiffusionSystem(*SYSTEMS[j[0]][:3]).coupling ** 2 \
        * len(SYSTEMS[j[0]][1]) ** 2 / j[1]
    jobs.sort(key=cost, reverse=True)
    t0 = time.perf_counter()
    rows = []
    with get_context("fork").Pool(4) as pool:
        for row in pool.imap_unordered(chunk, jobs):
            rows.append(row)
            print(f"[{len(rows)}/{len(jobs)}] {row['system']} τ={row['tau']:.5f} seed {row['seed']}: "
                  f"E = {row['E']:.5f} ± {row['err']:.5f} (acceptance {row['acceptance']:.3f}, "
                  f"{row['wall_s']} s)", flush=True)
    rows.sort(key=lambda r: (list(SYSTEMS).index(r["system"]), -r["tau_factor"], r["seed"]))
    with open(f"{OUT}/chunks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for n in names:
        pts = []
        for fac in FACTORS:
            sub = [r for r in rows if r["system"] == n and r["tau_factor"] == fac]
            e, err = combine(sub)
            pts.append({"tau": sub[0]["tau"], "E": e, "err": err})
        taus = np.array([p["tau"] for p in pts])
        es = np.array([p["E"] for p in pts])
        errs = np.array([p["err"] for p in pts])
        e0, e0err, slope = extrapolate(taus, es, errs)
        chi2 = float((((es - (e0 + slope * taus)) / errs) ** 2).sum())
        summary[n] = {"points": pts, "E0": e0, "E0_err": e0err, "slope": slope, "chi2_linear": chi2,
                      "exact": EXACT[n], **packets[n]}

    E0 = {n: (s["E0"], s["E0_err"]) for n, s in summary.items()}
    ev = lambda a, b: ((E0[a][0] - E0[b][0]) * HARTREE_EV, np.hypot(E0[a][1], E0[b][1]) * HARTREE_EV)
    derived = {}
    if {"He+", "He parallel spins"} <= E0.keys():
        derived["He parallel-spin ionization"] = ev("He+", "He parallel spins")
    if {"Li+", "Li"} <= E0.keys():
        derived["Li ionization"] = ev("Li+", "Li")
    if "Li" in E0:
        derived["Li binding"] = (-E0["Li"][0] * HARTREE_EV, E0["Li"][1] * HARTREE_EV)
    if "Be" in E0:
        derived["Be binding"] = (-E0["Be"][0] * HARTREE_EV, E0["Be"][1] * HARTREE_EV)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"git_commit": commit, "tau0": TAU0, "n_target": N_TARGET, "chunks": CHUNKS,
                   "systems": summary, "derived_eV": derived, "measured_eV": MEASURED,
                   "phase4_li_no_antisymmetry": PHASE4_LI,
                   "wall_s": round(time.perf_counter() - t0, 1)}, f, indent=1)

    print(f"\n  {'system':<20} {'packets':>10} {'E(τ→0) Hartree':>22} {'exact':>11} {'above exact':>12}")
    for n, s in summary.items():
        print(f"  {n:<20} {s['E_packets']:>10.4f} {s['E0']:>12.5f} ± {s['E0_err']:.5f} {s['exact']:>11.5f} "
              f"{(s['E0'] - s['exact']) * HARTREE_EV:>+8.3f} eV")
    print(f"\n  {'derived (eV)':<30} {'simulation':>18} {'measured':>10}")
    for k, (v, e) in derived.items():
        print(f"  {k:<30} {v:>9.3f} ± {e:<6.3f} {MEASURED[k]:>10.3f}")
    print(f"\n{len(jobs)} chunks in {time.perf_counter() - t0:.0f} s -> {OUT}/")


if __name__ == "__main__":
    main()
