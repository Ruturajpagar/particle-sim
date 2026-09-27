"""Phase 4: the whole system's wave, found by diffusion, with no assumed shape.

Rules (emergent/diffusion.py), identical for every particle:
  1. every particle random-walks with diffusion constant ħ/2m
  2. configurations multiply or die with weight exp(−τ[V − E_ref]),
     V = Coulomb energy of the whole configuration
Constants: ħ, masses, charges. Every particle of every walker starts at an
independent random point; nothing is placed near anything.

Stated before running:
  * for particle sets with no two identical same-spin fermions the result is
    the exact non-relativistic ground state (after τ → 0), so it is compared
    with exact literature values for the same Hamiltonian as well as with
    measurement;
  * the walkers carry no antisymmetry, so for sets with identical same-spin
    fermions (H₂ with parallel spins, lithium) the rules give a state the
    Pauli principle forbids. These are run to show the gap, not as results.

Numerical settings: τ ∈ τ₀ × {2, 1, 0.5} / coupling², extrapolated linearly
to τ = 0; walker target 2000; each point is 4 independent chunks (seeds).

    python run_diffusion.py         # ~20 min on 4 cores
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
from emergent.diffusion import DiffusionSystem, extrapolate, run_diffusion  # noqa: E402

OUT = "results/diffusion"
HARTREE_EV = 27.211386

E = Species("e", 1.0, -1.0)
POS = Species("e+", 1.0, +1.0)
P = Species("p", 1836.15267, +1.0)
HE4 = Species("He-4", 7294.29954, +2.0, fermion=False)
LI7 = Species("Li-7", 12786.39, +3.0)

TAU0 = 0.01
N_TARGET = 2000
CHUNKS = 4
# name: (species, kinds, spins, t_equil, t_measure per chunk, τ factors)
SYSTEMS = {
    "H":                ([P, E], [0, 1], [+1, +1], 20, 330, (2, 1, 0.5)),
    "Ps":               ([POS, E], [0, 1], [+1, +1], 20, 330, (2, 1, 0.5)),
    "H-":               ([P, E], [0, 1, 1], [+1, +1, -1], 200, 190, (2, 1, 0.5)),
    "He":               ([HE4, E], [0, 1, 1], [0, +1, -1], 10, 480, (2, 1, 0.5)),
    "H2+":              ([P, E], [0, 0, 1], [+1, -1, +1], 60, 330, (2, 1, 0.5)),
    "H2":               ([P, E], [0, 0, 1, 1], [+1, -1, +1, -1], 100, 300, (2, 1, 0.5)),
    "H2 parallel spins": ([P, E], [0, 0, 1, 1], [+1, -1, +1, +1], 100, 200, (2, 1, 0.5)),
    "Li (3 electrons)": ([LI7, E], [0, 1, 1, 1], [+1, +1, -1, +1], 10, 100, (1,)),
}
# Exact non-relativistic energies for the same particles and masses (Hartree).
EXACT = {
    "H": -0.5 / (1 + 1 / 1836.15267),       # reduced-mass hydrogen
    "Ps": -0.25,
    "H-": -0.527445881,                      # Frolov, finite proton mass
    "He": -2.903304557,                      # Drake, He-4
    "H2+": -0.597139063,                     # Moss / Bishop, non-Born-Oppenheimer
    "H2": -1.164025031,                      # Pachucki & Komasa, non-Born-Oppenheimer
    "H2 parallel spins": 2 * (-0.5 / (1 + 1 / 1836.15267)),   # unbound: two H atoms
    "Li (3 electrons)": -7.477451930,        # Puchalski & Pachucki, Li-7 (a real, Pauli-obeying Li)
}
# Measured (eV)
MEASURED = {"H ionization": 13.598434, "H- electron affinity": 0.754195,
            "He binding": 24.587387 + 54.417763, "H2+ bond (D0)": 2.6507,
            "H2 bond (D0)": 4.478007, "Ps binding": 6.8028}
HIST_EDGES = np.linspace(0, 12, 241)


def chunk(job):
    name, factor, seed = job
    sp, kinds, spins, t_eq, t_meas, _ = SYSTEMS[name]
    sys_ = DiffusionSystem(sp, kinds, spins)
    tau = TAU0 * factor / sys_.coupling ** 2
    t0 = time.perf_counter()
    r = run_diffusion(sys_, tau, t_eq, t_meas, n_target=N_TARGET, seed=seed, hist_edges=HIST_EDGES)
    half = len(r.series) // 2
    drift = float(r.series[half:].mean() - r.series[:half].mean())
    return {"system": name, "tau_factor": factor, "tau": tau, "seed": seed, "E": r.energy_pc,
            "err": r.error_pc, "E_uncorrected": r.energy, "E_growth": r.growth,
            "walkers": round(r.walkers_mean, 1), "capped_fraction": r.capped_fraction,
            "half_drift": drift, "pauli_free": sys_.pauli_free,
            "wall_s": round(time.perf_counter() - t0, 1)}, r.pair_hist


def combine(rows):
    w = np.array([1 / r["err"] ** 2 for r in rows])
    e = np.array([r["E"] for r in rows])
    mean = float((w * e).sum() / w.sum())
    # spread between chunks is the honest error if it exceeds the block estimate
    sem_blocks = float(1 / np.sqrt(w.sum()))
    sem_chunks = float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else sem_blocks
    return mean, max(sem_blocks, sem_chunks)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:]
    names = [n for n in SYSTEMS if not only or n in only]
    jobs = [(n, f, 100 + c) for n in names for f in SYSTEMS[n][5] for c in range(CHUNKS)]
    # longest first keeps the 4 workers busy to the end
    cost = lambda j: SYSTEMS[j[0]][4] * DiffusionSystem(*SYSTEMS[j[0]][:3]).coupling ** 2 / j[1]
    jobs.sort(key=cost, reverse=True)
    t0 = time.perf_counter()
    rows, hists = [], {}
    with get_context("fork").Pool(4) as pool:
        for row, h in pool.imap_unordered(chunk, jobs):
            rows.append(row)
            key = (row["system"], row["tau_factor"])
            hists[key] = hists.get(key, 0) + h
            print(f"[{len(rows)}/{len(jobs)}] {row['system']} τ={row['tau']:.5f} seed {row['seed']}: "
                  f"E = {row['E']:.5f} ± {row['err']:.5f} (drift {row['half_drift']:+.4f}, "
                  f"capped {row['capped_fraction']:.1e}, {row['wall_s']} s)", flush=True)
    rows.sort(key=lambda r: (list(SYSTEMS).index(r["system"]), -r["tau_factor"], r["seed"]))
    with open(f"{OUT}/chunks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for n in names:
        pts = []
        for fac in SYSTEMS[n][5]:
            sub = [r for r in rows if r["system"] == n and r["tau_factor"] == fac]
            e, err = combine(sub)
            pts.append({"tau": sub[0]["tau"], "E": e, "err": err,
                        "capped": max(r["capped_fraction"] for r in sub)})
        taus = np.array([p["tau"] for p in pts])
        es = np.array([p["E"] for p in pts])
        errs = np.array([p["err"] for p in pts])
        if len(pts) > 1:
            e0, e0err, slope = extrapolate(taus, es, errs)
            resid = (es - (e0 + slope * taus)) / errs
            chi2 = float((resid ** 2).sum())
        else:
            e0, e0err, slope, chi2 = es[0], errs[0], float("nan"), float("nan")
        r_hist = hists[(n, min(SYSTEMS[n][5]))]
        summary[n] = {"points": pts, "E0": e0, "E0_err": e0err, "slope": slope, "chi2_linear": chi2,
                      "exact": EXACT[n], "pauli_free": DiffusionSystem(*SYSTEMS[n][:3]).pauli_free,
                      "pair_hist": r_hist.tolist()}

    E0 = {n: (summary[n]["E0"], summary[n]["E0_err"]) for n in summary}

    def diff(a, b, sa=1.0, sb=1.0):   # sa·E(a) − sb·E(b) in eV, with error
        return ((sa * E0[a][0] - sb * E0[b][0]) * HARTREE_EV,
                np.hypot(sa * E0[a][1], sb * E0[b][1]) * HARTREE_EV)

    derived = {}
    if {"H"} <= E0.keys():
        derived["H ionization"] = (-E0["H"][0] * HARTREE_EV, E0["H"][1] * HARTREE_EV)
    if {"Ps"} <= E0.keys():
        derived["Ps binding"] = (-E0["Ps"][0] * HARTREE_EV, E0["Ps"][1] * HARTREE_EV)
    if {"He"} <= E0.keys():
        derived["He binding"] = (-E0["He"][0] * HARTREE_EV, E0["He"][1] * HARTREE_EV)
    if {"H", "H-"} <= E0.keys():
        derived["H- electron affinity"] = diff("H", "H-")
    if {"H", "H2+"} <= E0.keys():
        derived["H2+ bond (D0)"] = diff("H", "H2+")
    if {"H", "H2"} <= E0.keys():
        derived["H2 bond (D0)"] = diff("H", "H2", 2, 1)
    if {"H", "H2 parallel spins"} <= E0.keys():
        derived["H2 parallel-spin bond"] = diff("H", "H2 parallel spins", 2, 1)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"git_commit": commit, "tau0": TAU0, "n_target": N_TARGET, "chunks": CHUNKS,
                   "systems": summary, "derived_eV": derived, "measured_eV": MEASURED,
                   "hist_edges": HIST_EDGES.tolist(), "wall_s": round(time.perf_counter() - t0, 1)},
                  f, indent=1)

    print(f"\n  {'system':<20} {'E(τ→0) Hartree':>22} {'exact':>12} {'diff/σ':>8} {'χ² lin':>7}")
    for n, s in summary.items():
        dev = (s["E0"] - s["exact"]) / s["E0_err"]
        tag = "" if s["pauli_free"] else "   (Pauli not imposed)"
        print(f"  {n:<20} {s['E0']:>12.5f} ± {s['E0_err']:.5f} {s['exact']:>12.5f} {dev:>8.1f} "
              f"{s['chi2_linear']:>7.2f}{tag}")
    print(f"\n  {'derived (eV)':<24} {'diffusion':>18} {'measured':>10}")
    for k, (v, e) in derived.items():
        m = MEASURED.get(k)
        print(f"  {k:<24} {v:>9.3f} ± {e:<6.3f} {m if m is not None else '≤ 0':>10}")
    print(f"\n{len(jobs)} chunks in {time.perf_counter() - t0:.0f} s -> {OUT}/")


if __name__ == "__main__":
    main()
