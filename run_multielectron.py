"""Beyond one electron: what do the fundamental rules predict, and is the
Pauli rule's strength universal?

Rules: Coulomb (every pair), UncertaintyCore (r·p ≥ ħ, every pair, ξ = 1),
PauliCore (r·p ≥ ξ_P ħ, identical same-spin fermions only). Atomic units,
real nuclear masses. No rule mentions atoms, shells or bonds.

Part 1, no free constants (no same-spin electrons, so Pauli cannot act):
    H, H⁻, He, Li⁺ and H₂ with opposite electron spins.
Part 2, the Pauli map: ξ_P is the rule's one constant and cannot be fixed
    from first principles in this classical form (see PRINCIPLES.md). Rather
    than fit it to one system, scan it and ask whether a *single* value
    gets Li, H₂ with parallel spins and H₃ all right at once.

Ground states are the lowest minima of H(q, p) over many random starts
(emergent/groundstate.py), at wall stiffness α = 5, 10, 20, 40, then
extrapolated to α → ∞.

    python run_multielectron.py     # ~10 min on 4 cores
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from multiprocessing import get_context

import numpy as np

from emergent import Coulomb, PauliCore, Species, UncertaintyCore
from emergent.groundstate import find_ground_state
from emergent.phasespace import PhaseSpaceSystem

OUT = "results/multielectron"
HARTREE_EV, BOHR_A = 27.211386, 0.529177
ALPHAS = (5.0, 10.0, 20.0, 40.0)
XI_P = [1.0, 1.5, 2.0, 2.5, 2.767, 3.0, 3.5, 4.0]   # 2.767: Kirschbaum-Wilets fitted value

E = Species("e", 1.0, -1.0)
P = Species("p", 1836.15267, +1.0)                         # proton, spin 1/2
HE4 = Species("He-4", 7294.29954, +2.0, fermion=False)     # alpha particle, spin 0
LI7 = Species("Li-7", 12786.39, +3.0)                      # spin 3/2

# name: (species list, kind, spin); nuclei first
SYSTEMS = {
    "H":            ([P, E], [0, 1], [+1, +1]),
    "H-":           ([P, E], [0, 1, 1], [+1, +1, -1]),
    "He":           ([HE4, E], [0, 1, 1], [0, +1, -1]),
    "Li+":          ([LI7, E], [0, 1, 1], [+1, +1, -1]),
    "Li":           ([LI7, E], [0, 1, 1, 1], [+1, +1, -1, +1]),
    "H2 singlet":   ([P, E], [0, 0, 1, 1], [+1, -1, +1, -1]),
    "H2 triplet":   ([P, E], [0, 0, 1, 1], [+1, -1, +1, +1]),
    "H3":           ([P, E], [0, 0, 0, 1, 1, 1], [+1, -1, +1, +1, -1, +1]),
}
USES_PAULI = {"Li", "H2 triplet", "H3"}

# Measured values (eV). Total binding = sum of ionization energies (NIST).
MEASURED = {
    "H": 13.598434, "He": 24.587387 + 54.417763, "Li+": 75.64009 + 122.45436,
    "Li": 5.391715 + 75.64009 + 122.45436,
    "H electron affinity": 0.754195,       # E(H) − E(H⁻)
    "Li ionization": 5.391715,             # E(Li⁺) − E(Li)
    "H2 De": 4.7474, "H2 D0": 4.4781,      # bond energy without / with nuclear zero-point
    "H2 Re (a0)": 1.4011,
}


def build(name: str, xi_p: float) -> PhaseSpaceSystem:
    species, kind, spin = SYSTEMS[name]
    rules = [UncertaintyCore(xi=1.0)]
    if xi_p > 1.0:
        rules.append(PauliCore(xi=xi_p))
    return PhaseSpaceSystem(species, np.array(kind), [Coulomb()], rules, spin=np.array(spin))


def make_start(name: str):
    species, kind, _ = SYSTEMS[name]
    kind = np.array(kind)
    nuclei = np.flatnonzero([species[k].charge > 0 for k in kind])
    electrons = np.flatnonzero([species[k].charge < 0 for k in kind])

    def start(rng):
        q = np.zeros((len(kind), 3))
        p = np.zeros((len(kind), 3))
        if len(nuclei) > 1:
            q[nuclei] = rng.normal(size=(len(nuclei), 3)) * rng.uniform(0.5, 1.5)
        for i in electrons:
            host = rng.choice(nuclei)
            z = species[kind[host]].charge
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            q[i] = q[host] + d * rng.uniform(0.3, 2.0) / z
            v = rng.normal(size=3); v /= np.linalg.norm(v)
            p[i] = v * z * rng.uniform(0.3, 1.5)
        p[nuclei] -= p.sum(0) / len(nuclei)
        return q, p
    return start


def solve(args):
    name, xi_p = args
    t0 = time.perf_counter()
    n_starts = 32 if name.startswith("H2") or name == "H3" else 16
    sys_ = build(name, xi_p)
    gs = find_ground_state(sys_, make_start(name), n_starts=n_starts, alphas=ALPHAS, seed=7)
    a = ALPHAS[-1]
    q = gs.q[a]
    species, kind, _ = SYSTEMS[name]
    nuclei = [i for i, k in enumerate(kind) if species[k].charge > 0]
    electrons = [i for i, k in enumerate(kind) if species[k].charge < 0]
    nn = [float(np.linalg.norm(q[i] - q[j])) for n, i in enumerate(nuclei) for j in nuclei[n + 1:]]
    e_nuc = sorted(float(min(np.linalg.norm(q[i] - q[j]) for j in nuclei)) for i in electrons)
    return {"system": name, "xi_P": xi_p, "starts": n_starts,
            **{f"E_a{int(x)}": gs.energy[x] for x in ALPHAS},
            **{f"hits_a{int(x)}": gs.hits[x] for x in ALPHAS},
            "E_inf": gs.extrapolated(), "extent": gs.extent[a],
            "nucleus_distances": json.dumps([round(d, 4) for d in nn]),
            "electron_to_nearest_nucleus": json.dumps([round(d, 4) for d in e_nuc]),
            "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, 1.0) for n in SYSTEMS if n not in USES_PAULI]
    jobs += [(n, x) for n in SYSTEMS if n in USES_PAULI for x in XI_P]
    t0 = time.perf_counter()
    with get_context("fork").Pool(4) as pool:
        rows = pool.map(solve, jobs)
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{OUT}/provenance.json", "w") as f:
        json.dump({"git_commit": commit, "alphas": ALPHAS, "xi_P": XI_P, "measured_eV": MEASURED,
                   "wall_s": round(time.perf_counter() - t0, 1)}, f, indent=2)

    get = lambda n, x=1.0: next(r for r in rows if r["system"] == n and r["xi_P"] == x)
    ev = lambda h: -h * HARTREE_EV
    eH, eHm, eHe = get("H")["E_inf"], get("H-")["E_inf"], get("He")["E_inf"]
    eLip, eH2 = get("Li+")["E_inf"], get("H2 singlet")["E_inf"]
    h2 = get("H2 singlet")
    print("=== Part 1: no free constants ===")
    print(f"  {'quantity':<28} {'predicted':>10} {'measured':>10} {'error':>8}")
    for label, pred, meas in [
        ("H total binding (eV)", ev(eH), MEASURED["H"]),
        ("He total binding (eV)", ev(eHe), MEASURED["He"]),
        ("Li+ total binding (eV)", ev(eLip), MEASURED["Li+"]),
        ("H electron affinity (eV)", ev(eHm) - ev(eH), MEASURED["H electron affinity"]),
        ("H2 bond energy (eV)", ev(eH2) - 2 * ev(eH), MEASURED["H2 De"]),
    ]:
        print(f"  {label:<28} {pred:>10.3f} {meas:>10.3f} {100 * (pred - meas) / meas:>+7.1f}%")
    print(f"  H2 bond length: {json.loads(h2['nucleus_distances'])[0]:.3f} a0 "
          f"(measured {MEASURED['H2 Re (a0)']})")

    print("\n=== Part 2: Pauli map (which ξ_P, if any, fits all three?) ===")
    print(f"  {'ξ_P':>6} | {'Li ionization eV':>16} | {'H2 triplet bond eV':>18} | {'H3 − (H2+H) eV':>15} | Li electron radii (a0)")
    for x in XI_P:
        li, trip, h3 = get("Li", x), get("H2 triplet", x), get("H3", x)
        print(f"  {x:>6.3f} | {ev(li['E_inf']) - ev(eLip):>16.3f} | "
              f"{ev(trip['E_inf']) - 2 * ev(eH):>18.3f} | {ev(eH2) + ev(eH) - ev(h3['E_inf']):>15.3f} | "
              f"{li['electron_to_nearest_nucleus']}")
    print(f"  measured:  Li ionization {MEASURED['Li ionization']:.3f} eV; H2 triplet unbound (≤ 0); "
          f"H3 unbound (≥ 0)")
    print(f"\n{len(jobs)} ground-state searches in {time.perf_counter() - t0:.0f} s -> {OUT}/")


if __name__ == "__main__":
    main()
