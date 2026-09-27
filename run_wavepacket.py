"""Phase 3: do atoms, molecules and the Pauli principle emerge when every
particle is a wave packet?

Rules (emergent/wavepacket.py), applied identically to every particle:
  1. each particle is a Gaussian wave of width s; confining it costs 3ħ²/(2ms²)
  2. Coulomb acts between the packets' charge clouds
  3. identical same-spin fermions are antisymmetric (Slater determinant)
Constants: ħ, masses, charges, spins. Nothing else, and nothing mentions
atoms: every particle starts at a random point with a random width.

Main run: all particles are packets with their real masses.
Control:  the same rules with nuclear masses ×1e9 (fixed nuclei). It is not a
          rule change; it separates two approximations: the Gaussian shape
          (present in both) and the product form for nuclei (main run only).

Same systems as run_multielectron.py, plus He₂ (should not bind), so the two
rule sets can be compared directly.

    python run_wavepacket.py        # ~5 min on 4 cores
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

from emergent import Species  # noqa: E402
from emergent.wavepacket import WavePacketSystem, ground_state  # noqa: E402

OUT = "results/wavepacket"
HARTREE_EV = 27.211386

E = Species("e", 1.0, -1.0)
P = Species("p", 1836.15267, +1.0)
HE4 = Species("He-4", 7294.29954, +2.0, fermion=False)
LI7 = Species("Li-7", 12786.39, +3.0)

SYSTEMS = {
    "H":          ([P, E], [0, 1], [+1, +1]),
    "H-":         ([P, E], [0, 1, 1], [+1, +1, -1]),
    "He":         ([HE4, E], [0, 1, 1], [0, +1, -1]),
    "Li+":        ([LI7, E], [0, 1, 1], [+1, +1, -1]),
    "Li":         ([LI7, E], [0, 1, 1, 1], [+1, +1, -1, +1]),
    "H2 singlet": ([P, E], [0, 0, 1, 1], [+1, -1, +1, -1]),
    "H2 triplet": ([P, E], [0, 0, 1, 1], [+1, -1, +1, +1]),
    "H3":         ([P, E], [0, 0, 0, 1, 1, 1], [+1, -1, +1, +1, -1, +1]),
    "He2":        ([HE4, E], [0, 0, 1, 1, 1, 1], [0, 0, +1, -1, +1, -1]),
}
MEASURED = {  # eV; total binding = sum of ionization energies (NIST)
    "H": 13.598434, "He": 24.587387 + 54.417763, "Li+": 75.64009 + 122.45436,
    "Li": 5.391715 + 75.64009 + 122.45436, "H electron affinity": 0.754195,
    "Li ionization": 5.391715, "H2 De": 4.7474, "H2 D0": 4.4781, "H2 Re": 1.4011,
}


def build(name: str, fixed_nuclei: bool) -> WavePacketSystem:
    species, kind, spin = SYSTEMS[name]
    if fixed_nuclei:
        species = [replace(s, mass=s.mass * 1e9) if s.charge > 0 else s for s in species]
    return WavePacketSystem(species, np.array(kind), np.array(spin))


def solve(args):
    name, fixed = args
    t0 = time.perf_counter()
    sys_ = build(name, fixed)
    n_starts = 32 if sys_.n >= 4 else 16
    e, x, hits, capped, energies = ground_state(sys_, n_starts=n_starts, seed=11)
    R, s = sys_.unpack(x)
    nuc = [i for i in range(sys_.n) if sys_.charge[i] > 0]
    ele = [i for i in range(sys_.n) if sys_.charge[i] < 0]
    nn = [float(np.linalg.norm(R[i] - R[j])) for k, i in enumerate(nuc) for j in nuc[k + 1:]]
    e_nuc = [float(min(np.linalg.norm(R[i] - R[j]) for j in nuc)) for i in ele]
    return {"system": name, "fixed_nuclei": fixed, "E": e, "starts": n_starts, "hits": hits,
            "capped": capped, "second_lowest": next((v for v in energies if v > e + 1e-6), None),
            "nucleus_distances": json.dumps([round(d, 4) for d in nn]),
            "electron_to_nearest_nucleus": json.dumps([round(d, 4) for d in e_nuc]),
            "electron_widths": json.dumps([round(float(s[i]), 4) for i in ele]),
            "nucleus_widths": json.dumps([round(float(s[i]), 4) for i in nuc]),
            "wall_s": round(time.perf_counter() - t0, 1)}


def derived(get) -> dict:
    ev = lambda n: -get(n)["E"] * HARTREE_EV            # binding energy in eV
    h2 = get("H2 singlet")
    return {
        "H binding": ev("H"), "He binding": ev("He"), "Li+ binding": ev("Li+"), "Li binding": ev("Li"),
        "H electron affinity": ev("H-") - ev("H"),
        "Li ionization": ev("Li") - ev("Li+"),
        "H2 bond": ev("H2 singlet") - 2 * ev("H"),
        "H2 bond length": json.loads(h2["nucleus_distances"])[0],
        "H2 parallel-spin bond": ev("H2 triplet") - 2 * ev("H"),
        "H3 beyond H2+H": ev("H3") - ev("H2 singlet") - ev("H"),
        "He2 bond": ev("He2") - 2 * ev("He"),
    }


def make_figure(main: dict, ctrl: dict, wall: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    a = ax[0]
    keys = [("H binding", "H"), ("He binding", "He"), ("Li+ binding", "Li⁺"), ("Li binding", "Li")]
    meas = [MEASURED["H"], MEASURED["He"], MEASURED["Li+"], MEASURED["Li"]]
    x = np.arange(len(keys))
    for off, src, lab, col in [(-0.27, wall, "Phase 2 wall rules", "0.7"),
                               (0.0, main, "wave packets", "C0"),
                               (0.27, ctrl, "wave packets, fixed nuclei (control)", "C9")]:
        vals = [src.get(k, np.nan) / m for (k, _), m in zip(keys, meas)]
        a.bar(x + off, vals, 0.27, color=col, label=lab)
    a.axhline(1, color="k", lw=0.8)
    a.set_xticks(x, [l for _, l in keys])
    a.set_ylim(0.6, 1.1)
    a.set_ylabel("predicted ÷ measured binding")
    a.set_title("Atoms: the Gaussian shape underbinds; the wall matched Bohr")
    a.legend(fontsize=8, loc="lower right")

    a = ax[1]
    rows = [("H2 bond", "H₂ bond\n(meas. 4.75)", MEASURED["H2 De"]),
            ("H2 parallel-spin bond", "H₂ parallel spins\n(meas. unbound)", 0.0),
            ("H3 beyond H2+H", "H₃ beyond H₂+H\n(meas. unbound)", 0.0),
            ("Li ionization", "Li ionization\n(meas. 5.39)", MEASURED["Li ionization"]),
            ("H electron affinity", "H⁻ affinity\n(meas. 0.75)", MEASURED["H electron affinity"])]
    x = np.arange(len(rows))
    for off, src, lab, col in [(-0.27, wall, "Phase 2 wall rules", "0.7"),
                               (0.0, main, "wave packets", "C0"),
                               (0.27, ctrl, "wave packets, fixed nuclei (control)", "C9")]:
        a.bar(x + off, [src.get(k, np.nan) for k, _, _ in rows], 0.27, color=col, label=lab)
    for xi, (_, _, m) in zip(x, rows):
        a.plot([xi - 0.42, xi + 0.42], [m, m], color="k", lw=2)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(x, [l for _, l, _ in rows], fontsize=8)
    a.set_ylabel("eV (positive = bound)")
    a.set_title("Molecules and Pauli: black bars are measured values")
    a.legend(fontsize=8)
    fig.suptitle("Phase 3: every particle a wave packet (ħ, masses, charges, spins only)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def wall_reference() -> dict:
    """Phase 2 wall-rule values from results/multielectron (Pauli at ξ_P = 2.767)."""
    path = "results/multielectron/results.csv"
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    get = lambda n, x: next(float(r["E_inf"]) for r in rows if r["system"] == n and float(r["xi_P"]) == x)
    ev = lambda n, x=1.0: -get(n, x) * HARTREE_EV
    return {"H binding": ev("H"), "He binding": ev("He"), "Li+ binding": ev("Li+"),
            "Li binding": ev("Li", 2.767), "H electron affinity": ev("H-") - ev("H"),
            "Li ionization": ev("Li", 2.767) - ev("Li+"), "H2 bond": ev("H2 singlet") - 2 * ev("H"),
            "H2 parallel-spin bond": ev("H2 triplet", 2.767) - 2 * ev("H"),
            "H3 beyond H2+H": ev("H3", 2.767) - ev("H2 singlet") - ev("H")}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, f) for f in (False, True) for n in SYSTEMS]
    t0 = time.perf_counter()
    rows = []
    with get_context("fork").Pool(4) as pool:
        for r in pool.imap_unordered(solve, jobs):
            rows.append(r)
            print(f"[{len(rows)}/{len(jobs)}] {r['system']}{' (fixed nuclei)' if r['fixed_nuclei'] else ''}: "
                  f"E = {r['E']:.6f}, {r['hits']}/{r['starts']} starts, capped {r['capped']} "
                  f"({r['wall_s']} s)", flush=True)
    order = {j: i for i, j in enumerate(jobs)}
    rows.sort(key=lambda r: order[(r["system"], r["fixed_nuclei"])])
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    main_d = derived(lambda n: next(r for r in rows if r["system"] == n and not r["fixed_nuclei"]))
    ctrl_d = derived(lambda n: next(r for r in rows if r["system"] == n and r["fixed_nuclei"]))
    wall_d = wall_reference()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{OUT}/provenance.json", "w") as f:
        json.dump({"git_commit": commit, "measured_eV": MEASURED, "wave_packets": main_d,
                   "wave_packets_fixed_nuclei": ctrl_d, "phase2_wall": wall_d,
                   "wall_s": round(time.perf_counter() - t0, 1)}, f, indent=2)

    meas = {"H binding": MEASURED["H"], "He binding": MEASURED["He"], "Li+ binding": MEASURED["Li+"],
            "Li binding": MEASURED["Li"], "H electron affinity": MEASURED["H electron affinity"],
            "Li ionization": MEASURED["Li ionization"], "H2 bond": MEASURED["H2 De"],
            "H2 bond length": MEASURED["H2 Re"], "H2 parallel-spin bond": "≤ 0 (unbound)",
            "H3 beyond H2+H": "≤ 0 (unbound)", "He2 bond": "≈ 0 (0.001)"}
    print(f"\n  {'quantity (eV; bond length a0)':<30} {'wall (Ph.2)':>12} {'packets':>10} {'fixed nuc.':>10} {'measured':>15}")
    for k in main_d:
        wv = wall_d.get(k)
        wtxt = f"{wv:>12.3f}" if wv is not None else f"{'':>12}"
        m = meas[k]
        mtxt = f"{m:>15.3f}" if isinstance(m, float) else f"{m:>15}"
        print(f"  {k:<30} {wtxt} {main_d[k]:>10.3f} {ctrl_d[k]:>10.3f} {mtxt}")
    make_figure(main_d, ctrl_d, wall_d, f"{OUT}/wavepacket.png")
    print(f"\n{len(jobs)} ground-state searches in {time.perf_counter() - t0:.0f} s -> {OUT}/")


if __name__ == "__main__":
    main()
