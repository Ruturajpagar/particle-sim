"""Phase 1 experiment: do neutral bound clusters emerge from bare charges?

Protocol
  1. anneal : heat bath cooled from T=2.0 to T=0.05 (lets energy leave)
  2. isolated: bath switched off; energy must be conserved, and we watch
               whether the structures that formed are stable on their own.

Usage
  python run_phase1.py                 # default 2D run
  python run_phase1.py --dim 3
  python run_phase1.py --no-core       # remove the short-range core:
                                       # watch the classical collapse
  python run_phase1.py --export viewer/data/run.json   # replay in viewer/
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from emergent import (Coulomb, CoreRepulsion, DirectForceField, LangevinBAOAB,
                      ReflectingBox, Segment, Simulation, Species, random_state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--pairs", type=int, default=100, help="number of +/- pairs")
    ap.add_argument("--box", type=float, default=None)
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--anneal-steps", type=int, default=40000)
    ap.add_argument("--isolated-steps", type=int, default=12000)
    ap.add_argument("--no-core", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/phase1")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--export", metavar="JSON", help="also write a replay file for viewer/")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    box = args.box or (30.0 if args.dim == 2 else 14.0)

    # The entire "physics" of this universe. Mass ratio 10 (not 1836) keeps
    # the fast light-particle motion resolvable at a practical dt.
    species = [Species("p", mass=10.0, charge=+1.0),
               Species("e", mass=1.0, charge=-1.0)]
    rules = [Coulomb(k=1.0)]
    if not args.no_core:
        rules.append(CoreRepulsion(c=0.125, n=8))   # p-e well minimum at r = 1

    state = random_state(species, [args.pairs, args.pairs], args.dim, box,
                         temperature=2.0, rng=rng)
    sim = Simulation(
        state, DirectForceField(rules), ReflectingBox(box),
        LangevinBAOAB(dt=args.dt, rng=rng),
        log_path=f"{args.out}/log.csv", anomaly_path=f"{args.out}/anomalies.jsonl",
        log_every=args.log_every, r_bond=2.0, record=bool(args.export),
    )
    sim.run([
        Segment("anneal", args.anneal_steps, gamma=0.5, t_start=2.0, t_end=0.05),
        Segment("isolated", args.isolated_steps, gamma=0.0),
    ])
    np.savez(f"{args.out}/final_state.npz", pos=state.pos, vel=state.vel, kind=state.kind)

    summarise(sim)
    if args.export:
        from emergent.export import export_viewer
        label = ("Core removed" if args.no_core else "Coulomb + core") + f", {args.dim}D"
        export_viewer(sim, args.export, label, rules=[type(r).__name__ for r in rules])
        print(f"replay -> {args.export}")
    try:
        from plot import make_report
        make_report(sim, f"{args.out}/report.png")
        print(f"\nplot  -> {args.out}/report.png")
    except ImportError:
        print("(matplotlib not installed; skipping plot)")


def summarise(sim: Simulation) -> None:
    rows, rep = sim.rows, sim.last_report
    first = rows[0]
    iso = [r for r in rows if r["segment"] == "isolated"]
    print("\n=== Phase 1 summary ===")
    print(f"particles: {sim.state.n}   dim: {sim.state.dim}   steps: {sim.state.step}")
    print(f"clusters (>=2):   {first['clusters']:>4}  ->  {rep.n_clusters}")
    print(f"free particles:   {first['free']:>4}  ->  {rep.free_particles}")
    print(f"largest cluster:  {first['largest']:>4}  ->  {rep.largest}")
    print(f"spatial entropy:  {first['spatial_entropy']:.3f} -> {rows[-1]['spatial_entropy']:.3f}")
    print(f"binding energy / particle in clusters: {rep.mean_binding_per_particle:.3f}")
    print("cluster sizes (size: count):", rep.sizes)
    print("top compositions:", rep.compositions.most_common(6))
    if iso:
        e = np.array([r["E"] for r in iso])
        pers = np.array([r["bond_persistence"] for r in iso], dtype=float)
        print(f"isolated phase: rel. energy drift = {abs(e[-1] - e[0]) / abs(e[0]):.2e}, "
              f"mean bond persistence per log = {np.nanmean(pers):.3f}")
    cats: dict[str, int] = {}
    for ev in sim.monitor.events:
        cats[f"{ev['category']}:{ev['kind']}"] = cats.get(f"{ev['category']}:{ev['kind']}", 0) + 1
    print("anomalies:", cats or "none")


if __name__ == "__main__":
    main()
