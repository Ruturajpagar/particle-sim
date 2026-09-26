"""Experiment runner: config files -> parameter sweeps x seeds -> results table.

An experiment is a TOML file describing one universe (species, rules,
protocol) plus the parameters to vary. Every run is fully determined by
(config, parameter values, seed), so any row of the results table can be
reproduced exactly.

    python -m emergent.experiment experiments/atom_window.toml --workers 4

Sweeps are named parameters that may drive several config paths at once,
e.g. one final temperature that sets both the anneal target and the hold:

    [[sweep]]
    name = "T_final"
    paths = ["protocol.0.t_end", "protocol.1.t_start", "protocol.1.t_end"]
    values = [0.05, 0.1, 0.2]
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import time
import tomllib
from collections import Counter
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np

from . import diagnostics as dg
from .boundaries import OpenSpace, ReflectingBox
from .forces import DirectForceField
from .integrators import LangevinBAOAB
from .interactions import Coulomb, CoreRepulsion, PairInteraction, Yukawa
from .sim import Segment, Simulation
from .state import Species, random_state

RULES: dict[str, type[PairInteraction]] = {
    "Coulomb": Coulomb, "CoreRepulsion": CoreRepulsion, "Yukawa": Yukawa,
}


# ---------------------------------------------------------------- config

def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for key in ("name", "system", "species", "rules", "protocol"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required section '{key}'")
    return cfg


def set_path(cfg: dict, path: str, value) -> None:
    """Set a dotted path like 'rules.1.c' (list indices are integers)."""
    *parents, last = path.split(".")
    node = cfg
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if isinstance(node, list):
        node[int(last)] = value
    elif last in node:
        node[last] = value
    else:
        raise KeyError(f"sweep path '{path}': '{last}' not found in config")


@dataclass
class RunSpec:
    run_id: str
    params: dict      # sweep name -> value
    seed: int
    config: dict      # fully resolved config for this run


def expand(cfg: dict) -> list[RunSpec]:
    """Grid over all sweep parameters, times seeds. Every parameter point
    uses the same seeds (common random numbers), so differences between
    points are not masked by different initial conditions."""
    sweeps = cfg.get("sweep", [])
    names = [s["name"] for s in sweeps]
    clash = set(names) & (set(METRICS) | RESERVED)
    if clash:
        raise ValueError(f"sweep name(s) {sorted(clash)} clash with result columns; rename them")
    seeds = [cfg.get("base_seed", 0) + i for i in range(cfg.get("seeds", 1))]
    specs = []
    for combo in itertools.product(*[s["values"] for s in sweeps]):
        resolved = copy.deepcopy(cfg)
        resolved.pop("sweep", None)
        for sweep, value in zip(sweeps, combo):
            for path in sweep["paths"]:
                set_path(resolved, path, value)
        params = dict(zip(names, combo))
        tag = "_".join(f"{k}={v}" for k, v in params.items()) or "base"
        for seed in seeds:
            specs.append(RunSpec(f"{tag}_seed={seed}", params, seed, resolved))
    return specs


def scale_steps(cfg: dict, factor: float) -> dict:
    """Shorten (or lengthen) every protocol segment, e.g. for quick checks."""
    cfg = copy.deepcopy(cfg)
    for seg in cfg["protocol"]:
        seg["steps"] = max(1, int(round(seg["steps"] * factor)))
    return cfg


# ---------------------------------------------------------------- one run

def build(cfg: dict, seed: int, out_dir: str) -> tuple[Simulation, list[str]]:
    sysc = cfg["system"]
    rng = np.random.default_rng(seed)
    species = [Species(s["name"], s["mass"], s.get("charge", 0.0), s.get("strong", 0.0))
               for s in cfg["species"]]
    counts = [s["count"] for s in cfg["species"]]
    rules = []
    for r in cfg["rules"]:
        params = {k: v for k, v in r.items() if k != "type"}
        if r["type"] not in RULES:
            raise ValueError(f"unknown rule type '{r['type']}' (known: {sorted(RULES)})")
        rules.append(RULES[r["type"]](**params))
    box_size = sysc["box"]
    box = OpenSpace() if sysc.get("boundary") == "open" else ReflectingBox(box_size)
    state = random_state(species, counts, sysc["dim"], box_size,
                         temperature=cfg["protocol"][0].get("t_start", 1.0), rng=rng)
    os.makedirs(out_dir, exist_ok=True)
    sim = Simulation(state, DirectForceField(rules), box,
                     LangevinBAOAB(dt=sysc["dt"], rng=rng),
                     log_path=os.path.join(out_dir, "log.csv"),
                     anomaly_path=os.path.join(out_dir, "anomalies.jsonl"),
                     log_every=sysc.get("log_every", 200), r_bond=sysc.get("r_bond", 2.0))
    return sim, [r["type"] for r in cfg["rules"]]


def settle_free_change(rows: list[dict], n: int) -> float:
    """Change in free fraction over the second half of the last bath stage.

    Near zero = settled. Clearly negative = particles were still binding when
    the bath was switched off, so the run's outcome had not converged."""
    bath = [r for r in rows if r["segment"] != "isolated"]
    last = [r for r in bath if r["segment"] == bath[-1]["segment"]] if bath else []
    if len(last) < 3:
        return float("nan")
    return (last[-1]["free"] - last[len(last) // 2]["free"]) / n


def measure(sim: Simulation) -> dict:
    """Outcome metrics for one finished run."""
    s = sim.state
    fr = sim.field.compute(s)
    adj = dg.bound_pairs(s, fr, sim.r_bond)
    rep = dg.analyse_clusters(s, fr, adj)
    labels, counts = np.unique(rep.labels, return_counts=True)
    size_of = dict(zip(labels, counts))
    sizes = np.array([size_of[l] for l in rep.labels])
    n = s.n

    rows = sim.rows
    iso = [r for r in rows if r["segment"] == "isolated"]
    drift = (abs(iso[-1]["E"] - iso[0]["E"]) / abs(iso[0]["E"])) if len(iso) > 1 else float("nan")
    persistence = [r["bond_persistence"] for r in iso if not np.isnan(r["bond_persistence"])]
    anomalies = Counter(e["category"] for e in sim.monitor.events)
    return {
        "free_frac": rep.free_particles / n,
        "atom_frac": 2 * rep.compositions.get("1e+1p", 0) / n,
        "chain_frac": float((sizes >= 3).sum()) / n,
        "clusters": rep.n_clusters,
        "largest": rep.largest,
        "neutral_clusters": rep.neutral_clusters,
        "bind_per_particle": rep.mean_binding_per_particle,
        "T_measured": 2 * dg.kinetic_energy(s) / (s.dim * n),
        "spatial_entropy": dg.spatial_entropy(s, sim.box.size) if np.isfinite(sim.box.size) else float("nan"),
        "size_entropy": rep.size_entropy / np.log(n),
        "iso_energy_drift": drift,
        "iso_bond_persistence": float(np.mean(persistence)) if persistence else float("nan"),
        "settle_free_change": settle_free_change(rows, n),
        "anom_singularity": anomalies.get("singularity", 0),
        "anom_numerical": anomalies.get("numerical", 0),
    }


def run_one(args: tuple[RunSpec, str]) -> dict:
    spec, out_root = args
    t0 = time.perf_counter()
    sim, _ = build(spec.config, spec.seed, os.path.join(out_root, "runs", spec.run_id))
    sim.run([Segment(p["name"], p["steps"], p.get("gamma", 0.0),
                     p.get("t_start", 0.0), p.get("t_end", 0.0))
             for p in spec.config["protocol"]])
    return {"run_id": spec.run_id, **spec.params, "seed": spec.seed,
            **measure(sim), "wall_s": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------- experiment

RESERVED = {"run_id", "seed", "wall_s", "n"}
METRICS = ["free_frac", "atom_frac", "chain_frac", "clusters", "largest",
           "neutral_clusters", "bind_per_particle", "T_measured", "spatial_entropy",
           "size_entropy", "iso_energy_drift", "iso_bond_persistence",
           "settle_free_change", "anom_singularity", "anom_numerical"]


def summarise(results: list[dict], param_names: list[str]) -> list[dict]:
    """Mean, standard deviation and standard error per parameter point."""
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        groups.setdefault(tuple(r[p] for p in param_names), []).append(r)
    out = []
    for key, rs in groups.items():
        row = dict(zip(param_names, key))
        row["n"] = len(rs)
        for m in METRICS:
            v = np.array([r[m] for r in rs], dtype=float)
            v = v[~np.isnan(v)]
            row[f"{m}_mean"] = float(v.mean()) if v.size else float("nan")
            row[f"{m}_std"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
            row[f"{m}_sem"] = row[f"{m}_std"] / np.sqrt(v.size) if v.size else float("nan")
        out.append(row)
    return out


def _git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "emergent"],
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_experiment(config_path: str, out_root: str | None = None, workers: int = 1,
                   step_factor: float = 1.0, progress: bool = True) -> tuple[list[dict], list[dict]]:
    cfg = load_config(config_path)
    if step_factor != 1.0:
        cfg = scale_steps(cfg, step_factor)
    specs = expand(cfg)
    out_root = out_root or os.path.join("results", cfg["name"])
    os.makedirs(out_root, exist_ok=True)
    param_names = [s["name"] for s in cfg.get("sweep", [])]

    t0 = time.perf_counter()
    jobs = [(s, out_root) for s in specs]
    results: list[dict] = []
    if workers > 1:
        with get_context("fork").Pool(workers) as pool:
            for r in pool.imap_unordered(run_one, jobs):
                results.append(r)
                if progress:
                    print(f"[{len(results)}/{len(specs)}] {r['run_id']}  "
                          f"atoms={r['atom_frac']:.2f} chains={r['chain_frac']:.2f} ({r['wall_s']}s)",
                          flush=True)
    else:
        for job in jobs:
            results.append(run_one(job))
            if progress:
                print(f"[{len(results)}/{len(specs)}] {results[-1]['run_id']}", flush=True)
    order = {s.run_id: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: order[r["run_id"]])
    summary = summarise(results, param_names)

    _write_csv(os.path.join(out_root, "results.csv"), results)
    _write_csv(os.path.join(out_root, "summary.csv"), summary)
    with open(os.path.join(out_root, "provenance.json"), "w") as f:
        json.dump({"config_file": config_path, "config": cfg, "git_commit": _git_commit(),
                   "step_factor": step_factor, "workers": workers, "runs": len(specs),
                   "wall_s": round(time.perf_counter() - t0, 1)}, f, indent=2)
    return results, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out", help="output folder (default results/<name>)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--step-factor", type=float, default=1.0,
                    help="scale every protocol segment, e.g. 0.05 for a quick check")
    ap.add_argument("--dry-run", action="store_true", help="list runs and exit")
    args = ap.parse_args()

    if args.dry_run:
        specs = expand(load_config(args.config))
        for s in specs:
            print(s.run_id)
        print(f"{len(specs)} runs")
        return
    _, summary = run_experiment(args.config, args.out, args.workers, args.step_factor)
    print(f"\n{len(summary)} parameter points -> {args.out or 'results/<name>'}/summary.csv")


if __name__ == "__main__":
    main()
