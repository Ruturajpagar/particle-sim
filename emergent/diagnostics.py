"""Observables, emergent-structure detection and anomaly flags.

Nothing here feeds back into the dynamics; it only *measures*. Cluster
detection uses a physical, rule-agnostic criterion: two particles are
"bound" if their two-body energy in the centre-of-mass frame is negative,

    E_ij = 1/2 * mu_ij * |v_i - v_j|^2 + U_ij(r) < 0,   and   r < r_bond,

and clusters are connected components of that bond graph. Whatever
clusters appear were not programmed; they are discovered.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .forces import ForceResult
from .state import State


def kinetic_energy(state: State) -> float:
    return 0.5 * float((state.mass[:, None] * state.vel**2).sum())


def bound_pairs(state: State, fr: ForceResult, r_bond: float) -> np.ndarray:
    """Boolean (N, N) adjacency of energetically bound, nearby pairs."""
    m = state.mass
    mu = np.outer(m, m) / np.add.outer(m, m)
    dv2 = ((state.vel[:, None, :] - state.vel[None, :, :])**2).sum(-1)
    dx = state.pos[:, None, :] - state.pos[None, :, :]
    r = np.sqrt((dx**2).sum(-1))
    adj = (0.5 * mu * dv2 + fr.pair_energy < 0.0) & (r < r_bond)
    np.fill_diagonal(adj, False)
    return adj


def connected_components(adj: np.ndarray) -> np.ndarray:
    """Label propagation: each particle ends with the smallest id in its component."""
    labels = np.arange(len(adj))
    big = len(adj)
    while True:
        neigh_min = np.where(adj, labels[None, :], big).min(axis=1)
        new = np.minimum(labels, neigh_min)
        new = new[new]  # pointer jumping speeds convergence
        if np.array_equal(new, labels):
            return labels
        labels = new


def shannon(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@dataclass
class ClusterReport:
    labels: np.ndarray
    sizes: dict[int, int]            # cluster size -> number of clusters
    compositions: Counter            # e.g. "1e+1p" -> count
    n_clusters: int                  # clusters with >= 2 particles
    free_particles: int
    largest: int
    neutral_clusters: int
    mean_binding_per_particle: float  # over particles in clusters (negative = bound)
    size_entropy: float


def analyse_clusters(state: State, fr: ForceResult, adj: np.ndarray) -> ClusterReport:
    labels = connected_components(adj)
    uniq, counts = np.unique(labels, return_counts=True)
    sizes = Counter(counts.tolist())
    comps: Counter = Counter()
    neutral, bind_sum, bind_n = 0, 0.0, 0
    for lab, size in zip(uniq, counts):
        if size < 2:
            continue
        idx = np.flatnonzero(labels == lab)
        kinds = Counter(state.species[k].name for k in state.kind[idx])
        comps["+".join(f"{c}{name}" for name, c in sorted(kinds.items()))] += 1
        if abs(state.charge[idx].sum()) < 1e-9:
            neutral += 1
        m = state.mass[idx]
        v = state.vel[idx]
        v_com = (m[:, None] * v).sum(0) / m.sum()
        ke_int = 0.5 * float((m[:, None] * (v - v_com)**2).sum())
        pe_int = 0.5 * float(fr.pair_energy[np.ix_(idx, idx)].sum())
        bind_sum += ke_int + pe_int
        bind_n += size
    frac = counts / counts.sum()  # fraction of particles in each cluster
    return ClusterReport(
        labels=labels,
        sizes=dict(sorted(sizes.items())),
        compositions=comps,
        n_clusters=int((counts >= 2).sum()),
        free_particles=int((counts == 1).sum()),
        largest=int(counts.max()),
        neutral_clusters=neutral,
        mean_binding_per_particle=bind_sum / bind_n if bind_n else 0.0,
        size_entropy=shannon(frac),
    )


def spatial_entropy(state: State, box: float, bins: int | None = None) -> float:
    """Coarse-grained positional entropy, normalised to [0, 1].
    1 = uniform gas; lower = matter has condensed into structures.
    Default binning keeps ~5 particles per cell so a uniform gas reads ~1."""
    if bins is None:
        bins = max(2, int((state.n / 5) ** (1 / state.dim)))
    hist, _ = np.histogramdd(state.pos, bins=[bins] * state.dim,
                             range=[(0, box)] * state.dim)
    p = hist.ravel() / hist.sum()
    return shannon(p) / np.log(p.size)


def bond_persistence(prev: np.ndarray | None, now: np.ndarray) -> float:
    """Fraction of bonds present at the previous log that still exist."""
    if prev is None:
        return float("nan")
    iu = np.triu_indices(len(now), 1)
    p, n = prev[iu], now[iu]
    return float((p & n).sum() / p.sum()) if p.sum() else float("nan")


@dataclass
class AnomalyMonitor:
    """Flags events that deviate from what the rules should guarantee.

    Categories matter: `numerical` anomalies mean the *simulator* is
    wrong (fix dt/integrator); `singularity` anomalies mean the *rules*
    break down (e.g. classical collapse). Only the second kind is a
    candidate for 'new physics'.
    """
    path: str
    energy_tol: float = 1e-3
    spike_factor: float = 25.0
    events: list[dict] = field(default_factory=list)
    _e0: float | None = None
    _tol: float = 0.0
    _f_ema: float | None = None
    _singular_reported: int = 0

    def start_isolated_segment(self, energy: float) -> None:
        self._e0, self._tol = energy, self.energy_tol

    def end_isolated_segment(self) -> None:
        self._e0 = None

    def check(self, state: State, fr: ForceResult, energy: float) -> None:
        if self._e0 is not None:
            drift = abs(energy - self._e0) / max(abs(self._e0), 1e-12)
            if drift > self._tol:
                self._emit(state, "numerical", "energy_drift",
                           rel_drift=drift, hint="isolated run should conserve energy; reduce dt")
                self._tol *= 4.0  # re-flag only if it keeps growing
        if fr.clamped and self._singular_reported < 20:
            self._singular_reported += 1
            self._emit(state, "singularity", "close_approach", pairs=fr.clamped,
                       hint="particles reached r_min: the force law has no floor here")
        fmax = float(np.linalg.norm(fr.forces, axis=1).max())
        if self._f_ema is not None and fmax > self.spike_factor * self._f_ema:
            self._emit(state, "event", "force_spike", f_max=fmax, f_typical=self._f_ema)
        self._f_ema = fmax if self._f_ema is None else 0.99 * self._f_ema + 0.01 * fmax

    def _emit(self, state: State, category: str, kind: str, **data) -> None:
        ev = {"step": state.step, "time": round(state.time, 6),
              "category": category, "kind": kind, **data}
        self.events.append(ev)
        with open(self.path, "a") as f:
            f.write(json.dumps(ev) + "\n")
