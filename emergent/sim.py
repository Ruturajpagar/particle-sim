"""The emergence loop.

The loop knows nothing about atoms. It only: (1) integrates Newton's
equations under the registered pair rules, (2) optionally couples to a
heat bath, and (3) hands the state to observers that *measure* structure.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from . import diagnostics as dg
from .boundaries import ReflectingBox
from .forces import DirectForceField
from .integrators import LangevinBAOAB
from .state import State


@dataclass
class Segment:
    """One stage of an experiment protocol.
    gamma = 0 -> isolated system (energy must be conserved).
    gamma > 0 -> heat bath, temperature ramped linearly t_start -> t_end."""
    name: str
    steps: int
    gamma: float
    t_start: float = 0.0
    t_end: float = 0.0


class Simulation:
    def __init__(self, state: State, field: DirectForceField, box: ReflectingBox,
                 integrator: LangevinBAOAB, log_path: str, anomaly_path: str,
                 log_every: int = 100, r_bond: float = 2.0, record: bool = False):
        self.state, self.field, self.box, self.integ = state, field, box, integrator
        self.log_every, self.r_bond, self.record = log_every, r_bond, record
        self.monitor = dg.AnomalyMonitor(anomaly_path)
        self.log_path = log_path
        self.rows: list[dict] = []
        # Trajectory frames (only when record=True), one per log row:
        # positions, bound-pair list and cluster labels, for replay/export.
        self.frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.snapshots: dict[str, np.ndarray] = {"initial": state.pos.copy()}
        self._prev_adj: np.ndarray | None = None
        self._fr = field.compute(state)
        open(anomaly_path, "w").close()

    def energy(self) -> float:
        return dg.kinetic_energy(self.state) + self._fr.potential

    def run(self, protocol: list[Segment]) -> None:
        for seg in protocol:
            self.integ.gamma = seg.gamma
            if seg.gamma == 0.0:
                self.monitor.start_isolated_segment(self.energy())
            for i in range(seg.steps):
                if seg.gamma > 0.0:
                    frac = i / max(seg.steps - 1, 1)
                    self.integ.temperature = seg.t_start + frac * (seg.t_end - seg.t_start)
                self._fr = self.integ.step(self.state, self.field, self.box, self._fr)
                self.monitor.check(self.state, self._fr, self.energy())
                if self.state.step % self.log_every == 0:
                    self._log(seg)
            self.monitor.end_isolated_segment()
            self.snapshots[seg.name] = self.state.pos.copy()
        self._write_csv()

    def _log(self, seg: Segment) -> None:
        s, fr = self.state, self._fr
        ke = dg.kinetic_energy(s)
        adj = dg.bound_pairs(s, fr, self.r_bond)
        rep = dg.analyse_clusters(s, fr, adj)
        row = {
            "step": s.step, "time": round(s.time, 4), "segment": seg.name,
            "T_bath": round(self.integ.temperature, 4) if seg.gamma > 0 else "",
            "T_kin": 2 * ke / (s.dim * s.n),
            "KE": ke, "PE": fr.potential, "E": ke + fr.potential,
            "clusters": rep.n_clusters, "free": rep.free_particles,
            "largest": rep.largest, "neutral_clusters": rep.neutral_clusters,
            "pairs_1p1e": rep.compositions.get("1e+1p", 0),
            "bind_per_particle": rep.mean_binding_per_particle,
            "size_entropy": rep.size_entropy,
            "spatial_entropy": dg.spatial_entropy(s, self.box.size),
            "bond_persistence": dg.bond_persistence(self._prev_adj, adj),
        }
        self._prev_adj = adj
        self.last_report = rep
        self.rows.append(row)
        if self.record:
            bonds = np.argwhere(np.triu(adj))
            self.frames.append((s.pos.copy(), bonds, rep.labels.copy()))

    def _write_csv(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
