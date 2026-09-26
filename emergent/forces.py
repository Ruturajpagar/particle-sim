"""Force evaluation: sums every registered interaction over all pairs.

This is the O(N^2) direct-summation backend: exact and simple, and fine up
to a few thousand particles. Faster backends (cell lists, Barnes-Hut,
Ewald/PME, or a Rust/CUDA kernel) should implement the same `compute`
signature so they can be swapped in and cross-checked against this one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interactions import PairInteraction
from .state import State


@dataclass
class ForceResult:
    forces: np.ndarray        # (N, dim)
    pair_energy: np.ndarray   # (N, N) total pair potential, symmetric, 0 diagonal
    potential: float          # total potential energy
    clamped: int              # pairs closer than r_min (numerical-singularity count)


class DirectForceField:
    def __init__(self, interactions: list[PairInteraction], r_min: float = 0.05):
        self.interactions = interactions
        self.r_min = r_min

    def compute(self, state: State) -> ForceResult:
        if not self.interactions:  # free particles: skip the O(N^2) pair work
            return ForceResult(np.zeros_like(state.pos), np.zeros((state.n, state.n)), 0.0, 0)
        dx =state.pos[:, None, :] - state.pos[None, :, :]   # x_i - x_j
        r = np.sqrt((dx**2).sum(-1))
        np.fill_diagonal(r, np.inf)
        clamped = int(np.count_nonzero(r < self.r_min) // 2)
        r = np.maximum(r, self.r_min)

        u_total = np.zeros_like(r)
        f_over_r = np.zeros_like(r)
        for inter in self.interactions:
            u, f = inter.evaluate(r, state)
            u_total += u
            f_over_r += f
        np.fill_diagonal(u_total, 0.0)
        np.fill_diagonal(f_over_r, 0.0)

        forces = (f_over_r[:, :, None] * dx).sum(axis=1)
        return ForceResult(forces, u_total, 0.5 * u_total.sum(), clamped)
