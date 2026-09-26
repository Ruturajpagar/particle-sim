"""Particle state stored as structure-of-arrays (SoA).

Every per-particle property is a flat NumPy array indexed by particle id.
This layout vectorises well in NumPy and maps 1:1 onto a future Rust/C++
core (contiguous buffers, no per-object overhead).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Species:
    """A kind of fundamental particle. Charges are 'coupling constants'
    each interaction may read: `charge` for the EM-like force, `strong` for
    the short-range Yukawa-like force. New couplings = new fields here."""
    name: str
    mass: float
    charge: float = 0.0
    strong: float = 0.0


@dataclass
class State:
    dim: int
    species: list[Species]
    pos: np.ndarray            # (N, dim)
    vel: np.ndarray            # (N, dim)
    kind: np.ndarray           # (N,) index into `species`
    time: float = 0.0
    step: int = 0
    # Per-particle coupling arrays, derived from `kind` (cached for speed).
    mass: np.ndarray = field(init=False)
    charge: np.ndarray = field(init=False)
    strong: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.refresh_properties()

    def refresh_properties(self) -> None:
        """Call after changing `kind` (e.g. a future weak-decay rule)."""
        table = self.species
        self.mass = np.array([table[k].mass for k in self.kind], dtype=float)
        self.charge = np.array([table[k].charge for k in self.kind], dtype=float)
        self.strong = np.array([table[k].strong for k in self.kind], dtype=float)

    @property
    def n(self) -> int:
        return len(self.kind)


def random_state(species: list[Species], counts: list[int], dim: int,
                 box: float, temperature: float,
                 rng: np.random.Generator, min_sep: float = 0.8) -> State:
    """Scatter particles uniformly (rejecting overlaps) with Maxwell-Boltzmann
    velocities and zero total momentum."""
    kind = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
    n = len(kind)
    pos = np.empty((n, dim))
    placed = 0
    while placed < n:
        trial = rng.uniform(0.5, box - 0.5, size=dim)
        if placed == 0 or np.min(np.linalg.norm(pos[:placed] - trial, axis=1)) > min_sep:
            pos[placed] = trial
            placed += 1
    masses = np.array([species[k].mass for k in kind])
    vel = rng.normal(size=(n, dim)) * np.sqrt(temperature / masses)[:, None]
    vel -= (masses[:, None] * vel).sum(0) / masses.sum()  # remove COM drift
    return State(dim=dim, species=species, pos=pos, vel=vel, kind=kind)
