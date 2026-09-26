"""Fundamental pair interactions -- the *only* place physics rules live.

Each interaction is a central pair potential U(r). It returns, for all
pairs at once, the pair energy U_ij and the quantity -U'(r)/r, from which
the force on i due to j is  F_ij = (-U'(r)/r) * (x_i - x_j).

To add a new rule, subclass `PairInteraction` and implement `evaluate`.
Nothing else in the engine needs to change, and no rule anywhere mentions
atoms, bonds or molecules: those must emerge.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .state import State


class PairInteraction(ABC):
    name: str = "pair"

    @abstractmethod
    def evaluate(self, r: np.ndarray, state: State) -> tuple[np.ndarray, np.ndarray]:
        """r: (N, N) pair distances with the diagonal set to +inf.
        Returns (U, f_over_r), both (N, N) and symmetric."""


class Coulomb(PairInteraction):
    """EM-like long-range force: U = k q_i q_j / r (like charges repel)."""
    name = "coulomb"

    def __init__(self, k: float = 1.0):
        self.k = k

    def evaluate(self, r, state):
        qq = self.k * np.outer(state.charge, state.charge)
        u = qq / r
        return u, qq / r**3


class CoreRepulsion(PairInteraction):
    """Short-range hard core: U = c / r^n, applied to every pair.

    Physically this stands in for what classical point charges lack
    (Pauli exclusion / quantum zero-point motion). Without it, opposite
    charges spiral into r -> 0: the classical collapse that forced physics
    to invent quantum mechanics. Try running with it disabled.
    """
    name = "core"

    def __init__(self, c: float = 0.125, n: int = 8):
        self.c, self.n = c, n

    def evaluate(self, r, state):
        u = self.c / r**self.n
        return u, self.n * u / r**2


class Yukawa(PairInteraction):
    """Short-range 'strong-like' attraction from a massive mediator:
    U = -g_i g_j exp(-r/lam) / r. Only particles with nonzero `strong`
    coupling feel it. (Yukawa's 1935 model of the nuclear force.)"""
    name = "yukawa"

    def __init__(self, lam: float = 0.5):
        self.lam = lam

    def evaluate(self, r, state):
        gg = np.outer(state.strong, state.strong)
        e = np.exp(-r / self.lam)
        u = -gg * e / r
        dudr = gg * e * (1.0 / r**2 + 1.0 / (self.lam * r))
        return u, -dudr / r
