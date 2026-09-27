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


def _phase_space_wall(q, p, mass, xi, hbar, alpha, mask):
    """Pair wall V = ξ²ħ²/(4αμr²) · exp(α[1 − (r p_rel / ξħ)⁴]) on the pairs in
    `mask` (symmetric bool matrix, False on the diagonal). Returns
    (V_total, dV/dq, dV/dp)."""
    mi, mj = mass[:, None], mass[None, :]
    total = mi + mj
    mu = mi * mj / total
    dq = q[:, None, :] - q[None, :, :]                        # q_i − q_j
    prel = (mj[..., None] * p[:, None, :] - mi[..., None] * p[None, :, :]) / total[..., None]
    s = np.where(mask, (dq ** 2).sum(-1), 1.0)                 # r²
    P2 = (prel ** 2).sum(-1)                                   # p_rel²

    k4 = (xi * hbar) ** 4
    C = (xi * hbar) ** 2 / (4 * alpha * mu)
    E = np.exp(alpha * (1.0 - s ** 2 * P2 ** 2 / k4))
    V = np.where(mask, C * E / s, 0.0)
    dV_ds = np.where(mask, -C * E / s ** 2 - 2 * alpha * C * E * P2 ** 2 / k4, 0.0)
    dV_dP2 = np.where(mask, -2 * alpha * C * E * s * P2 / k4, 0.0)

    grad_q = (2 * dV_ds[..., None] * dq).sum(axis=1)
    grad_p = (2 * dV_dP2[..., None] * prel * (mj / total)[..., None]).sum(axis=1)
    return 0.5 * V.sum(), grad_q, grad_p


class UncertaintyCore:
    """Momentum-dependent pair rule: no pair can have r · p_rel below ξħ.

        V = ξ²ħ² / (4 α μ r²) · exp(α [1 − (r p / ξħ)⁴])

    r is the pair distance, p the relative momentum, μ the reduced mass.
    This is the Heisenberg constraint in the classical form of Kirschbaum &
    Wilets (Phys. Rev. A 21, 834, 1980), applied to *every* pair with ξ = 1
    fixed in advance (they fitted ξ to atoms; we do not). α only sets how
    hard the phase-space wall is: it is a numerical stiffness, and results
    are checked for convergence as α grows.

    Unlike PairInteraction this depends on momenta, so it returns
    derivatives with respect to both positions and momenta and is used by
    the phase-space integrator (emergent/phasespace.py).
    """
    name = "uncertainty"

    def __init__(self, xi: float = 1.0, hbar: float = 1.0, alpha: float = 5.0):
        self.xi, self.hbar, self.alpha = xi, hbar, alpha

    def bind(self, system) -> None:
        """Called once by the system; every pair is included."""

    def evaluate(self, q: np.ndarray, p: np.ndarray, mass: np.ndarray):
        """Return (V_total, dV/dq, dV/dp) for all pairs."""
        mask = ~np.eye(len(mass), dtype=bool)
        return _phase_space_wall(q, p, mass, self.xi, self.hbar, self.alpha, mask)


class PauliCore:
    """Pauli exclusion as a pair rule: two *identical fermions with the same
    spin* cannot have r · p_rel below ξ_P ħ.

    Same wall form as UncertaintyCore, but it acts only on pairs of the same
    species, that species a fermion, with equal spin projection. It mentions
    no atoms or shells. Kirschbaum & Wilets fitted ξ_P = 2.767 to atoms; here
    ξ_P is an explicit input, and experiments scan it to test whether a single
    value can describe several systems at once (see PRINCIPLES.md).
    Since UncertaintyCore already enforces r · p ≥ ħ on every pair, ξ_P ≤ 1
    has no effect.
    """
    name = "pauli"

    def __init__(self, xi: float, hbar: float = 1.0, alpha: float = 5.0):
        self.xi, self.hbar, self.alpha = xi, hbar, alpha
        self.mask: np.ndarray | None = None

    def bind(self, system) -> None:
        kind, spin = np.asarray(system.kind), np.asarray(system.spin)
        fermion = np.array([system.species[k].fermion for k in kind])
        self.mask = ((kind[:, None] == kind[None, :]) & (spin[:, None] == spin[None, :])
                     & fermion[:, None] & ~np.eye(len(kind), dtype=bool))

    def evaluate(self, q: np.ndarray, p: np.ndarray, mass: np.ndarray):
        if self.mask is None:
            raise RuntimeError("PauliCore must be bound to a system (needs species and spins)")
        if not self.mask.any():
            return 0.0, np.zeros_like(q), np.zeros_like(p)
        return _phase_space_wall(q, p, mass, self.xi, self.hbar, self.alpha, self.mask)
