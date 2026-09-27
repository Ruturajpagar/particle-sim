"""Ground states of phase-space systems by direct energy minimization.

With friction, the phase-space dynamics settle at a minimum of H(q, p)
(that is where q̇ = ∂H/∂p = 0 and ṗ = −∂H/∂q = 0). Searching for those
minima directly with L-BFGS from many random starts finds the same states
far faster than integrating the dynamics. `tests/test_groundstate.py` checks
that both routes agree.

Wall stiffness α is raised in steps (continuation): each start is minimized
at the smallest α, and its result seeds the next α. This keeps the search in
the physically allowed region of phase space as the wall hardens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .phasespace import PhaseSpaceSystem


def _pack(q, p):
    return np.concatenate([q.ravel(), p.ravel()])


def minimize_energy(system: PhaseSpaceSystem, q0: np.ndarray, p0: np.ndarray,
                    gtol: float = 1e-9, maxiter: int = 20000):
    """Local minimum of H(q, p) from (q0, p0). Returns (E, q, p, success)."""
    shape = q0.shape
    n = q0.size

    def f(x):
        q, p = x[:n].reshape(shape), x[n:].reshape(shape)
        dq, dp, _ = system.derivatives(q, p)
        return system.energy(q, p), _pack(dq, dp)

    res = minimize(f, _pack(q0, p0), jac=True, method="L-BFGS-B",
                   options={"gtol": gtol, "ftol": 1e-16, "maxiter": maxiter, "maxcor": 30})
    q, p = res.x[:n].reshape(shape), res.x[n:].reshape(shape)
    return float(res.fun), q, p, bool(res.success)


def set_stiffness(system: PhaseSpaceSystem, alpha: float) -> None:
    for rule in system.momentum_rules:
        rule.alpha = alpha


@dataclass
class GroundState:
    alphas: list[float]
    energy: dict[float, float]          # lowest energy found at each α
    hits: dict[float, int]              # starts that reached it (within tol)
    starts: int
    q: dict[float, np.ndarray]
    p: dict[float, np.ndarray]
    extent: dict[float, float]          # largest particle distance from the centre

    def extrapolated(self) -> float:
        """Two-point extrapolation to an infinitely hard wall, linear in 1/α."""
        a1, a2 = self.alphas[-2], self.alphas[-1]
        e1, e2 = self.energy[a1], self.energy[a2]
        return (a2 * e2 - a1 * e1) / (a2 - a1)


def find_ground_state(system: PhaseSpaceSystem, start: Callable[[np.random.Generator], tuple],
                      n_starts: int = 24, alphas=(5.0, 10.0, 20.0, 40.0), seed: int = 0,
                      tol: float = 1e-6) -> GroundState:
    rng = np.random.default_rng(seed)
    alphas = list(alphas)
    results = {a: [] for a in alphas}
    for _ in range(n_starts):
        q, p = start(rng)
        for a in alphas:
            set_stiffness(system, a)
            e, q, p, _ = minimize_energy(system, q, p)
            if np.isfinite(e):
                results[a].append((e, q.copy(), p.copy()))
    gs = GroundState(alphas, {}, {}, n_starts, {}, {}, {})
    for a in alphas:
        e, q, p = min(results[a], key=lambda t: t[0])
        gs.energy[a] = e
        gs.hits[a] = sum(abs(r[0] - e) < tol * max(1.0, abs(e)) for r in results[a])
        gs.q[a], gs.p[a] = q, p
        centre = (system.mass[:, None] * q).sum(0) / system.mass.sum()
        gs.extent[a] = float(np.linalg.norm(q - centre, axis=1).max())
    return gs
