"""Phase-space engine for rules that depend on momentum.

The Phase 1 engine assumes H = Σ p²/2m + U(q), so velocity is p/m and
velocity Verlet applies. A momentum-dependent rule (UncertaintyCore) breaks
that: velocity becomes q̇ = ∂H/∂p. This module integrates Hamilton's
equations directly, in canonical positions q and momenta p:

    q̇ = ∂H/∂p
    ṗ = −∂H/∂q − γ q̇

The −γ q̇ term is friction on the actual velocity. It guarantees
dH/dt = −γ |q̇|² ≤ 0, so it only removes energy (a stand-in for radiation)
and never decides what forms. With γ = 0 the dynamics are Hamiltonian and
the energy error measures the integrator.

Position-only rules (Coulomb, ...) are reused unchanged through
DirectForceField, so physics still lives only in interactions.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .forces import DirectForceField
from .interactions import PairInteraction, UncertaintyCore
from .state import Species, State


@dataclass
class PhaseSpaceSystem:
    species: list[Species]
    kind: np.ndarray
    position_rules: list[PairInteraction]
    momentum_rules: list[UncertaintyCore] = field(default_factory=list)
    r_min: float = 1e-6

    def __post_init__(self) -> None:
        dim = 3
        self._state = State(dim=dim, species=self.species,
                            pos=np.zeros((len(self.kind), dim)),
                            vel=np.zeros((len(self.kind), dim)), kind=np.asarray(self.kind))
        self.mass = self._state.mass
        self._field = DirectForceField(self.position_rules, r_min=self.r_min)

    def _position_part(self, q: np.ndarray):
        if self._state.pos.shape != q.shape:
            self._state.pos = q.copy()
            self._state.vel = np.zeros_like(q)
            self._state.dim = q.shape[1]
        self._state.pos = q
        fr = self._field.compute(self._state)
        return fr.potential, -fr.forces, fr.clamped

    def energy(self, q: np.ndarray, p: np.ndarray) -> float:
        U, _, _ = self._position_part(q)
        kin = float((p ** 2 / (2 * self.mass[:, None])).sum())
        return kin + U + sum(r.evaluate(q, p, self.mass)[0] for r in self.momentum_rules)

    def derivatives(self, q: np.ndarray, p: np.ndarray):
        """Return (∂H/∂q, ∂H/∂p, clamped pair count)."""
        _, dU_dq, clamped = self._position_part(q)
        dH_dq = dU_dq.copy()
        dH_dp = p / self.mass[:, None]
        for rule in self.momentum_rules:
            _, gq, gp = rule.evaluate(q, p, self.mass)
            dH_dq += gq
            dH_dp += gp
        return dH_dq, dH_dp, clamped


@dataclass
class Trajectory:
    t: np.ndarray
    q: np.ndarray        # (frames, N, dim)
    p: np.ndarray
    energy: np.ndarray
    clamped: int         # steps on which some pair hit r_min


def integrate(system: PhaseSpaceSystem, q0: np.ndarray, p0: np.ndarray, dt: float,
              steps: int, gamma: float = 0.0, record_every: int = 100) -> Trajectory:
    """Classical RK4 on Hamilton's equations with velocity friction γ."""
    q, p = q0.astype(float).copy(), p0.astype(float).copy()
    clamped = 0

    def rhs(q, p):
        dHq, dHp, c = system.derivatives(q, p)
        return dHp, -dHq - gamma * dHp, c

    ts, qs, ps, es = [0.0], [q.copy()], [p.copy()], [system.energy(q, p)]
    for step in range(1, steps + 1):
        k1q, k1p, c1 = rhs(q, p)
        k2q, k2p, _ = rhs(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
        k3q, k3p, _ = rhs(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
        k4q, k4p, _ = rhs(q + dt * k3q, p + dt * k3p)
        q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        clamped += c1 > 0
        if step % record_every == 0:
            ts.append(step * dt)
            qs.append(q.copy())
            ps.append(p.copy())
            es.append(system.energy(q, p))
    return Trajectory(np.array(ts), np.array(qs), np.array(ps), np.array(es), clamped)
