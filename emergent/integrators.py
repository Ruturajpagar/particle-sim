"""Time integration.

BAOAB Langevin splitting (Leimkuhler & Matthews). With gamma = 0 it is
exactly velocity Verlet: symplectic, time-reversible, with bounded
long-term energy error. That property matters here: if energy drifts in
an isolated run, it is a *numerical* anomaly, not new physics.

gamma > 0 couples the system to a heat bath at temperature T (k_B = 1).
Physically this stands in for radiation carrying energy away: two
particles cannot capture each other in an isolated two-body collision,
because something has to remove the excess energy.
"""
from __future__ import annotations

import numpy as np

from .boundaries import ReflectingBox
from .forces import DirectForceField, ForceResult
from .state import State


class LangevinBAOAB:
    def __init__(self, dt: float, gamma: float = 0.0, temperature: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.dt, self.gamma, self.temperature = dt, gamma, temperature
        self.rng = rng or np.random.default_rng()

    def step(self, state: State, field: DirectForceField, box: ReflectingBox,
             prev: ForceResult) -> ForceResult:
        dt, inv_m = self.dt, 1.0 / state.mass[:, None]

        state.vel += 0.5 * dt * prev.forces * inv_m                    # B
        state.pos += 0.5 * dt * state.vel                              # A
        box.apply(state)
        if self.gamma > 0.0:                                           # O
            c1 = np.exp(-self.gamma * dt)
            sigma = np.sqrt((1.0 - c1**2) * self.temperature * inv_m)
            state.vel = c1 * state.vel + sigma * self.rng.normal(size=state.vel.shape)
        state.pos += 0.5 * dt * state.vel                              # A
        box.apply(state)
        result = field.compute(state)
        state.vel += 0.5 * dt * result.forces * inv_m                  # B

        state.time += dt
        state.step += 1
        return result
