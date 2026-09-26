"""Boundary conditions.

Reflecting walls keep the system finite, so long-range Coulomb can be
summed exactly without Ewald machinery. Elastic reflection conserves
energy but not momentum, so only energy is checked for conservation.
"""
from __future__ import annotations

import numpy as np

from .state import State


class OpenSpace:
    """No boundary at all: a truly isolated system. Conserves energy,
    momentum and angular momentum, so it is the reference for validation."""
    size = float("inf")

    def apply(self, state: State) -> None:
        pass


class ReflectingBox:
    def __init__(self, size: float):
        self.size = size

    def apply(self, state: State) -> None:
        lo = state.pos < 0.0
        hi = state.pos > self.size
        state.pos = np.where(lo, -state.pos, state.pos)
        state.pos = np.where(hi, 2.0 * self.size - state.pos, state.pos)
        state.vel = np.where(lo | hi, -state.vel, state.vel)
