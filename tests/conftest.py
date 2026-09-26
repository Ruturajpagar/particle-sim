"""Shared helpers for building small, hand-specified systems."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Species, State
from emergent import diagnostics as dg

HEAVY = Species("p", mass=10.0, charge=+1.0, strong=1.0)
LIGHT = Species("e", mass=1.0, charge=-1.0)
SPECIES = [HEAVY, LIGHT]


def make_state(pos, vel=None, kinds=None, species=SPECIES) -> State:
    pos = np.asarray(pos, dtype=float)
    n, dim = pos.shape
    vel = np.zeros_like(pos) if vel is None else np.asarray(vel, dtype=float)
    kinds = np.zeros(n, int) if kinds is None else np.asarray(kinds)
    return State(dim=dim, species=species, pos=pos.copy(), vel=vel.copy(), kind=kinds)


def total_energy(state, field) -> float:
    return dg.kinetic_energy(state) + field.compute(state).potential


def momentum(state) -> np.ndarray:
    return (state.mass[:, None] * state.vel).sum(0)


def angular_momentum_2d(state) -> float:
    p = state.mass[:, None] * state.vel
    return float((state.pos[:, 0] * p[:, 1] - state.pos[:, 1] * p[:, 0]).sum())


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
