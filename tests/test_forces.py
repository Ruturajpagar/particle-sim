"""Every force law must be the exact negative gradient of its potential.

This is the test that protects new rules: a sign error or a wrong
derivative in a new PairInteraction fails here immediately.
"""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Coulomb, CoreRepulsion, DirectForceField, Yukawa

from .conftest import make_state

RULESETS = {
    "coulomb": [Coulomb()],
    "core": [CoreRepulsion()],
    "yukawa": [Yukawa(lam=0.7)],
    "all": [Coulomb(), CoreRepulsion(), Yukawa(lam=0.7)],
}


def random_config(rng, n, dim, min_sep=0.7):
    pos = []
    while len(pos) < n:
        p = rng.uniform(0, 4, size=dim)
        if all(np.linalg.norm(p - q) > min_sep for q in pos):
            pos.append(p)
    return np.array(pos), rng.integers(0, 2, size=n)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("name", list(RULESETS))
def test_force_is_negative_gradient_of_potential(name, dim, rng):
    pos, kinds = random_config(rng, 6, dim)
    field = DirectForceField(RULESETS[name])
    state = make_state(pos, kinds=kinds)
    forces = field.compute(state).forces

    h = 1e-6
    numeric = np.zeros_like(pos)
    for i in range(len(pos)):
        for d in range(dim):
            for sign in (+1, -1):
                state.pos = pos.copy()
                state.pos[i, d] += sign * h
                numeric[i, d] -= sign * field.compute(state).potential / (2 * h)

    scale = np.abs(forces).max()
    assert np.allclose(forces, numeric, atol=1e-6 * scale, rtol=1e-6)


@pytest.mark.parametrize("name", list(RULESETS))
def test_pair_forces_obey_newtons_third_law(name, rng):
    pos, kinds = random_config(rng, 8, 3)
    result = DirectForceField(RULESETS[name]).compute(make_state(pos, kinds=kinds))
    scale = np.abs(result.forces).max()
    assert np.allclose(result.forces.sum(0), 0.0, atol=1e-12 * scale * len(pos))
    assert np.allclose(result.pair_energy, result.pair_energy.T)
    assert np.all(np.diag(result.pair_energy) == 0.0)


def test_coulomb_signs():
    # opposite charges attract, like charges repel
    field = DirectForceField([Coulomb()])
    opp = field.compute(make_state([[0, 0], [2, 0]], kinds=[0, 1]))
    like = field.compute(make_state([[0, 0], [2, 0]], kinds=[0, 0]))
    assert opp.forces[0, 0] > 0 and opp.potential == pytest.approx(-0.5)
    assert like.forces[0, 0] < 0 and like.potential == pytest.approx(+0.5)


def test_core_places_pair_minimum_at_r_equals_one():
    # U(r) = -1/r + (1/8)/r^8 has dU/dr = 0 at r = 1, depth -0.875
    field = DirectForceField([Coulomb(), CoreRepulsion(c=0.125, n=8)])
    at_min = field.compute(make_state([[0, 0], [1, 0]], kinds=[0, 1]))
    assert np.allclose(at_min.forces, 0.0, atol=1e-12)
    assert at_min.potential == pytest.approx(-0.875)


def test_close_approaches_are_counted_not_hidden():
    field = DirectForceField([Coulomb()], r_min=0.05)
    result = field.compute(make_state([[0, 0], [0.01, 0], [5, 5]], kinds=[0, 1, 1]))
    assert result.clamped == 1
    assert np.all(np.isfinite(result.forces))
