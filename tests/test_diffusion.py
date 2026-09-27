"""Diffusion rules: free diffusion, exact potential, a known ground state."""
from __future__ import annotations

import numpy as np
import pytest

from emergent import Species
from emergent.diffusion import DiffusionSystem, extrapolate, random_walkers, run_diffusion

E = Species("e", 1.0, -1.0)
P = Species("p", 1836.15267, +1.0)
HEAVY = Species("heavy+1", 1e12, +1.0)


def test_potential_is_the_pairwise_coulomb_sum():
    rng = np.random.default_rng(0)
    s = DiffusionSystem([P, E], [0, 0, 1, 1], [1, -1, 1, -1])
    X = rng.normal(size=(5, 4, 3))
    q = s.charge
    direct = [sum(q[i] * q[j] / np.linalg.norm(x[i] - x[j]) for i in range(4) for j in range(i + 1, 4))
              for x in X]
    assert np.allclose(s.potential(X), direct)


def test_free_particles_spread_with_variance_hbar_t_over_m():
    neutral = [Species("a", 1.0, 0.0), Species("b", 4.0, 0.0)]
    s = DiffusionSystem(neutral, [0, 1], [0, 0])
    X0 = np.zeros((4000, 2, 3))
    X0[:, 1, 0] = 5.0
    r = run_diffusion(s, tau=0.01, t_equil=1.0, t_measure=0.01, n_target=4000, seed=1, X0=X0)
    assert len(r.X) == 4000                       # V = 0: no branching at all
    var = (r.X - X0).var(axis=(0, 2))
    assert var == pytest.approx([1.0, 0.25], rel=0.05)


def test_hydrogen_ground_state_emerges_from_random_starts():
    s = DiffusionSystem([HEAVY, E], [0, 1], [0, 1])
    r = run_diffusion(s, tau=0.01, t_equil=10, t_measure=40, n_target=1000, seed=3)
    assert r.energy == pytest.approx(-0.5, abs=0.02)
    assert r.growth == pytest.approx(-0.5, abs=0.02)
    assert 900 < r.walkers_mean < 1100


def test_pauli_flag_marks_identical_same_spin_fermions():
    assert DiffusionSystem([P, E], [0, 0, 1, 1], [1, -1, 1, -1]).pauli_free
    assert not DiffusionSystem([P, E], [0, 0, 1, 1], [1, -1, 1, 1]).pauli_free
    boson = Species("b", 7294.3, 2.0, fermion=False)
    assert DiffusionSystem([boson, E], [0, 0, 1], [0, 0, 1]).pauli_free


def test_random_walkers_treat_every_particle_alike():
    s = DiffusionSystem([P, E], [0, 1, 1], [1, 1, -1])
    X = random_walkers(s, 20000, np.random.default_rng(0), radius=2.0)
    r = np.linalg.norm(X, axis=-1)
    assert r.max() <= 2.0
    assert np.allclose(r.mean(0), r.mean(0)[0], atol=0.02)   # same spread for nucleus and electrons


def test_extrapolation_recovers_the_intercept_of_a_line():
    taus = np.array([0.02, 0.01, 0.005])
    e0, err, slope = extrapolate(taus, -0.5 + 0.3 * taus, np.full(3, 1e-3))
    assert e0 == pytest.approx(-0.5, abs=1e-12)
    assert slope == pytest.approx(0.3)
